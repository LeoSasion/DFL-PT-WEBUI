"""TAE engine implementing the existing ME training and merger contracts."""
from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import numpy as np
import torch
from me_backend.engine import resolve_devices, validate_scaler_state
from me_backend.tensor_checks import all_finite, finite_unit_interval, gradients_are_finite, scalar_metrics
from .checkpoint import FORMAT, VERSION, atomic_save, read_checkpoint, rng_state, restore_rng
from .config import TrainConfig
from .losses import reconstruction_loss
from .masking import image_loss_mask
from .model import LIAEModel


class TAEEngine:
    model_class = "TAE"
    checkpoint_name = "tae.pt"
    checkpoint_format = FORMAT
    supports_backup = False
    supports_evaluation = False

    def __init__(self, config, device="cuda", seed=0, weights=None, vae=None, *, training=True):
        if type(training) is not bool:
            raise ValueError('training must be a boolean')
        self._inference_only = not training
        self.config = config.validate()
        self.training_config = config
        self.device, self.device_ids = resolve_devices(device)
        if len(self.device_ids) > 1:
            raise ValueError("TAE currently supports one GPU per training process")
        if training:
            torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self._validate_precision(config)
        # Checkpoint inference replaces initialized head/decoder weights. Keep
        # construction local to its RNG scope instead of resetting caller RNGs.
        with nullcontext() if training else torch.random.fork_rng(devices=self.device_ids):
            if vae is None:
                from .vae import TAEQI21Codec
                vae = TAEQI21Codec(weights, device=self.device, decoder_mode=config.decoder_mode)
            vae.set_decoder_mode(config.decoder_mode)
            vae.compute_mode = self._compute_mode(config)
            self.vae_weights = getattr(vae, "weights_path", str(Path(weights).resolve()) if weights else None)
            self.vae_sha256 = getattr(vae, "weight_sha256", None)
            self.vae_contract = vae.contract()
            self.network = LIAEModel(vae, config.inter_dims, config.hidden_dims, config.resolution).to(self.device)
        self.iteration = self.optimizer_updates = self.stage_updates = self.stage_index = 0
        self.stage_history = [self._stage_record()]
        self.data_state = None
        self.training_mode = "identity"
        self.pretraining_data = None
        self._make_optimizer(training=training)
        if not training:
            self.network.eval()
        self._frozen_versions = self._versions()

    def _stage_record(self):
        return {"index": self.stage_index, "started_at_iteration": self.iteration,
                "precision": self.config.precision,
                "batch_size": self.config.batch_size,
                "ct_mode": self.config.ct_mode,
                "random_warp": self.config.random_warp,
                "learning_rate": self.config.learning_rate,
                "eyes_mouth_weight": self.config.eyes_mouth_weight,
                "training_mask_mode": self.config.training_mask_mode,
                "decoder_mode": self.config.decoder_mode,
                "decoder_lr_multiplier": self.config.decoder_lr_multiplier}

    def _validate_precision(self, config):
        if self.device.type == "cuda" and config.precision == "bf16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("This GPU does not support BF16; select FP32")

    def _compute_mode(self, config):
        return "fp32-master-bf16-autocast" if self.device.type == "cuda" and config.precision == "bf16" else "fp32"

    def _make_optimizer(self, *, training=True):
        head = [p for name, p in self.network.named_parameters() if not name.startswith("vae.") and p.requires_grad]
        decoder = [p for p in self.network.vae.decoder.parameters() if p.requires_grad]
        groups = [{"params": head, "lr_scale": 1., "lr": self.config.learning_rate}]
        if decoder:
            groups.append({"params": decoder, "lr_scale": self.config.decoder_lr_multiplier,
                           "lr": self.config.learning_rate * self.config.decoder_lr_multiplier})
        self.parameters = head + decoder
        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay) if training else None

    def reconfigure_decoder(self, requested):
        requested.validate()
        if getattr(self, '_inference_only', False):
            raise RuntimeError('Cannot reconfigure an inference-only engine')
        if requested.resume_contract() != self.config.resume_contract():
            raise ValueError('Resume preserves architecture, loss, augmentation and base optimization settings')
        if requested.to_dict() == self.config.to_dict():
            return False
        self._validate_precision(requested)
        self.assert_frozen()
        self.optimizer.zero_grad(set_to_none=True)
        self.config = self.training_config = requested
        self.network.vae.set_decoder_mode(requested.decoder_mode)
        self.network.vae.compute_mode = self._compute_mode(requested)
        self.vae_contract = self.network.vae.contract()
        self.stage_index += 1
        self.stage_updates = 0
        self.stage_history.append(self._stage_record())
        self._make_optimizer()  # Explicit new stage: retain weights/data, reset AdamW and warmup.
        self._frozen_versions = self._versions()
        return True

    def _versions(self):
        tensors = [(n, p) for n, p in self.network.vae.named_parameters() if not p.requires_grad]
        tensors += list(self.network.vae.named_buffers())
        return {name: (id(tensor), tensor._version) for name, tensor in tensors}

    def assert_frozen(self):
        self.network.vae.assert_frozen()
        if self._versions() != self._frozen_versions:
            raise RuntimeError("Frozen TAEQI2.1 parameter/buffer changed in-place")

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == "cuda" and self.config.precision == "bf16")

    def tensors(self, batch):
        return tuple(torch.as_tensor(value, dtype=torch.float32, device=self.device) for value in batch)

    def _batch(self, values, *, for_preview=False):
        if len(values) != 4:
            raise ValueError("Expected DFL (warped BGR, target BGR, full mask, encoded eyes/mouth) tuple")
        # Native DFL batches are CPU arrays. Validate their FP32 representation
        # before uploading; callers that already own device tensors still get
        # the same checks, with only one packed flag transfer to the host.
        on_cpu = all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in values)
        checked = (tuple(torch.as_tensor(value, dtype=torch.float32, device="cpu") for value in values)
                   if on_cpu else self.tensors(values))
        batch_size = checked[0].shape[0] if checked[0].ndim == 4 else 0
        for tensor, channels in zip(checked, (3, 3, 1, 1)):
            if tuple(tensor.shape) != (batch_size, channels, self.config.resolution, self.config.resolution) or batch_size < 1:
                raise ValueError("DFL batch has incorrect NCHW shape or resolution")
        finite, bad_image_range, bad_encoded_range = torch.stack((
            all_finite(checked),
            torch.stack([((value < 0) | (value > 1)).any() for value in checked[:3]]).any(),
            ((checked[3] < 0) | (checked[3] > 3.001)).any(),
        )).cpu().tolist()
        if not finite:
            raise FloatingPointError("Nonfinite DFL batch")
        if bad_image_range:
            raise ValueError("DFL image and full mask values must lie in [0,1]")
        if bad_encoded_range:
            raise ValueError("Invalid DFL encoded eyes/mouth mask")
        if for_preview:
            # Preview uses only aligned targets. Keep validation of all four
            # inputs, but do not upload warped images/masks or build loss masks.
            if (self.config.training_mask_mode == "expanded_feather" and
                    not bool((checked[2] >= 0.5).flatten(1).any(1).all())):
                raise ValueError("Expanded training mask needs a face region with weight >= 0.5")
            target = torch.as_tensor(checked[1], dtype=torch.float32, device=self.device)
            return {"target": target[:, [2, 1, 0]]}
        data = self.tensors(checked) if on_cpu else checked
        # ME encodes eyes as 1+soft_mask and mouth as 2+soft_mask.
        encoded = data[3]
        region = (encoded-1).clamp(0, 1)
        coverage = data[2]
        if self.config.training_mask_mode == "full":
            coverage = torch.ones_like(coverage)
        elif self.config.training_mask_mode == "expanded_feather":
            # The DFL loader supplies CPU arrays: build this target there, avoiding
            # an unnecessary GPU-to-CPU copy of the image/mask batch.
            original = checked[2].detach().cpu().numpy()
            coverage = torch.as_tensor(image_loss_mask(original, self.config.training_mask_mode),
                                       dtype=torch.float32, device=self.device)
        return {"input": data[0][:, [2, 1, 0]], "target": data[1][:, [2, 1, 0]],
                "mask": data[2], "image_mask": coverage, "eyes_mouth": region}

    def train_step(self, source, destination):
        if getattr(self, '_inference_only', False):
            raise RuntimeError('Load with training=True before resuming training')
        previous_updates = self.optimizer_updates
        previous_lrs = [group['lr'] for group in self.optimizer.param_groups]
        try:
            return self._train_step(source, destination)
        except BaseException:
            # An incomplete paired update must never leak into the next one.
            self.optimizer.zero_grad(set_to_none=True)
            if self.optimizer_updates == previous_updates:
                for group, lr in zip(self.optimizer.param_groups, previous_lrs):
                    group['lr'] = lr
            raise

    def _train_step(self, source, destination):
        self.network.train()
        self.assert_frozen()
        src, dst = self._batch(source), self._batch(destination)
        lr = self.config.learning_rate * min(1., (self.stage_updates+1)/max(1, self.config.warmup_steps))
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        # Backprop each reconstruction before decoding the other side. Only one
        # expensive TAE decoder graph stays live; both accumulate into one step.
        self.optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for domain, batch in (("src", src), ("dst", dst)):
            with self.autocast():
                output = self.network.reconstruct(batch["input"], domain=domain)
            loss, terms = reconstruction_loss(output["image"], output["mask"], batch, self.config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite TAE loss; optimizer was not updated")
            (loss * 0.5).backward()
            metrics[domain+"_loss"] = loss.detach()
            metrics.update({domain+"_"+key: value.detach() for key, value in terms.items()})
            del output, loss, terms
        if not self.parameters or any(parameter.grad is None for parameter in self.parameters):
            raise FloatingPointError("TAE decoder gradients are nonfinite or disconnected; no update accepted")
        try:
            # The L2 norm already detects every NaN/Inf gradient before clipping.
            # Avoid scanning/reducing all gradients a second time on valid steps.
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.grad_clip, error_if_nonfinite=True)
        except RuntimeError as error:
            if (str(error).startswith('The total norm of order ') and
                    ' is non-finite, so it cannot be clipped.' in str(error) and
                    not gradients_are_finite(self.parameters)):
                # Preserve the original distinction: bad gradients raise here;
                # finite gradients whose norm overflows keep Torch's error.
                raise FloatingPointError("TAE decoder gradients are nonfinite or disconnected; no update accepted") from error
            raise
        self.optimizer.step()
        self.iteration += 1
        self.optimizer_updates += 1
        self.stage_updates += 1
        self.assert_frozen()
        metrics["grad_norm"] = norm
        metrics = scalar_metrics(metrics)
        memory = {} if self.device.type != 'cuda' else {
            'cuda_allocated_mib': torch.cuda.memory_allocated(self.device) / 2**20,
            'cuda_reserved_mib': torch.cuda.memory_reserved(self.device) / 2**20,
            'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated(self.device) / 2**20,
        }
        return {"iteration": self.iteration, **metrics, "lr": lr, **memory,
                "optimizer_updates": self.optimizer_updates,
                "amp_retries": 0, "encoder_frozen": True, "decoder_mode": self.config.decoder_mode,
                "stage_index": self.stage_index, "stage_updates": self.stage_updates,
                "decoder_lr": lr * self.config.decoder_lr_multiplier if self.config.decoder_mode != "frozen" else 0.}

    @torch.inference_mode()
    def preview_outputs(self, source, destination):
        self.network.eval()
        src, dst = self._batch(source, for_preview=True), self._batch(destination, for_preview=True)
        with self.autocast():
            result = self.network(src["target"], dst["target"], include_swap=True)
        return {key: (value[:, [2, 1, 0]].clamp(0, 1) if value.shape[1] == 3 else value).float()
                for key, value in result.items()}

    @torch.inference_mode()
    def predict(self, bgr):
        # DFL supplies CPU arrays. Validate before upload; direct tensor callers
        # retain validation on their source device.
        array = np.asarray(bgr, dtype=np.float32) if isinstance(bgr, np.ndarray) else None
        value = array if array is not None else torch.as_tensor(bgr, dtype=torch.float32)
        size = self.config.resolution
        if value.ndim != 4 or value.shape[1:] != (3, size, size) or len(value) < 1:
            raise ValueError("Predictor expects BGR NCHW at the model resolution")
        valid = (np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1
                 if array is not None else bool(finite_unit_interval(value)))
        if not valid:
            raise ValueError("Predictor input must be finite BGR float in [0,1]")
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.network.eval()
        with self.autocast():
            result = self.network.predict(value[:, [2, 1, 0]])
        outputs = (result["swap"][:, [2, 1, 0]], result["swap_mask"], result["dst_mask"])
        # NumPy outputs already require these copies. Validate the returned
        # values instead of synchronizing three additional device booleans.
        arrays = tuple(np.ascontiguousarray(x.float().cpu().numpy()) for x in outputs)
        if not all(np.isfinite(x).all() for x in arrays):
            raise FloatingPointError("Nonfinite TAE prediction")
        return arrays

    def metadata(self):
        return {"training_mode": self.training_mode,
                "pretraining_data": self.pretraining_data,
                "vae_weights": self.vae_weights, "vae_sha256": self.vae_sha256,
                "vae_contract": self.vae_contract, "architecture": self.network.architecture,
                "optimizer_updates": self.optimizer_updates, "stage_updates": self.stage_updates,
                "stage_index": self.stage_index, "stage_history": self.stage_history,
                "parameters": {"trainable": sum(p.numel() for p in self.parameters),
                    "encoder": sum(p.numel() for p in self.network.vae.encoder.parameters()),
                    "decoder_trainable": sum(p.numel() for p in self.network.vae.decoder.parameters() if p.requires_grad),
                    "decoder_total": sum(p.numel() for p in self.network.vae.decoder.parameters())}}

    def save(self, path, data_state=None):
        if self._inference_only:
            raise RuntimeError('Cannot save an inference-only engine; load with training=True')
        self.assert_frozen()
        payload = {"format": FORMAT, "version": VERSION, "config": self.config.to_dict(),
            "iteration": self.iteration, "optimizer_updates": self.optimizer_updates,
            "model": self.network.trainable_state_dict(), "optimizer": self.optimizer.state_dict(),
            "scaler": {}, "rng": rng_state(),
            "data_state": data_state if data_state is not None else self.data_state, **self.metadata()}
        atomic_save(payload, path)

    @classmethod
    def load(cls, path, device="cuda", weights=None, vae=None, training=True):
        if type(training) is not bool:
            raise ValueError('training must be a boolean')
        payload = read_checkpoint(path, mmap=not training)
        config = TrainConfig.from_dict(payload["config"])
        # Explicit relocation > environment override > original path; SHA still
        # binds the exact base weights, so changing directories cannot swap VAE.
        asset = weights or os.environ.get("TAE_VAE_WEIGHTS") or payload.get("vae_weights")
        engine = cls(config, device=device, weights=asset, vae=vae, training=training)
        engine.training_mode = payload.get("training_mode", "identity")
        engine.pretraining_data = payload.get("pretraining_data")
        if engine.training_mode not in {"identity", "pretraining"}:
            raise ValueError("Invalid TAE training mode")
        saved_contract, actual_contract = dict(payload.get("vae_contract", {})), dict(engine.vae_contract)
        if not training:
            saved_contract.pop('compute_mode', None)
            actual_contract.pop('compute_mode', None)
        if payload.get("vae_sha256") != engine.vae_sha256 or saved_contract != actual_contract:
            raise ValueError("TAE base weight or numerical contract differs from checkpoint")
        if payload.get("architecture") != engine.network.architecture:
            raise ValueError("TAE architecture differs from checkpoint")
        state = payload["model"]
        if not isinstance(state, dict) or any(not isinstance(x, torch.Tensor) or not bool(torch.isfinite(x).all()) for x in state.values()):
            raise ValueError("Invalid/nonfinite TAE checkpoint tensor")
        engine.network.load_trainable_state_dict(state)
        iteration, updates = payload["iteration"], payload["optimizer_updates"]
        if type(iteration) is not int or type(updates) is not int or iteration < 0 or iteration != updates:
            raise ValueError("TAE iteration/optimizer count mismatch")
        stage_updates, stage_index = payload.get("stage_updates"), payload.get("stage_index")
        history = payload.get("stage_history")
        if (type(stage_updates) is not int or not 0 <= stage_updates <= updates or
            type(stage_index) is not int or stage_index < 0 or not isinstance(history, list) or len(history) != stage_index + 1):
            raise ValueError("Invalid TAE optimization stage")
        for index, record in enumerate(history):
            if not isinstance(record, dict):
                raise ValueError("Invalid TAE stage history")
            record.setdefault("batch_size", config.batch_size)
            record.setdefault("ct_mode", "none")
            record.setdefault("training_mask_mode", "face")
            record.setdefault("random_warp", config.random_warp)
            record.setdefault("learning_rate", config.learning_rate)
            record.setdefault("eyes_mouth_weight", config.eyes_mouth_weight)
            if (record.get("index") != index or
                type(record.get("started_at_iteration")) is not int or
                not 0 <= record["started_at_iteration"] <= updates or
                (index == 0 and record["started_at_iteration"] != 0) or
                (index and record["started_at_iteration"] < history[index-1]["started_at_iteration"])):
                raise ValueError("Invalid TAE stage history")
            TrainConfig.from_dict({**config.to_dict(), "decoder_mode": record.get("decoder_mode"),
                                   "decoder_lr_multiplier": record.get("decoder_lr_multiplier"),
                                   "precision": record.get("precision", "bf16"),
                                   "batch_size": record.get("batch_size", config.batch_size),
                                   "ct_mode": record.get("ct_mode", "none"),
                                   "training_mask_mode": record["training_mask_mode"],
                                   "random_warp": record["random_warp"],
                                   "learning_rate": record["learning_rate"],
                                   "eyes_mouth_weight": record["eyes_mouth_weight"]})
        last = history[-1]
        if (updates - last["started_at_iteration"] != stage_updates or
            last["decoder_mode"] != config.decoder_mode or last["decoder_lr_multiplier"] != config.decoder_lr_multiplier or
            last.get("precision", "bf16") != config.precision or
            last.get("batch_size", config.batch_size) != config.batch_size or
            last.get("ct_mode", "none") != config.ct_mode or
            last["training_mask_mode"] != config.training_mask_mode or
            last["random_warp"] != config.random_warp or
            last["learning_rate"] != config.learning_rate or
            last["eyes_mouth_weight"] != config.eyes_mouth_weight):
            raise ValueError("TAE stage/config mismatch")
        engine.stage_updates, engine.stage_index, engine.stage_history = stage_updates, stage_index, history
        engine._frozen_versions = engine._versions()
        if not training:
            engine.iteration, engine.optimizer_updates = iteration, updates
            return engine
        saved_groups = payload['optimizer'].get('param_groups', [])
        expected_groups = engine.optimizer.state_dict()['param_groups']
        if len(saved_groups) != len(expected_groups):
            raise ValueError('Invalid TAE optimizer parameter groups')
        expected_ids = []
        for actual, expected in zip(saved_groups, expected_groups):
            factor = min(1., stage_updates/max(1, config.warmup_steps)) if stage_updates else 1.
            expected['lr'] = config.learning_rate * factor * expected['lr_scale']
            for key in ('lr', 'lr_scale', 'params', 'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize'):
                if actual.get(key) != expected[key]:
                    raise ValueError('TAE optimizer settings differ from saved configuration: '+key)
            expected_ids.extend(expected['params'])
        expected_state_ids = set(expected_ids) if stage_updates else set()
        if set(payload['optimizer'].get('state', {})) != expected_state_ids:
            raise ValueError('Incomplete or unexpected TAE AdamW history')
        engine.optimizer.load_state_dict(payload["optimizer"])
        for parameter in engine.parameters:
            entry = engine.optimizer.state.get(parameter, {})
            if stage_updates and set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Incomplete TAE AdamW state")
            if entry:
                if float(entry["step"]) != stage_updates:
                    raise ValueError("TAE AdamW step mismatch")
                for key in ("exp_avg", "exp_avg_sq"):
                    if entry[key].shape != parameter.shape or not bool(torch.isfinite(entry[key]).all()):
                        raise ValueError("Invalid TAE AdamW tensor")
                if bool((entry['exp_avg_sq'] < 0).any()):
                    raise ValueError('Invalid TAE AdamW second moment')
        validate_scaler_state(payload["scaler"])
        if payload["scaler"]:
            raise ValueError("TAEQI2.1 BF16/FP32 checkpoints must not use gradient scaling")
        engine.iteration, engine.optimizer_updates = iteration, updates
        engine.data_state = payload.get("data_state")
        restore_rng(payload["rng"])
        return engine
