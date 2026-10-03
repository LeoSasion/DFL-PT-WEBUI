"""ME training, runtime policy and atomic native checkpoints."""
import copy
from contextlib import nullcontext
import math
import os
from pathlib import Path
import tempfile
import torch
from .config import MEConfig
from .network import MENetwork
from .optimizer import MEOptimizer, optimizer_settings
from .losses import me_losses
from .replay import HardSampleReplay


class TrainingForward(torch.nn.Module):
    """Autocast belongs inside forward because DataParallel uses worker threads."""
    def __init__(self, network, use_fp16):
        super().__init__()
        self.network, self.use_fp16 = network, use_fp16

    def forward(self, source, destination, face_style=False):
        with torch.autocast(source.device.type, dtype=torch.float16,
                            enabled=self.use_fp16 and source.device.type == 'cuda'):
            return self.network(source, destination, face_style)


def resolve_devices(device):
    text = str(device)
    if text == 'cpu':
        return torch.device('cpu'), []
    if text == 'cuda':
        indexes = [torch.cuda.current_device()] if torch.cuda.is_available() else [0]
    elif text.startswith('cuda:'):
        try:
            indexes = [int(part) for part in text[5:].split(',')]
        except ValueError as error:
            raise ValueError('GPU devices must be cuda:0 or cuda:0,1') from error
    else:
        raise ValueError('ME device must be cpu, cuda, or cuda:0,1')
    if not indexes or len(set(indexes)) != len(indexes) or any(index < 0 for index in indexes):
        raise ValueError('GPU indexes must be unique nonnegative integers')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
    if any(index >= torch.cuda.device_count() for index in indexes):
        raise ValueError('Requested GPU is unavailable')
    return torch.device('cuda', indexes[0]), indexes


def validate_scaler_state(state):
    if not isinstance(state, dict):
        raise ValueError('Invalid AMP scaler checkpoint')
    if not state:
        return
    if set(state) != {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}:
        raise ValueError('Invalid AMP scaler fields')
    for name in ('scale', 'growth_factor', 'backoff_factor'):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid AMP scaler value')
    if state['scale'] <= 0 or state['growth_factor'] <= 1 or not 0 < state['backoff_factor'] < 1:
        raise ValueError('Invalid AMP scaler value')
    interval, tracker = state['growth_interval'], state['_growth_tracker']
    if type(interval) is not int or interval < 1 or type(tracker) is not int or not 0 <= tracker < interval:
        raise ValueError('Invalid AMP scaler growth tracker')


class MEEngine:
    def __init__(self, config, device='cuda', seed=0):
        self.config = config
        self.training_config = config.effective()
        self.device, self.device_ids = resolve_devices(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(seed)
        self.network = MENetwork(self.training_config).to(self.device)
        self.optimizer = MEOptimizer((p for p in self.network.parameters() if p.requires_grad), config)
        self.iteration = self.optimizer_updates = 0
        self.data_state = None
        self.replay = HardSampleReplay(config.retraining_capacity)
        self._build_runtime()
        self._build_adversarial()

    def _build_runtime(self):
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_fp16 and self.device.type == 'cuda', init_scale=1024.)
        self.forward_network = TrainingForward(self.network, self.config.use_fp16)
        if len(self.device_ids) > 1:
            if self.config.batch_size < len(self.device_ids):
                raise ValueError('Batch size must be at least the number of selected GPUs')
            self.forward_network = torch.nn.DataParallel(self.forward_network, device_ids=self.device_ids, output_device=self.device_ids[0])

    def _build_adversarial(self):
        from .adversarial import AdversarialTraining
        self.adversarial = AdversarialTraining(self.training_config, self.network).to(self.device)
        parameters = list(self.adversarial.parameters())
        self.adversarial_optimizer = MEOptimizer(parameters, self.config) if parameters else None

    def tensors(self, batch):
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch)

    def _update(self, source, destination, profiler=None, stage_prefix=''):
        try:
            return self._update_impl(source, destination, profiler, stage_prefix)
        except BaseException:
            # Recreate through public APIs to clear any partially completed
            # unscale stage while keeping the current scale/growth history.
            if self.scaler.is_enabled():
                state = self.scaler.state_dict()
                self.scaler = torch.amp.GradScaler('cuda', enabled=True)
                self.scaler.load_state_dict(state)
            self.optimizer.zero_grad(set_to_none=True)
            self.adversarial.requires_grad_(True)
            if self.adversarial_optimizer is not None:
                self.adversarial_optimizer.zero_grad(set_to_none=True)
            raise

    def _update_impl(self, source, destination, profiler=None, stage_prefix=''):
        def timed(name):
            return profiler.stage(stage_prefix + name) if profiler is not None else nullcontext()

        with timed('transfer'):
            source, destination = self.tensors(source), self.tensors(destination)
        for attempt in range(64):
            self.optimizer.zero_grad(set_to_none=True)
            self.adversarial.requires_grad_(False)
            with timed('forward'):
                outputs = self.forward_network(source[0], destination[0], bool(self.training_config.face_style_power))
                outputs = {name: value.float() for name, value in outputs.items()}
            with timed('loss'):
                src_loss, dst_loss = me_losses(self.training_config, source, destination, outputs)
                generator_loss = self.adversarial.generator_loss(outputs, source, destination)
                total = (src_loss + dst_loss + generator_loss).mean()
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError('Nonfinite ME loss; no optimizer update performed')
            with timed('generator_backward'):
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
            with timed('gradient_check'):
                grads = [p.grad for p in self.network.parameters() if p.grad is not None]
                finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
            discriminator_loss = None
            self.adversarial.requires_grad_(True)
            if self.adversarial_optimizer is not None:
                with timed('discriminator_backward'):
                    self.adversarial_optimizer.zero_grad(set_to_none=True)
                    discriminator_loss = self.adversarial.discriminator_loss(outputs, source, destination)
                    if not bool(torch.isfinite(discriminator_loss)):
                        raise FloatingPointError('Nonfinite ME discriminator loss; no optimizer update performed')
                    self.scaler.scale(discriminator_loss).backward()
                    self.scaler.unscale_(self.adversarial_optimizer)
                    finite = finite and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.adversarial.parameters())
            if finite:
                with timed('optimizer'):
                    self.scaler.step(self.optimizer)
                    if self.adversarial_optimizer is not None:
                        self.scaler.step(self.adversarial_optimizer)
                    self.scaler.update()
                    self.optimizer_updates += 1
                return src_loss.detach(), dst_loss.detach(), generator_loss.detach(), discriminator_loss
            if not self.scaler.is_enabled() or self.scaler.get_scale() <= 1. or attempt == 63:
                raise FloatingPointError('Missing/nonfinite ME gradients; no optimizer update performed')
            self.scaler.update(new_scale=max(self.scaler.get_scale() / 2., 1.))
        raise RuntimeError('AMP retry loop did not complete')

    def train_step(self, source, destination, profiler=None):
        if self.config.use_fp16 and self.device.type != 'cuda':
            raise ValueError('FP16 training requires CUDA; CPU inference/export use FP32')
        if profiler is not None:
            if len(self.device_ids) > 1:
                raise ValueError('Stage profiling currently supports one GPU')
            profiler.start(self.device)
        self.network.train()
        self.adversarial.train()
        src_loss, dst_loss, generator_loss, discriminator_loss = self._update(source, destination, profiler)
        self.iteration += 1
        replayed = False
        if self.config.retraining_samples:
            self.replay.add(source, destination, src_loss, dst_loss)
            if self.iteration % self.config.retraining_every == 0:
                replay_batch = self.replay.batch(self.config.batch_size)
                if replay_batch is not None:
                    self._update(*replay_batch, profiler=profiler, stage_prefix='replay_')
                    replayed = True
        result = dict(iteration=self.iteration, src_loss=float(src_loss.mean()), dst_loss=float(dst_loss.mean()))
        if self.adversarial_optimizer is not None:
            result.update(generator_loss=float(generator_loss.mean()), discriminator_loss=float(discriminator_loss.detach()))
        if self.config.retraining_samples:
            result.update(replayed=replayed, optimizer_updates=self.optimizer_updates)
        if profiler is not None:
            profiler.finish((self.optimizer, self.adversarial_optimizer))
        return result

    @torch.inference_mode()
    def predict(self, destination):
        self.network.eval()
        x = torch.as_tensor(destination, dtype=torch.float32, device=self.device)
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.config.resolution, self.config.resolution):
            raise ValueError('Expected NCHW aligned BGR input at model resolution')
        return tuple(t.float().cpu().numpy() for t in self.network.predict(x))

    def reconfigure(self, config, reset_optimizer=False):
        """Build replacement training state before committing any runtime change."""
        self.config.assert_compatible_network(config)
        if config.adabelief != self.config.adabelief and not reset_optimizer:
            raise ValueError('Changing optimizer type requires --reset-optimizer')
        if len(self.device_ids) > config.batch_size:
            raise ValueError('Batch size must be at least the number of selected GPUs')
        from .adversarial import AdversarialTraining
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []
        try:
            effective = config.effective()
            adversarial = AdversarialTraining(effective, self.network).to(self.device)
            old_weights, new_weights = self.adversarial.state_dict(), adversarial.state_dict()
            preserved_components = set()
            for component in ('gan', 'true_face'):
                prefix = component + '.'
                old = {key: value for key, value in old_weights.items() if key.startswith(prefix)}
                new = {key: value for key, value in new_weights.items() if key.startswith(prefix)}
                if old and new:
                    compatible = set(old) == set(new) and all(old[key].shape == new[key].shape for key in old)
                    if not compatible and not reset_optimizer:
                        raise ValueError('Changing discriminator shape requires --reset-optimizer')
                    if compatible:
                        new_weights.update(old)
                        preserved_components.add(component)
            adversarial.load_state_dict(new_weights, strict=True)
            selected = [(name, param) for name, param in self.network.named_parameters()
                        if not (name.startswith('inter_AB.') and not effective.random_warp)]
            optimizer = MEOptimizer([param for _, param in selected], config)
            updates = 0 if reset_optimizer else self.optimizer_updates
            def transfer_state(old_state, parameter):
                device = 'cpu' if config.optimizer_on_cpu else parameter.device
                if old_state:
                    return {key: value.to(device=device).clone() for key, value in old_state.items()}
                state = {'variance': torch.zeros_like(parameter, dtype=torch.float32, device=device)}
                if config.adabelief:
                    state['momentum'] = torch.zeros_like(parameter, dtype=torch.float32, device=device)
                return state
            if not reset_optimizer:
                for _, parameter in selected:
                    if updates:
                        optimizer.state[parameter] = transfer_state(self.optimizer.state.get(parameter), parameter)
                for group in optimizer.param_groups:
                    group['iteration'] = updates
            adv_parameters = list(adversarial.parameters())
            adv_optimizer = MEOptimizer(adv_parameters, config) if adv_parameters else None
            if adv_optimizer is not None and self.adversarial_optimizer is not None and not reset_optimizer:
                old_parameters = dict(self.adversarial.named_parameters())
                adv_updates = self.adversarial_optimizer.param_groups[0]['iteration']
                for name, parameter in adversarial.named_parameters():
                    component = name.partition('.')[0]
                    old_parameter = old_parameters.get(name)
                    if adv_updates:
                        previous = self.adversarial_optimizer.state.get(old_parameter) if component in preserved_components else None
                        adv_optimizer.state[parameter] = transfer_state(previous, parameter)
                for group in adv_optimizer.param_groups:
                    group['iteration'] = adv_updates
            optimizer.validate_state(updates)
            if adv_optimizer is not None:
                adv_optimizer.validate_state()
        except BaseException:
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            raise
        if all(key.partition('.')[0] in preserved_components for key in new_weights):
            # A policy-only change must not consume dropout/label RNG through
            # temporary discriminator initialization whose weights were discarded.
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
        scaler_state = (self.scaler.state_dict() or getattr(self, '_saved_scaler_state', {})) if config.use_fp16 == self.config.use_fp16 else None
        replay_state = self.replay.state_dict()
        replay_state = {key: entries[:config.retraining_capacity] for key, entries in replay_state.items()}
        self.config, self.training_config = config, effective
        self.optimizer, self.optimizer_updates = optimizer, updates
        self.adversarial, self.adversarial_optimizer = adversarial, adv_optimizer
        self.network.use_rg = config.use_rg
        if self.network.archi_type == 'liae':
            self.network.inter_AB.requires_grad_(effective.random_warp)
        self.replay = HardSampleReplay(config.retraining_capacity)
        self.replay.load_state_dict(replay_state)
        self._build_runtime()
        self._saved_scaler_state = scaler_state or {}
        if scaler_state and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)

    def save(self, path, data_state=None, import_metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format='me-pytorch', version=1, config=self.config.to_dict(), iteration=self.iteration,
            model=self.network.state_dict(), optimizer=self.optimizer.state_dict(), optimizer_updates=self.optimizer_updates,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            data_state=data_state if data_state is not None else self.data_state,
            scaler=self.scaler.state_dict() or getattr(self, '_saved_scaler_state', {}), replay=self.replay.state_dict(), adversarial=self.adversarial.state_dict(),
            adversarial_optimizer=self.adversarial_optimizer.state_dict() if self.adversarial_optimizer else None)
        provenance = import_metadata if import_metadata is not None else getattr(self, 'import_metadata', None)
        if provenance is not None:
            payload['import_metadata'] = provenance
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, device='cuda'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        required = {'format','version','config','iteration','model','optimizer','cpu_rng','cuda_rng','data_state'}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError('Incomplete ME checkpoint')
        if payload['format'] != 'me-pytorch' or payload['version'] != 1:
            raise ValueError('Unsupported ME checkpoint format')
        engine = cls(MEConfig.from_dict(payload['config']), device)
        engine.network.load_state_dict(payload['model'], strict=True)
        engine.optimizer.load_state_dict(payload['optimizer'])
        for group in engine.optimizer.param_groups:
            if any(group.get(key) != value for key, value in optimizer_settings(engine.config).items()):
                raise ValueError('Checkpoint optimizer settings differ from its ME configuration')
        engine.iteration = payload['iteration']
        engine.optimizer_updates = payload.get('optimizer_updates', engine.iteration)
        if any(type(value) is not int or value < 0 for value in (engine.iteration, engine.optimizer_updates)):
            raise ValueError('Invalid checkpoint iteration')
        if hasattr(engine.optimizer, 'validate_state'):
            engine.optimizer.validate_state(engine.optimizer_updates, require_complete=True)
        else:
            engine._validate_optimizer()
        if any(not bool(torch.isfinite(param).all()) for param in engine.network.parameters()):
            raise ValueError('Nonfinite checkpoint parameter')
        if engine.adversarial_optimizer is not None:
            if 'adversarial' not in payload or payload.get('adversarial_optimizer') is None:
                raise ValueError('Incomplete discriminator checkpoint')
            engine.adversarial.load_state_dict(payload['adversarial'], strict=True)
            engine.adversarial_optimizer.load_state_dict(payload['adversarial_optimizer'])
            engine.adversarial_optimizer.validate_state(require_complete=True)
            for group in engine.adversarial_optimizer.param_groups:
                if any(group.get(key) != value for key, value in optimizer_settings(engine.config).items()):
                    raise ValueError('Checkpoint discriminator optimizer settings differ from its ME configuration')
            if any(not bool(torch.isfinite(param).all()) for param in engine.adversarial.parameters()):
                raise ValueError('Nonfinite discriminator checkpoint parameter')
        elif payload.get('adversarial'):
            raise ValueError('Unexpected discriminator checkpoint')
        scaler_state = payload.get('scaler', {})
        validate_scaler_state(scaler_state)
        if scaler_state and not engine.config.use_fp16:
            raise ValueError('Unexpected AMP scaler checkpoint')
        engine._saved_scaler_state = scaler_state
        if scaler_state and engine.scaler.is_enabled():
            engine.scaler.load_state_dict(scaler_state)
        if payload.get('replay') is not None:
            engine.replay.load_state_dict(payload['replay'])
            for entries in (engine.replay.source, engine.replay.destination):
                for _, sample in entries:
                    if any(tuple(value.shape) != (channels, engine.config.resolution, engine.config.resolution)
                           for channels, value in zip((3, 3, 1, 1), sample)):
                        raise ValueError('Invalid hard-sample replay shape')
        torch.set_rng_state(payload['cpu_rng'])
        if engine.device.type == 'cuda' and payload['cuda_rng']:
            for index, state in enumerate(payload['cuda_rng'][:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, index)
        engine.data_state = payload['data_state']
        engine.import_metadata = payload.get('import_metadata')
        return engine

    def _validate_optimizer(self):
        for group in self.optimizer.param_groups:
            if group['iteration'] != self.optimizer_updates:
                raise ValueError('Optimizer/model iteration mismatch')
            expected = {'momentum', 'variance'} if group['adabelief'] else {'variance'}
            for param in group['params']:
                if self.optimizer_updates:
                    state = self.optimizer.state.get(param)
                    if not state:
                        raise ValueError('Incomplete optimizer state')
                    if set(state) != expected or any(value.shape != param.shape or not bool(torch.isfinite(value).all()) for value in state.values()):
                        raise ValueError('Invalid optimizer tensor state')
