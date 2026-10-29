"""TAE engine contracts using a tiny explicit test codec; no TAE/GPU claim."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_internal" / "DeepFaceLab"))

from tae.config import TrainConfig
from tae.engine import TAEEngine
from tae.losses import reconstruction_loss


class TinyFrozenVAE(nn.Module):
    """Fixed deterministic codec, specifically not a substitute TAE checkpoint."""

    latent_channels = 4
    spatial_compression = 16
    weight_sha256 = "tiny-test-only"
    source_sha256 = "tiny-runtime-test-only"
    compute_mode = "float32-test-only"

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 3, 1))
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.arange(p.numel()).view_as(p) / max(24., p.numel()))
        self.set_decoder_mode("frozen")
        self.eval()
        self.encoded_inputs = []
        self.nan_decode = False

    def set_decoder_mode(self, mode):
        self.decoder_mode = mode
        self.encoder.requires_grad_(False).eval()
        for name, p in self.decoder.named_parameters():
            p.requires_grad_(mode == "full" or (mode == "tail" and name.startswith("1.")))
            p.grad = None
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode and self.decoder_mode != "frozen")
        self.encoder.eval()
        return self

    def encode(self, rgb):
        self.encoded_inputs.append(rgb.detach().clone())
        return self.encoder(F.avg_pool2d(rgb, 16))

    def decode(self, latent, *, clamp=False):
        rgb = F.interpolate(self.decoder(latent), scale_factor=16, mode="bilinear", align_corners=False)
        if self.nan_decode:
            rgb = rgb * float("nan")
        return rgb.clamp(0, 1) if clamp else rgb

    def assert_frozen(self):
        if self.encoder.training or any(p.requires_grad or p.grad is not None for p in self.encoder.parameters()):
            raise RuntimeError("The test encoder was unfrozen")
        for name, p in self.decoder.named_parameters():
            if p.requires_grad != (self.decoder_mode == "full" or (self.decoder_mode == "tail" and name.startswith("1."))):
                raise RuntimeError("Decoder scope changed")

    def contract(self):
        return {"weights_sha256": self.weight_sha256, "source_sha256": self.source_sha256,
                "compute_mode": self.compute_mode, "encoder_frozen": True,
                "decoder_backward": "ordinary autograd, tiny-test-only"}


def batch(seed=0):
    rng = np.random.default_rng(seed)
    target = rng.random((1, 3, 256, 256), dtype=np.float32)
    warped = np.clip(target * .9 + .02, 0, 1)
    mask = np.ones((1, 1, 256, 256), dtype=np.float32)
    mask[:, :, :4, :] = 0
    encoded = np.ones_like(mask)
    encoded[:, :, 20:26, 10:54] = 1.5  # Fractional eye-mask edge.
    encoded[:, :, 40:48, 20:44] = 2.5  # Mouth is also in eyes+mouth union.
    return warped, target, mask, encoded


def config():
    return TrainConfig(resolution=256, face_type="wf", inter_dims=8,
                       hidden_dims=8, batch_size=1, learning_rate=1e-3,
                       warmup_steps=3, random_warp=False, random_flip=False)


def engine(seed=7):
    return TAEEngine(config(), device="cpu", seed=seed, vae=TinyFrozenVAE())


def nested_keys(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from nested_keys(child, path)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from nested_keys(child, f"{prefix}[{index}]")


class EngineTests(unittest.TestCase):
    def test_inference_mapping_does_not_pin_checkpoint_or_load_mutable_adam_state(self):
        worker = engine()
        worker.train_step(batch(11), batch(12))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tae.pt'
            worker.save(path)
            expected = worker.predict(batch(13)[0])
            inference = TAEEngine.load(path, 'cpu', vae=TinyFrozenVAE(), training=False)
            self.assertIsNone(inference.optimizer)
            resumed = TAEEngine.load(path, 'cpu', vae=TinyFrozenVAE())
            resumed.train_step(batch(14), batch(15))
            # On Windows a retained mmap would prevent this atomic replacement.
            resumed.save(path)
            self.assertEqual(TAEEngine.load(path, 'cpu', vae=TinyFrozenVAE()).iteration, 2)
            for actual, original in zip(inference.predict(batch(13)[0]), expected):
                np.testing.assert_array_equal(actual, original)
            with self.assertRaisesRegex(RuntimeError, 'training=True'):
                inference.train_step(batch(14), batch(15))

    def test_precision_stage_and_legacy_checkpoint_compatibility(self):
        from dataclasses import replace
        worker = engine()
        worker.train_step(batch(29), batch(30))
        fingerprint_policy = worker.config.resume_contract()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tae.pt'
            worker.save(path)
            old = torch.load(path, map_location='cpu', weights_only=True)
            old['config'].pop('precision')
            old['config'].pop('ct_mode')
            for stage in old['stage_history']:
                stage.pop('precision')
                stage.pop('batch_size')
                stage.pop('ct_mode')
            torch.save(old, path)
            worker = TAEEngine.load(path, 'cpu', vae=TinyFrozenVAE())
            self.assertEqual(worker.config.precision, 'bf16')
            self.assertEqual(worker.config.resume_contract(), fingerprint_policy)
            previous = worker.network.trainable_state_dict()
            self.assertTrue(worker.reconfigure_decoder(replace(worker.config, precision='fp32',
                                                               batch_size=8, ct_mode='rct')))
            self.assertEqual(worker.iteration, 1)
            self.assertEqual(worker.stage_updates, 0)
            self.assertFalse(worker.optimizer.state)
            self.assertEqual(worker.config.resume_contract(), fingerprint_policy)
            for key, tensor in previous.items():
                torch.testing.assert_close(tensor, worker.network.trainable_state_dict()[key], rtol=0, atol=0)
            worker.train_step(batch(29), batch(30))
            worker.save(path)
            restored = TAEEngine.load(path, 'cpu', vae=TinyFrozenVAE())
            self.assertEqual(restored.config.precision, 'fp32')
            self.assertEqual(restored.config.batch_size, 8)
            self.assertEqual(restored.config.ct_mode, 'rct')
            self.assertEqual(restored.stage_index, 1)

    def test_lct_and_expanded_mask_stage_keeps_weights_and_saves_new_settings(self):
        from dataclasses import replace
        worker = TAEEngine(replace(config(), ct_mode='rct'), device='cpu', seed=7, vae=TinyFrozenVAE())
        worker.train_step(batch(31), batch(32))
        previous = {key: value.clone() for key, value in worker.network.trainable_state_dict().items()}
        worker.data_state = {'sentinel': 'preserve sampler'}
        self.assertTrue(worker.reconfigure_decoder(replace(worker.config, ct_mode='lct',
                                                          training_mask_mode='expanded_feather')))
        self.assertEqual(worker.iteration, 1)
        self.assertEqual(worker.stage_updates, 0)
        self.assertFalse(worker.optimizer.state)
        self.assertEqual(worker.data_state, {'sentinel': 'preserve sampler'})
        for key, tensor in previous.items():
            torch.testing.assert_close(tensor, worker.network.trainable_state_dict()[key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'tae.pt'
            worker.save(checkpoint)
            restored = TAEEngine.load(checkpoint, 'cpu', vae=TinyFrozenVAE())
            self.assertEqual(restored.config.ct_mode, 'lct')
            self.assertEqual(restored.config.training_mask_mode, 'expanded_feather')
            self.assertEqual(restored.stage_history[-1]['started_at_iteration'], 1)
            self.assertEqual(restored.stage_index, 1)

    def test_disable_warp_retains_legacy_stage_weights_and_iteration(self):
        from dataclasses import replace
        worker = TAEEngine(replace(config(), random_warp=True), device='cpu', seed=7, vae=TinyFrozenVAE())
        worker.train_step(batch(31), batch(32))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'tae.pt'
            worker.save(checkpoint)
            old = torch.load(checkpoint, map_location='cpu', weights_only=True)
            old['stage_history'][0].pop('random_warp')
            torch.save(old, checkpoint)
            restored = TAEEngine.load(checkpoint, 'cpu', vae=TinyFrozenVAE())
            previous = restored.network.trainable_state_dict()
            self.assertTrue(restored.reconfigure_decoder(replace(restored.config, random_warp=False)))
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.stage_updates, 0)
            self.assertFalse(restored.optimizer.state)
            self.assertTrue(restored.stage_history[0]['random_warp'])
            self.assertFalse(restored.stage_history[1]['random_warp'])
            for key, tensor in previous.items():
                torch.testing.assert_close(tensor, restored.network.trainable_state_dict()[key], rtol=0, atol=0)
            restored.save(checkpoint)
            self.assertFalse(TAEEngine.load(checkpoint, 'cpu', vae=TinyFrozenVAE()).config.random_warp)

    def test_decoder_stages_keep_weights_reset_adam_and_resume_exactly(self):
        from dataclasses import replace
        worker = engine()
        worker.train_step(batch(31), batch(32))
        encoder = {k: v.clone() for k, v in worker.network.vae.encoder.state_dict().items()}
        for mode in ('tail', 'full', 'frozen'):
            before = worker.network.trainable_state_dict()
            old_iteration = worker.iteration
            changed = worker.reconfigure_decoder(replace(worker.config, decoder_mode=mode))
            self.assertTrue(changed)
            self.assertEqual(worker.stage_updates, 0)
            self.assertFalse(worker.optimizer.state)
            self.assertEqual(worker.iteration, old_iteration)
            for key, value in before.items():
                torch.testing.assert_close(worker.network.trainable_state_dict()[key], value, rtol=0, atol=0)
            worker.train_step(batch(31), batch(32))
            after = worker.network.trainable_state_dict()
            for key in before:
                if key.startswith('vae.decoder.'):
                    trainable = dict(worker.network.named_parameters())[key].requires_grad
                    self.assertEqual(not torch.equal(before[key], after[key]), trainable)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'tae.pt'
                prediction = worker.predict(batch(31)[0])
                worker.save(path)
                resumed = TAEEngine.load(path, device='cpu', vae=TinyFrozenVAE())
                inference = TAEEngine.load(path, device='cpu', vae=TinyFrozenVAE(), training=False)
                for actual in (resumed.predict(batch(31)[0]), inference.predict(batch(31)[0])):
                    for a, b in zip(prediction, actual):
                        np.testing.assert_array_equal(a, b)
                worker.train_step(batch(31), batch(32))
                resumed.train_step(batch(31), batch(32))
                for key, value in worker.network.trainable_state_dict().items():
                    torch.testing.assert_close(resumed.network.trainable_state_dict()[key], value, rtol=0, atol=0)
            for key, value in encoder.items():
                torch.testing.assert_close(worker.network.vae.encoder.state_dict()[key], value, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            worker.reconfigure_decoder(replace(worker.config, resolution=448))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_updates_liae_once_and_keeps_encoder_frozen(self):
        worker = engine()
        before = worker.network.trainable_state_dict()
        frozen = {name: value.clone() for name, value in worker.network.vae.state_dict().items()}
        worker.train_step(batch(1), batch(2))
        self.assertEqual(worker.iteration, 1)
        self.assertEqual(worker.optimizer_updates, 1)
        after = worker.network.trainable_state_dict()
        self.assertTrue(any(not torch.equal(before[key], after[key]) for key in before))
        for key, value in worker.network.vae.state_dict().items():
            self.assertTrue(torch.equal(frozen[key], value))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in worker.network.vae.parameters()))
        self.assertFalse(worker.network.vae.training)

    def test_save_resume_matches_uninterrupted_second_adam_update(self):
        source, destination = batch(3), batch(4)
        uninterrupted = engine()
        uninterrupted.train_step(source, destination)
        uninterrupted.train_step(source, destination)
        interrupted = engine()
        interrupted.train_step(source, destination)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tae.pt"
            state = {"source": {"exposures": [1, 0]}, "destination": {"exposures": [0, 1]}}
            interrupted.save(path, data_state=state)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertFalse(any(key.startswith("vae.encoder.") for key in payload["model"]))
            self.assertTrue(any(key.startswith("vae.decoder.") for key in payload["model"]))
            self.assertFalse(any("vae.model." in key for key in nested_keys(payload)))
            resumed = TAEEngine.load(path, device="cpu", vae=TinyFrozenVAE())
            self.assertEqual(resumed.iteration, 1)
            self.assertEqual(resumed.optimizer_updates, 1)
            self.assertEqual(resumed.data_state, state)
            resumed.train_step(source, destination)
        self.assertEqual(resumed.iteration, uninterrupted.iteration)
        self.assertEqual(resumed.optimizer_updates, uninterrupted.optimizer_updates)
        for key, tensor in uninterrupted.network.trainable_state_dict().items():
            torch.testing.assert_close(resumed.network.trainable_state_dict()[key], tensor, rtol=0, atol=0)
        saved_optim, actual_optim = uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
        self.assertEqual(saved_optim["param_groups"], actual_optim["param_groups"])
        for key, state in saved_optim["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(actual_optim["state"][key][field], value, rtol=0, atol=0)

    def test_reject_nonfinite_input_without_partial_update(self):
        worker = engine()
        before = worker.network.trainable_state_dict()
        bad = list(batch(5))
        bad[0][0, 0, 0, 0] = np.nan
        with self.assertRaises((ValueError, FloatingPointError)):
            worker.train_step(tuple(bad), batch(6))
        self.assertEqual(worker.iteration, 0)
        self.assertEqual(worker.optimizer_updates, 0)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, worker.network.trainable_state_dict()[key]))
        # The next valid update must remain possible after the rejected input.
        worker.train_step(batch(5), batch(6))
        self.assertEqual(worker.optimizer_updates, 1)

    def test_reject_nonfinite_decoder_output_without_partial_update(self):
        worker = engine()
        before = worker.network.trainable_state_dict()
        worker.network.vae.nan_decode = True
        with self.assertRaises((ValueError, FloatingPointError)):
            worker.train_step(batch(7), batch(8))
        self.assertEqual(worker.optimizer_updates, 0)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, worker.network.trainable_state_dict()[key]))
        worker.network.vae.nan_decode = False
        worker.train_step(batch(7), batch(8))
        self.assertEqual(worker.optimizer_updates, 1)

    def test_empty_face_mask_rejected(self):
        worker = engine()
        invalid = list(batch(1))
        invalid[2].fill(0)
        with self.assertRaises(ValueError):
            worker.train_step(tuple(invalid), batch(2))
        self.assertEqual(worker.optimizer_updates, 0)

    def test_nonfinite_backward_gradient_rejected_and_recoverable(self):
        worker = engine()
        before = worker.network.trainable_state_dict()
        parameter = worker.network.inter_AB.compress.weight
        hook = parameter.register_hook(lambda gradient: gradient * float("nan"))
        try:
            with self.assertRaises(FloatingPointError):
                worker.train_step(batch(13), batch(14))
        finally:
            hook.remove()
        self.assertEqual(worker.iteration, 0)
        self.assertEqual(worker.optimizer_updates, 0)
        self.assertFalse(worker.optimizer.state)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, worker.network.trainable_state_dict()[key]))
        self.assertTrue(all(parameter.grad is None for parameter in worker.network.trainable_parameters()))
        worker.train_step(batch(13), batch(14))
        self.assertEqual(worker.optimizer_updates, 1)

    def test_inplace_frozen_weight_change_is_detected(self):
        worker = engine()
        with torch.no_grad():
            worker.network.vae.encoder.weight.add_(.1)
        with self.assertRaisesRegex(RuntimeError, "Frozen|freeze|frozen"):
            worker.train_step(batch(15), batch(16))
        self.assertEqual(worker.optimizer_updates, 0)

    def test_training_bgr_conversion_and_fractional_priority_masks(self):
        worker = engine()
        source, destination = batch(9), batch(10)
        calls = []

        def capture(image, predicted_mask, prepared, cfg):
            calls.append({key: value.detach().clone() for key, value in prepared.items()})
            return reconstruction_loss(image, predicted_mask, prepared, cfg)

        with mock.patch("tae.engine.reconstruction_loss", side_effect=capture):
            worker.train_step(source, destination)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(worker.network.vae.encoded_inputs), 2)
        for index, original in enumerate((source, destination)):
            torch.testing.assert_close(worker.network.vae.encoded_inputs[index],
                                       torch.from_numpy(original[0][:, ::-1].copy()))
            torch.testing.assert_close(calls[index]["target"],
                                       torch.from_numpy(original[1][:, ::-1].copy()))
            torch.testing.assert_close(calls[index]["mask"], torch.from_numpy(original[2]))
            torch.testing.assert_close(calls[index]["eyes_mouth"],
                                       torch.from_numpy(np.clip(original[3] - 1, 0, 1)))

    def test_prediction_bgr_and_dfl_mask_order(self):
        worker = engine()
        bgr = batch(11)[0]
        image = torch.empty((1, 3, 256, 256))
        image[:, 0], image[:, 1], image[:, 2] = .2, .4, .8
        swap_mask = torch.full((1, 1, 256, 256), .3)
        dst_mask = torch.full((1, 1, 256, 256), .7)
        with mock.patch.object(worker.network, "predict", return_value={
            "swap": image, "swap_mask": swap_mask, "dst_mask": dst_mask,
        }) as predict:
            output = worker.predict(bgr)
        self.assertEqual(len(output), 3)
        torch.testing.assert_close(predict.call_args.args[0], torch.from_numpy(bgr[:, ::-1].copy()))
        torch.testing.assert_close(torch.as_tensor(output[0]), image[:, [2, 1, 0]])
        torch.testing.assert_close(torch.as_tensor(output[1]), swap_mask)
        torch.testing.assert_close(torch.as_tensor(output[2]), dst_mask)

    def test_prediction_rejects_bad_pixels_before_network_and_each_nonfinite_output(self):
        worker = engine()
        good = batch(11)[0]
        with mock.patch.object(worker.network, 'predict') as predict:
            for invalid in (float('nan'), float('inf'), -float('inf'), -.01, 1.01):
                value = good.copy(); value[0, 0, 0, 0] = invalid
                with self.assertRaisesRegex(ValueError, 'finite BGR'):
                    worker.predict(value)
            predict.assert_not_called()
        outputs = {'swap': torch.full((1, 3, 256, 256), .5),
                   'swap_mask': torch.ones(1, 1, 256, 256),
                   'dst_mask': torch.ones(1, 1, 256, 256)}
        for name in outputs:
            for invalid in (float('nan'), float('inf'), -float('inf')):
                corrupted = {key: value.clone() for key, value in outputs.items()}
                corrupted[name][0, 0, 0, 0] = invalid
                with mock.patch.object(worker.network, 'predict', return_value=corrupted):
                    with self.assertRaisesRegex(FloatingPointError, 'Nonfinite TAE prediction'):
                        worker.predict(good)

    def test_checkpoint_rejects_different_frozen_asset_contract(self):
        worker = engine()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tae.pt"
            worker.save(path)
            other = TinyFrozenVAE()
            other.weight_sha256 = "different-frozen-asset"
            with self.assertRaises(ValueError):
                TAEEngine.load(path, device="cpu", vae=other)

    def test_checkpoint_rejects_missing_adam_history(self):
        worker = engine()
        worker.train_step(batch(17), batch(18))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tae.pt"
            worker.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["optimizer"]["state"].pop(next(iter(payload["optimizer"]["state"])))
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "AdamW"):
                TAEEngine.load(path, device="cpu", vae=TinyFrozenVAE())

    def test_checkpoint_rejects_negative_adam_second_moment(self):
        worker = engine()
        worker.train_step(batch(17), batch(18))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tae.pt"
            worker.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            entry = next(iter(payload["optimizer"]["state"].values()))
            entry["exp_avg_sq"].view(-1)[0] = -.1
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                TAEEngine.load(path, device="cpu", vae=TinyFrozenVAE())

    def test_checkpoint_rejects_optimizer_setting_tamper(self):
        worker = engine()
        worker.train_step(batch(19), batch(20))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tae.pt"
            worker.save(path)
            original = torch.load(path, map_location="cpu", weights_only=True)
            for field, replacement in (("weight_decay", .5), ("betas", (.5, .5)), ("eps", .2)):
                with self.subTest(field=field):
                    payload = copy.deepcopy(original)
                    payload["optimizer"]["param_groups"][0][field] = replacement
                    torch.save(payload, path)
                    with self.assertRaises(ValueError):
                        TAEEngine.load(path, device="cpu", vae=TinyFrozenVAE())

    def test_failure_after_backward_clears_partial_gradients(self):
        worker = engine()
        before = worker.network.trainable_state_dict()
        with mock.patch("tae.engine.torch.nn.utils.clip_grad_norm_", side_effect=FloatingPointError("injected norm overflow")):
            with self.assertRaises(FloatingPointError):
                worker.train_step(batch(21), batch(22))
        self.assertEqual(worker.optimizer_updates, 0)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, worker.network.trainable_state_dict()[key]))
        self.assertTrue(all(parameter.grad is None for parameter in worker.network.trainable_parameters()))
        worker.train_step(batch(21), batch(22))
        self.assertEqual(worker.optimizer_updates, 1)

    def test_checkpoint_after_rejected_step_preserves_committed_warmup_lr(self):
        for prior_steps in (0, 1):
            with self.subTest(prior_steps=prior_steps):
                worker = engine()
                for _ in range(prior_steps):
                    worker.train_step(batch(23), batch(24))
                prior_lr = worker.optimizer.param_groups[0]["lr"]
                with mock.patch("tae.engine.torch.nn.utils.clip_grad_norm_", side_effect=FloatingPointError("injected failure")):
                    with self.assertRaises(FloatingPointError):
                        worker.train_step(batch(23), batch(24))
                self.assertEqual(worker.optimizer.param_groups[0]["lr"], prior_lr)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "tae.pt"
                    worker.save(path)
                    restored = TAEEngine.load(path, device="cpu", vae=TinyFrozenVAE())
                    self.assertEqual(restored.optimizer_updates, prior_steps)
                    restored.train_step(batch(23), batch(24))
                    self.assertEqual(restored.optimizer_updates, prior_steps + 1)


if __name__ == "__main__":
    unittest.main()
