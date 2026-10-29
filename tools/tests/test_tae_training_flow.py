"""Shared ME runner/controls with native DFL data and injected tiny TAE codec."""

from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DFL = ROOT / "_internal" / "DeepFaceLab"
sys.path.insert(0, str(DFL))
sys.path.insert(0, str(Path(__file__).parent))

from tae.config import TrainConfig
from tae.data import build_pair
from tae.engine import TAEEngine
from tae.train import train_web
from me_backend.web_bridge import TrainingHeartbeat, WebControl, _run_training
from test_tae_engine import TinyFrozenVAE
from functools import partial
from tests.me_fixtures import make_aligned as _make_aligned
make_aligned = partial(_make_aligned, size=256)

spec = importlib.util.spec_from_file_location("tae_cli_for_test", DFL / "tae.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class InjectedEngine(TAEEngine):
    """Use all production engine behavior with a fixed tiny codec fixture."""
    def __init__(self, *args, **kwargs):
        kwargs["vae"] = kwargs.get("vae") or TinyFrozenVAE()
        super().__init__(*args, **kwargs)


class TrainingFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = make_aligned(self.root / "src", count=3)
        self.dst = make_aligned(self.root / "dst", count=3, offset=39)
        self.config = TrainConfig(resolution=256, face_type="f", inter_dims=8, hidden_dims=8,
                                  batch_size=1, random_warp=True, random_flip=True,
                                  warmup_steps=3, learning_rate=1e-3)
        self.environment = {name: str(self.root / filename) for name, filename in (
            ("DFL_WEB_CONTROL_FILE", "control.jsonl"),
            ("DFL_WEB_CONTROL_ACK_FILE", "control-ack.json"),
            ("DFL_WEB_HEARTBEAT_FILE", "heartbeat.json"),
            ("DFL_WEB_PREVIEW_FILE", "web-preview.png"),
        )}
        self.environment["TAE_VAE_WEIGHTS"] = ""
        patch = mock.patch.dict(os.environ, self.environment)
        patch.start()
        self.addCleanup(patch.stop)

    def arguments(self, model="model", *, steps=1, extra=()):
        return cli.parser().parse_args(["train", "--src", str(self.src), "--dst", str(self.dst),
            "--model", str(self.root / model), "--name", "synthetic-tae", "--device", "cpu",
            "--steps", str(steps), "--seed", "7", "--save-every", "100", "--preview-every", "100",
            "--config-json", json.dumps(self.config.to_dict()), *extra])

    def run_train(self, args):
        with mock.patch("tae.train.TAEEngine", InjectedEngine), redirect_stdout(io.StringIO()):
            train_web(args)

    def test_pretraining_512_checkpoint_prediction_resume_and_mode_guard(self):
        from tae.data import _pretraining_holdout
        from DFLIMG import DFLJPG
        self.src = _make_aligned(self.root / 'pretrain512', count=4, size=512)
        self.dst = self.src
        held = next(f'g{n}' for n in range(1000) if _pretraining_holdout('source-id:' + f'g{n}'))
        trained = next(f'g{n}' for n in range(1000) if not _pretraining_holdout('source-id:' + f'g{n}'))
        for index, path in enumerate(sorted(self.src.glob('*.jpg'))):
            dfl = DFLJPG.load(str(path))
            dfl.get_dict()['source_id'] = held if index == 0 else trained
            dfl.save()
        self.config.resolution = 512
        self.run_train(self.arguments(extra=('--pretraining',)))
        checkpoint = self.root / 'model/tae.pt'
        model = InjectedEngine.load(checkpoint, device='cpu')
        self.assertEqual(model.training_mode, 'pretraining')
        self.assertEqual(model.network.inter_AB.grid, 8)
        with build_pair(self.src, self.dst, self.config, 7, pretraining=True) as pair:
            outputs = model.predict(pair.source_data.batch()[0])
            self.assertEqual(outputs[0].shape, (1, 3, 512, 512))
            self.assertTrue(all(np.isfinite(x).all() for x in outputs))
        before = checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, 'preserve pretraining'):
            self.run_train(self.arguments(extra=('--resume',)))
        self.assertEqual(checkpoint.read_bytes(), before)
        self.run_train(self.arguments(extra=('--resume', '--pretraining')))
        self.assertEqual(InjectedEngine.load(checkpoint, device='cpu').iteration, 2)

    def test_fixed_deadline_does_not_restart_the_relative_budget(self):
        from datetime import datetime, timezone
        args = self.arguments(steps=0)
        worker = InjectedEngine(self.config, device='cpu', seed=7)
        cutoff = datetime(2026, 10, 17, 7, 48, tzinfo=timezone.utc)
        # Preparation already consumed most of the original budget: ten seconds
        # remain, even though this invocation requests sixty more minutes.
        current = datetime(2026, 10, 17, 7, 47, 50, tzinfo=timezone.utc)
        clock = [100.]
        train_step = worker.train_step
        def advance(*batches):
            result = train_step(*batches)
            clock[0] += 11
            return result
        heartbeat = mock.MagicMock()
        folder = self.root / 'deadline'
        with build_pair(self.src, self.dst, self.config, 7) as pair, \
             mock.patch('me_backend.web_bridge.time.monotonic', side_effect=lambda: clock[0]), \
             mock.patch('me_backend.web_bridge.datetime', wraps=datetime) as wall, \
             mock.patch.object(worker, 'train_step', side_effect=advance), redirect_stdout(io.StringIO()):
            wall.now.return_value = current
            _run_training(args, worker, 'deadline', folder, pair, heartbeat, stop_at_utc=cutoff)
        self.assertEqual(worker.iteration, 1)
        self.assertEqual(InjectedEngine.load(folder / 'tae.pt', 'cpu').iteration, 1)

    def test_expired_fixed_deadline_saves_without_training(self):
        args = self.arguments(steps=0, extra=('--stop-at-utc', '2000-01-01T00:00:00Z'))
        self.run_train(args)
        self.assertEqual(InjectedEngine.load(self.root / 'model/tae.pt', 'cpu').iteration, 0)
        args = self.arguments(extra=('--stop-at-utc', '2026-10-17T07:48:00'))
        with self.assertRaisesRegex(ValueError, 'timezone'):
            self.run_train(args)

    def test_shared_training_metadata_checkpoint_and_exact_resume_cursor(self):
        self.run_train(self.arguments("uninterrupted", steps=2))
        self.run_train(self.arguments("resumed", steps=1))
        first = torch.load(self.root / "resumed" / "tae.pt", map_location="cpu", weights_only=True)
        self.assertEqual(sum(first["data_state"]["src"]["exposures"]), 1)
        self.assertEqual(sum(first["data_state"]["dst"]["exposures"]), 1)
        self.run_train(self.arguments("resumed", steps=1, extra=("--resume",)))
        expected = InjectedEngine.load(self.root / "uninterrupted" / "tae.pt", device="cpu")
        actual = InjectedEngine.load(self.root / "resumed" / "tae.pt", device="cpu")
        self.assertEqual(actual.iteration, 2)
        self.assertEqual(actual.data_state, expected.data_state)
        for name, tensor in expected.network.trainable_state_dict().items():
            torch.testing.assert_close(actual.network.trainable_state_dict()[name], tensor, rtol=0, atol=0)
        folder = self.root / "resumed"
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["modelClass"], "TAE")
        self.assertEqual(metadata["checkpoint"], "tae.pt")
        self.assertEqual(metadata["format"], "tae-liae-pytorch")
        self.assertEqual(metadata["iteration"], 2)
        self.assertFalse((folder / "me.pt").exists())
        self.assertTrue((folder / "preview.png").is_file())
        history = [json.loads(line) for line in (folder / "loss-history.jsonl").read_text().splitlines()]
        self.assertEqual([row["iteration"] for row in history], [1, 2])
        heartbeat = json.loads(Path(self.environment["DFL_WEB_HEARTBEAT_FILE"]).read_text())
        self.assertEqual(heartbeat["phase"], "finished")
        self.assertEqual(heartbeat["iteration"], 2)

    def test_save_preview_close_controls_ack_before_update_and_resume(self):
        requests = [{"operation": operation, "requestedAt": f"synthetic-{operation}"}
                    for operation in ("save", "preview", "close")]
        control = Path(self.environment["DFL_WEB_CONTROL_FILE"])
        control.write_text("".join(json.dumps(value) + "\n" for value in requests), encoding="utf-8")
        complete = WebControl.complete
        acks = []

        def capture(controller, *args, **kwargs):
            complete(controller, *args, **kwargs)
            acks.append(json.loads(Path(self.environment["DFL_WEB_CONTROL_ACK_FILE"]).read_text()))

        with mock.patch.object(WebControl, "complete", capture):
            self.run_train(self.arguments(steps=0))
        self.assertEqual([value["operation"] for value in acks], ["save", "preview", "close"])
        for ack in acks:
            self.assertEqual(ack["status"], "completed")
            self.assertEqual(ack["iteration"], 0)
            self.assertEqual(Path(ack["checkpoint"]).name, "tae.pt")
        self.assertTrue(Path(self.environment["DFL_WEB_PREVIEW_FILE"]).is_file())
        saved = torch.load(self.root / "model" / "tae.pt", map_location="cpu", weights_only=True)
        self.assertEqual(sum(saved["data_state"]["src"]["exposures"]), 0)
        control.write_text("", encoding="utf-8")
        self.run_train(self.arguments(steps=1, extra=("--resume",)))
        loaded = InjectedEngine.load(self.root / "model" / "tae.pt", device="cpu")
        self.assertEqual(loaded.iteration, 1)

    def test_debug_samples_does_not_mutate_existing_checkpoint_or_cursor(self):
        self.run_train(self.arguments())
        path = self.root / "model" / "tae.pt"
        before = path.read_bytes()
        self.run_train(self.arguments(extra=("--resume", "--debug-samples")))
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue((self.root / "model" / "debug-src.png").is_file())
        self.assertTrue((self.root / "model" / "debug-dst.png").is_file())

    def test_decoder_stage_resume_keeps_data_cursor_without_reset(self):
        self.run_train(self.arguments(steps=2))
        path = self.root / "model" / "tae.pt"
        initial = InjectedEngine.load(path, device="cpu")
        for index, mode in enumerate(('tail', 'full', 'frozen'), 1):
            args = self.arguments(steps=1, extra=("--resume",))
            args.config_json = json.dumps({"decoder_mode": mode})
            self.run_train(args)
            restored = InjectedEngine.load(path, device="cpu")
            self.assertEqual(restored.iteration, 2 + index)
            self.assertEqual(restored.stage_updates, 1)
            self.assertEqual(restored.stage_index, index)
            self.assertEqual(restored.config.decoder_mode, mode)
            for side in ('src', 'dst'):
                self.assertEqual(restored.data_state[side]['fingerprint'], initial.data_state[side]['fingerprint'])
                self.assertEqual(sum(restored.data_state[side]['exposures']), (2 + index) * restored.config.batch_size)

    def test_reject_resume_config_change_and_me_directory_without_writing(self):
        self.run_train(self.arguments())
        path = self.root / "model" / "tae.pt"
        before = path.read_bytes()
        args = self.arguments(extra=("--resume",))
        args.config_json = json.dumps({"grad_clip": .5})
        with self.assertRaisesRegex(ValueError, "preserves architecture"):
            self.run_train(args)
        self.assertEqual(path.read_bytes(), before)
        folder = self.root / "existing-me"
        folder.mkdir()
        (folder / "me.pt").write_bytes(b"existing ME sentinel")
        with self.assertRaisesRegex(ValueError, "separate TAE"):
            self.run_train(self.arguments("existing-me"))
        self.assertEqual((folder / "me.pt").read_bytes(), b"existing ME sentinel")
        self.assertEqual(len(list(folder.iterdir())), 1)

    def test_learning_rate_resume_records_stage_and_preserves_sampler(self):
        self.run_train(self.arguments())
        before = torch.load(self.root/'model/tae.pt', map_location='cpu', weights_only=True)
        before['stage_history'][0].pop('learning_rate')
        torch.save(before, self.root/'model/tae.pt')
        args = self.arguments(extra=('--resume',))
        args.config_json = json.dumps({'learning_rate': self.config.learning_rate/2})
        self.run_train(args)
        restored = InjectedEngine.load(self.root/'model/tae.pt', 'cpu')
        self.assertEqual(restored.iteration, 2)
        self.assertEqual(restored.stage_index, 1)
        self.assertEqual(restored.stage_updates, 1)
        self.assertEqual(restored.stage_history[0]['learning_rate'], self.config.learning_rate)
        self.assertEqual(restored.stage_history[1]['learning_rate'], self.config.learning_rate/2)
        self.assertEqual(sum(restored.data_state['src']['exposures']), 2)
        self.assertAlmostEqual(restored.optimizer.param_groups[0]['lr'], self.config.learning_rate/2/3)

    def test_eyes_mouth_resume_records_stage_and_preserves_sampler(self):
        self.run_train(self.arguments())
        before = torch.load(self.root/'model/tae.pt', map_location='cpu', weights_only=True)
        before['stage_history'][0].pop('eyes_mouth_weight')
        torch.save(before, self.root/'model/tae.pt')
        args = self.arguments(extra=('--resume',))
        args.config_json = json.dumps({'eyes_mouth_weight': self.config.eyes_mouth_weight*1.5})
        self.run_train(args)
        restored = InjectedEngine.load(self.root/'model/tae.pt', 'cpu')
        self.assertEqual(restored.iteration, 2)
        self.assertEqual(restored.stage_index, 1)
        self.assertEqual(restored.stage_updates, 1)
        self.assertEqual(restored.stage_history[0]['eyes_mouth_weight'], self.config.eyes_mouth_weight)
        self.assertEqual(restored.stage_history[1]['eyes_mouth_weight'], self.config.eyes_mouth_weight*1.5)
        self.assertEqual(sum(restored.data_state['src']['exposures']), 2)
        self.assertAlmostEqual(restored.optimizer.param_groups[0]['lr'], self.config.learning_rate/3)

    def test_step_bound_does_not_decode_an_unused_next_batch(self):
        args = self.arguments(steps=1)
        worker = InjectedEngine(self.config, device="cpu", seed=7)
        with build_pair(self.src, self.dst, self.config, 7) as pair:
            original_batch = pair.batch
            calls = 0

            def bounded_batch():
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise ValueError("A completed run must not fetch another batch")
                return original_batch()

            with mock.patch.object(pair, "batch", bounded_batch), redirect_stdout(io.StringIO()):
                _run_training(args, worker, args.name, Path(args.model), pair, TrainingHeartbeat(None, 0))
            self.assertEqual(calls, 1)
            self.assertEqual(worker.iteration, 1)
            saved = torch.load(Path(args.model) / "tae.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["iteration"], 1)

    def test_training_copies_exposure_pool_only_when_saving(self):
        class ObservedCounters(list):
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        args = self.arguments(steps=3)
        worker = InjectedEngine(self.config, device="cpu", seed=7)
        with build_pair(self.src, self.dst, self.config, 7) as pair:
            for dataset in (pair.source_data, pair.destination_data):
                dataset.exposures = ObservedCounters(dataset.exposures)
            with mock.patch.object(pair, "state_dict", side_effect=AssertionError("eager sampler copy")), \
                 mock.patch.object(pair.source_data, "state_dict", side_effect=AssertionError("eager exposure copy")), \
                 mock.patch.object(pair.destination_data, "state_dict", side_effect=AssertionError("eager exposure copy")), \
                 redirect_stdout(io.StringIO()):
                _run_training(args, worker, args.name, Path(args.model), pair, TrainingHeartbeat(None, 0))
            self.assertEqual(worker.iteration, 3)
            # Initial and final checkpoints are the only full counter copies.
            for dataset in (pair.source_data, pair.destination_data):
                self.assertEqual(dataset.exposures.iterations, 2)
                self.assertEqual(sum(dataset.exposures), 3)
            saved = torch.load(Path(args.model) / "tae.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["data_state"], pair.state_dict())

    def test_close_preview_failure_preserves_unconsumed_batch_for_resume(self):
        args = self.arguments(steps=0)
        worker = InjectedEngine(self.config, device="cpu", seed=7)
        original_step = worker.train_step
        control = Path(self.environment["DFL_WEB_CONTROL_FILE"])

        def step_then_close(*batches):
            result = original_step(*batches)
            control.write_text(json.dumps({"operation": "close", "requestedAt": "after-first-step"}) + "\n")
            return result

        with build_pair(self.src, self.dst, self.config, 7) as pair:
            with mock.patch.object(worker, "train_step", side_effect=step_then_close), \
                 mock.patch("me_backend.web_bridge.make_preview", side_effect=[
                     np.zeros((256, 256, 3), np.float32), ValueError("synthetic close preview failure")]), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "synthetic close preview failure"):
                    _run_training(args, worker, args.name, Path(args.model), pair, TrainingHeartbeat(None, 0))
            self.assertEqual(worker.iteration, 1)
            self.assertEqual(sum(pair.source_data.exposures), 2)
            self.assertEqual(sum(pair.destination_data.exposures), 2)
        saved = torch.load(Path(args.model) / "tae.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["iteration"], 1)
        for side in ("src", "dst"):
            self.assertEqual(sum(saved["data_state"][side]["exposures"]), 1)
        ack = json.loads(Path(self.environment["DFL_WEB_CONTROL_ACK_FILE"]).read_text())
        self.assertEqual(ack["status"], "failed")
        with build_pair(self.src, self.dst, self.config, 7) as expected, \
             build_pair(self.src, self.dst, self.config, 99) as resumed:
            expected.batch()
            resumed.load_state_dict(saved["data_state"])
            for actual_side, expected_side in zip(resumed.batch(), expected.batch()):
                for actual, wanted in zip(actual_side, expected_side):
                    np.testing.assert_array_equal(actual, wanted)

    def test_loop_wall_budget_stops_before_update_when_already_elapsed(self):
        args = self.arguments(steps=0)
        args.max_minutes = 1
        worker = InjectedEngine(self.config, device="cpu", seed=7)
        with build_pair(self.src, self.dst, self.config, 7) as pair:
            # First monotonic sample defines the deadline; subsequent checks
            # observe expired time. No sleeps or hardware runtime assumptions.
            samples = iter([0.0, 61.0])
            with mock.patch("me_backend.web_bridge.time.monotonic", side_effect=lambda: next(samples, 61.0)), redirect_stdout(io.StringIO()):
                _run_training(args, worker, args.name, Path(args.model), pair, TrainingHeartbeat(None, 0))
            self.assertEqual(worker.iteration, 0)
            saved = torch.load(Path(args.model) / "tae.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["iteration"], 0)
            self.assertEqual(sum(saved["data_state"]["src"]["exposures"]), 0)


if __name__ == "__main__":
    unittest.main()
