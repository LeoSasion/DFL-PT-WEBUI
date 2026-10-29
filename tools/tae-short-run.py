"""Bounded real-TAE functional acceptance using real facesets or artificial DFL fixtures.

This verifies training/save/resume/prediction/control, not learned identity or
visual quality. The outer process enforces a five-minute hard timeout.
"""
import argparse
from argparse import Namespace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_internal" / "DeepFaceLab"))


def worker(args):
    import numpy as np
    import torch
    from tae.config import TrainConfig
    from tae.engine import TAEEngine
    from tae.checkpoint import frozen_digest, read_checkpoint, file_sha256
    from tae.data import build_pair
    from me_backend.web_bridge import _run_training, TrainingHeartbeat, atomic_json
    from tests.me_fixtures import make_aligned

    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def fixtures(directory, offset):
        if args.resolution <= 128:
            return make_aligned(directory, count=3, offset=offset)
        # Generate native pixels at this resolution, never enlarge test targets.
        import cv2
        from DFLIMG import DFLJPG
        from facelib import LandmarksProcessor
        directory.mkdir(parents=True)
        size = args.resolution
        yy, xx = np.mgrid[:size, :size]
        for index in range(3):
            image = np.stack(((xx+index*7+offset)%size/(size-1), (yy+index*3)%size/(size-1),
                              .35+.2*np.sin((xx+yy+offset)/18)), axis=-1)
            path = directory / f'{index:03d}.jpg'
            ok, encoded = cv2.imencode('.jpg', (image*255).astype(np.uint8))
            assert ok
            encoded.tofile(path)
            dfl = DFLJPG.load(str(path))
            dfl.set_face_type('full_face')
            dfl.set_landmarks(LandmarksProcessor.get_canonical_68(size))
            dfl.set_xseg_mask((((xx-size*.5)/(size*.34))**2+((yy-size*.5)/(size*.44))**2 < 1).astype(np.float32)[..., None])
            dfl.save()
        return directory
    source = Path(args.src).resolve() if args.src else fixtures(output / "src", 0)
    destination = Path(args.dst).resolve() if args.dst else fixtures(output / "dst", 49)
    config = TrainConfig(resolution=args.resolution, face_type=args.face_type if args.src else "f", inter_dims=args.inter_dims, hidden_dims=args.hidden_dims,
                         batch_size=1, warmup_steps=0, random_warp=True, decoder_mode=args.decoder_mode, precision=args.precision)
    model_dir = output / "model"
    options = Namespace(steps=args.steps, save_every=args.steps, preview_every=args.steps, max_minutes=2,
        target_iterations=0, debug_samples=False, src=source, dst=destination)
    print("Loading real TAEQI2.1 asset for bounded synthetic acceptance", flush=True)
    engine = TAEEngine(config, args.device, seed=11, weights=args.weights)
    before = frozen_digest(engine.network.vae.encoder)
    trainable_before = frozen_digest(engine.network.inter_AB)
    decoder_before = {name: tensor.detach().cpu().clone() for name, tensor in engine.network.vae.decoder.named_parameters()}
    from tae.losses import reconstruction_loss
    with build_pair(source, destination, config, 19) as probe_pair:
        probe_src, probe_dst = probe_pair.batch()
    def fixed_probe():
        with torch.no_grad(), engine.autocast():
            losses = []
            for domain, values in (("src", probe_src), ("dst", probe_dst)):
                batch = engine._batch(values)
                out = engine.network.reconstruct(batch["input"], domain)
                with torch.autocast(device_type=engine.device.type, enabled=False):
                    loss, _ = reconstruction_loss(out["image"], out["mask"], batch, config)
                losses.append(float(loss))
            return losses
    fixed_before = fixed_probe()
    with build_pair(source, destination, config, 7) as pair:
        _run_training(options, engine, "tae-functional", model_dir, pair, TrainingHeartbeat(None, 0))
    assert engine.iteration == args.steps
    assert frozen_digest(engine.network.vae.encoder) == before
    assert frozen_digest(engine.network.inter_AB) != trainable_before
    test_input = np.full((1, 3, args.resolution, args.resolution), .4, np.float32)
    prediction = engine.predict(test_input)
    decoder_changed = {name: not torch.equal(decoder_before[name], tensor.detach().cpu())
                       for name, tensor in engine.network.vae.decoder.named_parameters()}
    for name, tensor in engine.network.vae.decoder.named_parameters():
        if not tensor.requires_grad:
            assert not decoder_changed[name], name
    assert any(decoder_changed.values()) == (args.decoder_mode != 'frozen')
    assert all(np.isfinite(value).all() for value in prediction)
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Reloading saved AdamW/model and checking identical prediction", flush=True)
    engine = TAEEngine.load(model_dir / "tae.pt", args.device, training=True)
    resumed_prediction = engine.predict(test_input)
    max_diff = max(float(np.max(np.abs(left-right))) for left, right in zip(prediction, resumed_prediction))
    assert max_diff == 0.0, max_diff
    assert frozen_digest(engine.network.vae.encoder) == before
    with build_pair(source, destination, config, 99) as pair:
        pair.load_state_dict(engine.data_state)
        _run_training(options, engine, "tae-functional", model_dir, pair, TrainingHeartbeat(None, args.steps))
    assert engine.iteration == args.steps * 2
    assert frozen_digest(engine.network.vae.encoder) == before
    # Actual WebUI close request/ack at a completed iteration, without training.
    control = output / "control.jsonl"
    ack = output / "ack.json"
    control.write_text(json.dumps({"operation": "close", "requestedAt": "tae-functional-close"})+"\n", encoding="utf-8")
    os.environ["DFL_WEB_CONTROL_FILE"] = str(control)
    os.environ["DFL_WEB_CONTROL_ACK_FILE"] = str(ack)
    with build_pair(source, destination, config, 99) as pair:
        pair.load_state_dict(read_checkpoint(model_dir / "tae.pt")["data_state"])
        _run_training(options, engine, "tae-functional", model_dir, pair, TrainingHeartbeat(None, args.steps * 2))
    assert engine.iteration == args.steps * 2
    ack_data = json.loads(ack.read_text())
    assert ack_data["status"] == "completed" and ack_data["operation"] == "close"
    payload = read_checkpoint(model_dir / "tae.pt")
    assert not any(key.startswith("vae.encoder.") for key in payload["model"])
    assert any(key.startswith("vae.decoder.") for key in payload["model"])
    metadata = json.loads((model_dir / "metadata.json").read_text())
    assert metadata["modelClass"] == "TAE" and not (model_dir / "me.pt").exists()
    history = [json.loads(line) for line in (model_dir / "loss-history.jsonl").read_text().splitlines()]
    assert [line["iteration"] for line in history] == list(range(1, args.steps * 2 + 1))
    assert all(line["encoder_frozen"] for line in history)
    result = {"status": "passed", "kind": "functional-real-TAEQI2.1", "data_kind": "real-native-aligned" if args.src else "artificial-DFL-fixtures",
        "config": config.to_dict(), "base": engine.metadata(), "steps": args.steps * 2, "decoder_mode": args.decoder_mode,
        "decoder_changed_tensors": sum(decoder_changed.values()), "fixed_probe_loss_before": fixed_before, "fixed_probe_loss_after": fixed_probe(),
        "frozen_runtime_sha256_before_after": before, "prediction_reload_max_abs_error": max_diff,
        "checkpoint_sha256": file_sha256(model_dir / "tae.pt"),
        "checkpoint_excludes_encoder_includes_decoder": True, "close_ack": ack_data,
        "seconds": time.perf_counter()-started, "losses": history,
        "quality_claim": False, "formal_identity_training": False,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if engine.device.type == "cuda" else None}
    atomic_json(output / "result.json", result)
    print(json.dumps({"status": "passed", "result": str(output / "result.json"), "seconds": result["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights")
    parser.add_argument("--decoder-mode", choices=("frozen", "tail", "full"), default="frozen")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--src")
    parser.add_argument("--dst")
    parser.add_argument("--face-type", default="wf")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, choices=(256, 448, 640, 832), default=256)
    parser.add_argument("--inter-dims", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, default=96)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.src) != bool(args.dst) or not 1 <= args.steps <= 100:
        raise ValueError("Provide both facesets; bounded steps must be 1..100")
    if args.worker:
        worker(args)
        return
    if Path(args.output).exists():
        raise FileExistsError("Choose a new acceptance output directory")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--output", args.output, "--device", args.device, "--resolution", str(args.resolution),
               "--inter-dims", str(args.inter_dims), "--hidden-dims", str(args.hidden_dims)]
    command += ["--decoder-mode", args.decoder_mode, "--precision", args.precision, "--steps", str(args.steps), "--face-type", args.face_type]
    if args.weights:
        command += ["--weights", args.weights]
    if args.src:
        command += ["--src", args.src, "--dst", args.dst]
    subprocess.run(command, cwd=ROOT, check=True, timeout=300)


if __name__ == "__main__":
    main()
