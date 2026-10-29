"""Parallel TAE CLI, reusing the ME WebUI training and DFL merger framework."""
from argparse import ArgumentParser
import json
from pathlib import Path


def parser():
    root = ArgumentParser(description="TAEQI2.1: frozen encoder, optional decoder training, PyTorch LIAE")
    sub = root.add_subparsers(dest="command", required=True)
    train = sub.add_parser("web-train", aliases=["train"])
    train.add_argument("--src", "--training-data-src-dir", required=True)
    train.add_argument("--dst", "--training-data-dst-dir", required=True)
    train.add_argument("--model", "--model-dir", required=True, help="Dedicated model instance directory")
    train.add_argument("--name", default="TAE")
    train.add_argument("--vae-weights", help="Pinned TAEQI2.1 safetensors path; defaults to bundled pretrained asset")
    from me_backend.cli_options import config_options
    config_options(train)
    train.add_argument("--resolution", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device", default="cuda")
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--steps", type=int, default=0, help="Updates this invocation; 0 uses time/target bound")
    train.add_argument("--target-iterations", type=int, default=0)
    train.add_argument("--max-minutes", type=float, default=60)
    train.add_argument("--stop-at-utc", help="Fixed ISO 8601 deadline with timezone; includes preparation on resumed runs")
    train.add_argument("--save-every", type=int, default=100)
    train.add_argument("--preview-every", type=int, default=100)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--reset-data-state", action="store_true")
    train.add_argument("--pretraining", action="store_true", help="Use one common SRC/DST pool, with deterministic 5 percent source-group holdout")
    train.add_argument("--debug-samples", action="store_true")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("checkpoint")
    validate = sub.add_parser("validate-data")
    validate.add_argument("--src", required=True)
    validate.add_argument("--dst", required=True)
    config_options(validate)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    from core.torch_runtime_switch import assert_runtime_available
    assert_runtime_available(Path(__file__).resolve().parents[2])
    if args.command in {"web-train", "train"}:
        import torch
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        from tae.train import train_web
        train_web(args)
    elif args.command == "inspect":
        from tae.checkpoint import read_checkpoint
        value = read_checkpoint(args.checkpoint)
        print(json.dumps({key: value[key] for key in ("format", "version", "config", "iteration", "vae_weights", "vae_sha256", "vae_contract", "architecture")}, indent=2))
    else:
        from tae.config import TrainConfig
        from tae.data import build_pair
        from me_backend.web_bridge import load_config_options
        with build_pair(args.src, args.dst, TrainConfig.from_dict(load_config_options(args)), 0) as pair:
            print(json.dumps({side: {"count": len(data.paths), "fingerprint": data.fingerprint,
                "source_groups": len(data.groups)} for side, data in (("src", pair.source_data), ("dst", pair.destination_data))}, indent=2))


if __name__ == "__main__":
    main()
