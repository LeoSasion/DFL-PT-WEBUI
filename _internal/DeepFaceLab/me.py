"""ME PyTorch training, prediction and explicit legacy weight conversion."""
import argparse
import json
from pathlib import Path
import os
import tempfile
import cv2
import numpy as np
import torch
from me_backend.config import MEConfig
from me_backend.data import read_aligned
from me_backend.engine import MEEngine


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, np.round(np.clip(image, 0, 1) * 255).astype(np.uint8))
    if not ok:
        raise IOError(f'Could not encode {path}')
    encoded.tofile(path)


def train(args):
    from me_backend.web_bridge import train_web
    if not getattr(args, 'name', None):
        args.name = Path(args.model).resolve().name
    train_web(args)


def infer(args):
    engine = MEEngine.load(Path(args.model) / 'me.pt', args.device)
    image, _, _ = read_aligned(Path(args.input), engine.config)
    swap, src_mask, dst_mask = engine.predict(image.transpose(2, 0, 1)[None])
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Inference output directory must be empty')
    save_image(output / 'swap.png', swap[0].transpose(1, 2, 0))
    save_image(output / 'src-mask.png', src_mask[0].transpose(1, 2, 0))
    save_image(output / 'dst-mask.png', dst_mask[0].transpose(1, 2, 0))
    mask = (src_mask * dst_mask)[0].transpose(1, 2, 0)
    save_image(output / 'aligned-composite.png', swap[0].transpose(1, 2, 0) * mask + image * (1-mask))
    print(json.dumps(dict(output=str(output.resolve()), iteration=engine.iteration)), flush=True)


def import_tf(args):
    from me_backend.checkpoint_import import import_tf_me_weights
    from me_backend.web_bridge import atomic_json, load_config_options, validate_model_name, utc_now
    destination = Path(args.model).resolve()
    name = validate_model_name(destination.name)
    if destination.exists():
        raise FileExistsError('Import requires a new model directory; existing models are never overwritten')
    config = MEConfig.from_dict(load_config_options(args))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + name + '-import-', dir=destination.parent))
    try:
        report = import_tf_me_weights(args.source, args.name, config, staging / 'me.pt')
        report = {**report, 'path': str(destination / 'me.pt')}
        atomic_json(staging / 'metadata.json', dict(format='me-pytorch', version=1, schemaVersion=1,
            name=name, model_class='ME', modelClass='ME', iteration=0, config=config.to_dict(),
            checkpoint='me.pt', savedAt=utc_now(), importedFrom=report))
        os.rename(staging, destination)
    finally:
        if staging.exists():
            # Only this invocation's new, checked temporary directory is removed.
            for item in staging.iterdir():
                if item.is_file():
                    item.unlink()
            staging.rmdir()
    print(json.dumps(dict(model=str(destination), imported=report), ensure_ascii=False), flush=True)


def backup_command(args):
    from me_backend.checkpoint_backup import list_backups, restore_backup
    if args.command == 'list-backups':
        result = {'model': str(Path(args.model).resolve()), 'backups': list_backups(args.model, verify=True)}
    else:
        result = restore_backup(args.model, args.backup_id)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def config_options(parser, required=False):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--config', help='JSON configuration file; partial updates merge with a resumed checkpoint')
    group.add_argument('--config-json', help='Inline JSON configuration object; partial updates merge with a resumed checkpoint')


def training_options(parser):
    config_options(parser)
    parser.add_argument('--resolution', type=int, default=None, help='New models default to 128; explicit resume changes are checked')
    parser.add_argument('--batch-size', type=int, default=None, help='New models default to 4; explicit resume changes require permission flag')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--preview-every', type=int, default=10)
    parser.add_argument('--backup-every', type=int, default=1000, help='Automatic verified backup interval in iterations')
    parser.add_argument('--backup-keep', type=int, default=3, help='Automatic backup generations to retain (2..32)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reset-data-state', action='store_true')
    parser.add_argument('--allow-config-change', action='store_true', help='Explicitly apply compatible training settings during resume')
    parser.add_argument('--reset-optimizer', action='store_true', help='Explicitly discard optimizer history when changing optimizer type')
    parser.add_argument('--initialize-from', help='Initialize only network weights from a native pretrained me.pt or model directory')
    parser.add_argument('--pretraining-data-dir', help='Dedicated aligned faceset required when pretrain=true')
    parser.add_argument('--target-iterations', type=int, default=0, help='Save and stop at this total iteration; zero disables the limit')
    parser.add_argument('--debug-samples', action='store_true')
    parser.add_argument('--archi', default=None)
    parser.add_argument('--rg', dest='use_rg', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--fp16', dest='use_fp16', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--optimizer-on-cpu', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--pretrain', action=argparse.BooleanOptionalAction, default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    w = sub.add_parser('web-train', help='ME training with local WebUI controls')
    for command in (t, w):
        for name in ('src', 'dst', 'model'):
            command.add_argument('--' + name, required=True)
        command.add_argument('--name', required=command is w)
        command.add_argument('--steps', type=int, default=100 if command is t else 0)
        training_options(command)
        command.set_defaults(func=train)
    i = sub.add_parser('infer')
    for name in ('model', 'input', 'output'):
        i.add_argument('--' + name, required=True)
    i.set_defaults(func=infer)
    legacy = sub.add_parser('import-tf', help='Strictly import local TF network tensors; start a fresh optimizer')
    for name in ('source', 'name', 'model'):
        legacy.add_argument('--' + name, required=True)
    config_options(legacy, required=True)
    legacy.set_defaults(func=import_tf)
    backup_list = sub.add_parser('list-backups', help='List verified ME checkpoint backups')
    backup_list.add_argument('--model', required=True)
    backup_list.set_defaults(func=backup_command)
    backup_restore = sub.add_parser('restore-backup', help='Restore an explicitly selected ME backup after training stops')
    backup_restore.add_argument('--model', required=True)
    backup_restore.add_argument('--backup-id', required=True)
    backup_restore.set_defaults(func=backup_command)
    for command in (t, w, i):
        command.add_argument('--device', default='cuda', help='cpu, cuda, cuda:0, or multiple GPUs cuda:0,1')
    for command in (t, w, i, legacy, backup_list, backup_restore):
        command.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == '__main__':
    main()
