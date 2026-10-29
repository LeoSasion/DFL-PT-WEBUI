"""Rebuild a quantized DFM engine, or run an existing engine on NHWC BGR NPY."""
import argparse
from pathlib import Path
import numpy as np
from .runtime import build_engine, TensorRTRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('dfm', type=Path)
    build.add_argument('--device', default='cuda:0')
    build.add_argument('--max-batch', type=int, default=4)
    infer = commands.add_parser('infer')
    infer.add_argument('engine', type=Path)
    infer.add_argument('input', type=Path, help='float32 NHWC BGR [0,1] .npy')
    infer.add_argument('output', type=Path, help='output .npz (dst_mask, swapped_bgr, src_mask)')
    infer.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'build':
        if not 1 <= args.max_batch <= 64:
            parser.error('--max-batch must be 1..64')
        result = build_engine(args.dfm.resolve(), args.device, max_batch=args.max_batch)
        print(result[0])
    else:
        values = np.load(args.input, allow_pickle=False)
        if values.ndim != 4 or values.shape[-1] != 3 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            parser.error('input must be finite NHWC BGR in [0,1]')
        outputs = TensorRTRunner(args.engine, args.device)(values)
        np.savez(args.output, dst_mask=outputs[0], swapped_bgr=outputs[1], src_mask=outputs[2])


if __name__ == '__main__':
    main()
