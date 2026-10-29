"""Prepare a matching CUDA runtime, then let the user retain or delete the old one."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'_internal/DeepFaceLab'))
from core.torch_runtime_switch import RuntimeSwitch, probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status','prepare','keep-retained','delete-retained','recover','_probe'])
    parser.add_argument('--target', choices=['cu126','cu128'])
    parser.add_argument('--transaction-id')
    parser.add_argument('--confirm-delete', action='store_true')
    parser.add_argument('--prefix')
    parser.add_argument('--output')
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    progress = lambda value: print('DFL_CUDA_EVENT|'+json.dumps(value,ensure_ascii=False),flush=True)
    try:
        if args.command == '_probe':
            if not args.prefix or not args.output or not args.target:
                parser.error('_probe requires --prefix, --output and --target')
            result = probe(ROOT,args.prefix,args.target,args.output,args.require_gpu)
        else:
            manager = RuntimeSwitch(ROOT, progress)
            if args.command == 'status':result=manager.status()
            elif args.command == 'prepare':
                if not args.target:parser.error('prepare requires --target')
                result=manager.prepare(args.target)
            elif args.command == 'recover':result=manager.recover(args.transaction_id)
            elif args.command == 'keep-retained':result=manager.keep_retained(args.transaction_id)
            else:result=manager.delete_retained(args.transaction_id,args.confirm_delete)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0
    except Exception as error:
        print(str(error),file=sys.stderr,flush=True)
        print(json.dumps(dict(status='error',error=str(error),message=str(error),canSwitch=False,canDelete=False),ensure_ascii=False),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
