"""Optional isolated ME read-only evaluation smoke with real aligned data."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal' / 'DeepFaceLab'))
from me_backend.engine import MEEngine
from me_backend.web_bridge import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'src', 'dst', 'name', 'manifest', 'evaluation-root', 'model-key'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    checkpoint = Path(args.model) / 'me.pt'
    digest_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    engine = MEEngine.load(checkpoint, args.device)
    iteration_before = engine.iteration
    parameters_before = {key: value.clone() for key, value in engine.network.state_dict().items()}
    os.environ.update(DFL_WEB_EVAL_MANIFEST=args.manifest, DFL_WEB_EVAL_ROOT=args.evaluation_root,
        DFL_WEB_EVAL_MODEL_KEY=args.model_key, DFL_WEBUI_PYTHON=str(ROOT / 'webui' / 'python'))
    summary = evaluate(engine, args.name, args.src, args.dst)
    assert engine.iteration == iteration_before
    assert digest_before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert all((value == parameters_before[key]).all() for key, value in engine.network.state_dict().items())
    print('ME evaluation published ' + summary['snapshotId'] + ' without changing weights or checkpoint')


if __name__ == '__main__':
    main()
