"""Project-local entry for rebuilding and running an extreme DFM TensorRT engine."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_internal' / 'DeepFaceLab'))
from dfm_export.__main__ import main

if __name__ == '__main__':
    main()
