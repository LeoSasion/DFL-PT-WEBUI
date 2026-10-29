"""Standard-library entry point usable by the base Python before Torch is installed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.gpu_runtime import main, validate_runtime

if __name__ == '__main__':
    main(ROOT)
