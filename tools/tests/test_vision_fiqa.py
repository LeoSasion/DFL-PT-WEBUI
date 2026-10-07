"""Boundaries and actual pinned student isolation, without acquiring resources."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'webui/python'))
from vision_fiqa import ASSETS, make_quality_scorer, verified_assets


class AssetContract(unittest.TestCase):
    def test_missing_and_modified_assets_fail_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Missing pinned'):
                verified_assets(root)
            name, (_, size) = next(iter(ASSETS.items()))
            (root / name).write_bytes(bytes(size))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verified_assets(root)


class ActualStudent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.assets = verified_assets()
        except (OSError, ValueError) as exc:
            raise unittest.SkipTest(str(exc))

    def test_parent_timm_unchanged_input_validation_and_child_exit(self):
        import timm
        original_module = timm
        original_version = timm.__version__
        with make_quality_scorer() as scorer:
            child = scorer.process
            for invalid in (np.zeros((32, 32, 3), np.float32), np.zeros((32, 32), np.uint8),
                            np.zeros((8, 8, 3), np.uint8)):
                with self.assertRaises(ValueError):
                    scorer(invalid)
            image = np.random.default_rng(1).integers(0, 256, (160, 192, 3), dtype=np.uint8)
            result = scorer(image)
            self.assertTrue(np.isfinite(result['score']))
            self.assertEqual(result['provenance']['weightSha256'], ASSETS['EdgeNeXt_XXS_checkpoint.pt'][0])
            self.assertEqual(result, scorer(image))
        self.assertIsNotNone(child.poll())
        self.assertIs(sys.modules['timm'], original_module)
        self.assertEqual(timm.__version__, original_version)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            scorer(image)

    def test_exact_official_model_and_preprocessing_agree(self):
        """Run unmodified official classes/transform in an independent reference."""
        import cv2
        image = np.random.default_rng(93).integers(0, 256, (129, 177, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'face.png'
            cv2.imwrite(str(image_path), image)
            script = r'''
import importlib.util, sys, torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,str(root/'timm-1.0.19-py3-none-any.whl'))
spec=importlib.util.spec_from_file_location('official',root/'FIQA_model.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
model=module.FIQA_EdgeNeXt_XXS(is_pretrained=False)
model.load_state_dict(torch.load(root/'EdgeNeXt_XXS_checkpoint.pt',map_location='cpu',weights_only=True),strict=True)
model.eval(); torch.set_num_threads(4)
transform=transforms.Compose([transforms.Resize(352),transforms.CenterCrop(352),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
with torch.inference_mode(): print(float(model(transform(Image.open(sys.argv[2]).convert('RGB')).unsqueeze(0)).item()))
'''
            reference = subprocess.run([sys.executable, '-c', script, str(self.assets), str(image_path)],
                                       check=True, capture_output=True, text=True, timeout=90,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with make_quality_scorer() as scorer:
                actual = scorer(image)['provenance']['rawScore']
            self.assertAlmostEqual(actual, float(reference.stdout.strip()), places=6)


if __name__ == '__main__':
    unittest.main()
