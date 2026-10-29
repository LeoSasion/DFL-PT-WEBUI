"""DFL merger contract checks; no pretrained TAE weights or GPU are loaded."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))

from core.leras import nn
from me_backend.tae_adapter import TAEInferenceModel


class _Engine:
    config = SimpleNamespace(resolution=64, face_type='wf')
    device = SimpleNamespace(type='cpu', index=None)
    iteration = 17
    last_input = None

    def predict(self, value):
        self.last_input = value.copy()
        shape = (len(value), 1, 64, 64)
        return value, np.full(shape, .25, np.float32), np.full(shape, .75, np.float32)


class TAEMergerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.engine = _Engine()
        self.loader = patch.dict(sys.modules, {'tae.engine': SimpleNamespace(
            TAEEngine=SimpleNamespace(load=self._load))})
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.runtime = patch.object(nn, 'initialize')
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.loaded = None

    def _load(self, path, device, weights, training):
        self.assertFalse(training)
        self.loaded = (path, device, weights)
        return self.engine

    def model(self, name='selected'):
        directory = self.root / name
        directory.mkdir()
        (directory / 'tae.pt').touch()
        return directory

    def test_real_model_registration_and_merger_bgr_mask_order(self):
        from models import import_model
        self.assertIs(import_model('TAE'), TAEInferenceModel)
        directory = self.model()
        model = TAEInferenceModel(self.root, cpu_only=True, vae_weights='explicit.safetensors')
        predictor, shape, config = model.get_MergerConfig()
        face = np.empty(shape, np.float32)
        face[:] = [.1, .4, .9]
        swap, swap_mask, dst_mask = predictor(face)
        np.testing.assert_array_equal(swap, face)
        np.testing.assert_array_equal(self.engine.last_input[0, :, 0, 0], face[0, 0])
        self.assertEqual(shape, (64, 64, 3))
        self.assertEqual(swap_mask.shape, (64, 64))
        self.assertEqual(dst_mask.shape, (64, 64))
        self.assertTrue(np.all(swap_mask == .25))
        self.assertTrue(np.all(dst_mask == .75))
        self.assertEqual(model.get_iter(), 17)
        self.assertEqual(model.get_model_name(), 'selected_TAE')
        self.assertEqual(self.loaded, (directory / 'tae.pt', 'cpu', 'explicit.safetensors'))
        self.assertEqual(model.get_strpath_storage_for_file('merger_session.dat'),
                         str(directory / 'merger_session.dat'))

    def test_ambiguous_selection_never_loads_me_checkpoint(self):
        (self.root / 'me.pt').touch()
        with self.assertRaisesRegex(ValueError, 'explicitly'):
            TAEInferenceModel(self.root, cpu_only=True)
        self.model('one')
        self.model('two')
        with self.assertRaisesRegex(ValueError, 'explicitly'):
            TAEInferenceModel(self.root, cpu_only=True)
        selected = TAEInferenceModel(self.root, force_model_name='two', cpu_only=True)
        self.assertEqual(selected.name, 'two')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            TAEInferenceModel(self.root, force_model_name='../one', cpu_only=True)

    def test_bad_pixels_masks_and_training_mode_fail_explicitly(self):
        directory = self.model()
        model = TAEInferenceModel(directory, cpu_only=True)
        predictor = model.get_MergerConfig()[0]
        for value in (np.zeros((32, 32, 3), np.float32), np.zeros((64, 64, 3), np.uint8),
                      np.full((64, 64, 3), np.nan, np.float32),
                      np.full((64, 64, 3), 1.1, np.float32)):
            with self.assertRaisesRegex(ValueError, 'BGR HWC'):
                predictor(value)
        self.engine.predict = lambda value: (value, value[:, :1], value[:, :1, :32])
        with self.assertRaisesRegex(ValueError, 'invalid BGR'):
            predictor(np.zeros((64, 64, 3), np.float32))
        with self.assertRaisesRegex(ValueError, 'training is provided'):
            TAEInferenceModel(directory, is_training=True)
        with self.assertRaisesRegex(ValueError, 'one nonnegative'):
            TAEInferenceModel(directory, force_gpu_idxs='0,1')


if __name__ == '__main__':
    unittest.main()
