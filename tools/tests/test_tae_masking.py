"""CPU mask/loss and checkpoint contracts; no production training job."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal' / 'DeepFaceLab'))
sys.path.insert(0, str(Path(__file__).parent))
from tae.config import TrainConfig
from tae.masking import expansion_geometry, image_loss_mask
from tae.losses import reconstruction_loss
from tae.engine import TAEEngine
from test_tae_engine import TinyFrozenVAE, engine, batch


class MaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_radii_match_resolution_and_euclidean_support(self):
        self.assertEqual([expansion_geometry(r) for r in (256, 448, 512, 640, 832)],
                         [(26, 26), (45, 45), (51, 51), (64, 64), (83, 83)])
        mask = np.zeros((1, 1, 512, 512), np.float32)
        mask[0, 0, 256, 256] = 1
        result = image_loss_mask(mask, 'expanded_feather')[0, 0]
        self.assertEqual(result[256, 256], 1)
        for distance in (1, 13, 25, 26, 38, 50, 51, 52):
            self.assertAlmostEqual(float(result[256, 256+distance]), max(0, 1-distance/51), places=6)
        self.assertTrue(np.all(np.diff(result[256, 256:308]) < 0))
        self.assertGreater(result[256, 300], 0)
        self.assertLess(result[256, 300], 1)
        self.assertGreater(result[256, 306], 0)
        self.assertEqual(result[256, 307], 0)  # Radius 51: zero weight.
        self.assertEqual(result[293, 293], 0)  # Square dilation would include it.
        np.testing.assert_array_equal(result, result.T)
        self.assertEqual(mask.sum(), 1)

    def test_face_full_soft_and_image_edges(self):
        mask = np.zeros((2, 1, 256, 256), np.float32)
        mask[0, 0, :20, :20] = 1
        mask[1, 0, 100:140, 100:140] = 1
        mask[1, 0, 99, 100:140] = .25
        np.testing.assert_array_equal(image_loss_mask(mask), mask)
        self.assertTrue((image_loss_mask(mask, 'full') == 1).all())
        result = image_loss_mask(mask, 'expanded_feather')
        # Even a faint original halo outside the expansion must not bypass the
        # zero-weight outer boundary; the mask-head target remains untouched.
        mask[1, 0, 10, 10] = .1
        result = image_loss_mask(mask, 'expanded_feather')
        self.assertEqual(result[1, 0, 10, 10], 0)
        self.assertEqual(result[0, 0, -1, -1], 0)
        self.assertTrue(((result > 0) & (result < 1)).any())
        self.assertTrue((result[mask == 1] == 1).all())

    def test_only_image_loss_grows_mask_and_eye_targets_stay_native(self):
        original = list(batch())
        original[1].fill(0)
        original[2].fill(0)
        original[2][:, :, 100:140, 100:140] = 1
        worker = engine()
        mask_terms, eyes_terms = [], []
        for mode in ('face', 'expanded_feather', 'full'):
            worker.config = replace(worker.config, training_mask_mode=mode)
            prepared = worker._batch(tuple(original))
            torch.testing.assert_close(prepared['mask'], torch.from_numpy(original[2]))
            prediction = torch.full((1, 3, 256, 256), .4, requires_grad=True)
            _, terms = reconstruction_loss(prediction, torch.full_like(prepared['mask'], .3), prepared, worker.config)
            terms['rgb'].backward()
            near, far = prediction.grad[0, 0, 120, 150], prediction.grad[0, 0, 120, 220]
            self.assertEqual(bool(near > 0), mode != 'face')
            self.assertEqual(bool(far > 0), mode == 'full')
            if mode == 'expanded_feather':
                self.assertGreater(near, prediction.grad[0, 0, 120, 161])
            mask_terms.append(terms['mask'].detach())
            eyes_terms.append(terms['eyes_mouth'].detach())
        for values in (mask_terms, eyes_terms):
            for value in values[1:]:
                torch.testing.assert_close(value, values[0], rtol=0, atol=0)

    def test_old_checkpoint_defaults_and_mode_stage_preserve_weights_and_data(self):
        worker = engine()
        worker.train_step(batch(1), batch(2))  # One tiny CPU fixture update.
        worker.data_state = {'sentinel': 'sampling-state'}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'tae.pt'
            worker.save(file)
            old = torch.load(file, weights_only=True)
            old['config'].pop('training_mask_mode')
            for stage in old['stage_history']:
                stage.pop('training_mask_mode')
            torch.save(old, file)
            restored = TAEEngine.load(file, 'cpu', vae=TinyFrozenVAE())
            self.assertEqual(restored.config.training_mask_mode, 'face')
            self.assertFalse(restored.reconfigure_decoder(restored.config))
            before = {k: v.clone() for k, v in restored.network.trainable_state_dict().items()}
            policy = restored.config.resume_contract()
            for mode in ('expanded_feather', 'full', 'face'):
                self.assertTrue(restored.reconfigure_decoder(replace(restored.config, training_mask_mode=mode)))
                self.assertEqual(restored.iteration, 1)
                self.assertEqual(restored.stage_updates, 0)
                self.assertEqual(restored.data_state, worker.data_state)
                self.assertEqual(restored.config.resume_contract(), policy)
                self.assertFalse(restored.optimizer.state)
                for key, value in before.items():
                    torch.testing.assert_close(restored.network.trainable_state_dict()[key], value, rtol=0, atol=0)
                restored.save(file)
                restored = TAEEngine.load(file, 'cpu', vae=TinyFrozenVAE())
                self.assertEqual(restored.config.training_mask_mode, mode)
                self.assertEqual(restored.stage_history[-1]['training_mask_mode'], mode)
            tampered = torch.load(file, weights_only=True)
            tampered['stage_history'][-1]['training_mask_mode'] = 'full'
            torch.save(tampered, file)
            with self.assertRaisesRegex(ValueError, 'stage/config'):
                TAEEngine.load(file, 'cpu', vae=TinyFrozenVAE())

    def test_invalid_modes_and_masks_are_rejected(self):
        with self.assertRaises(ValueError):
            TrainConfig(training_mask_mode='blur_background').validate()
        with self.assertRaises(ValueError):
            image_loss_mask(np.zeros((1, 1, 256, 256), np.float32), 'expanded_feather')
        with self.assertRaises(ValueError):
            image_loss_mask(np.full((1, 1, 256, 256), np.nan), 'full')


if __name__ == '__main__':
    unittest.main()
