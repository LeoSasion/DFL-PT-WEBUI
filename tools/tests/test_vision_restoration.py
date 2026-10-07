import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "python"))
import vision_restoration as restoration


class RestorationContract(unittest.TestCase):
    def test_full_image_preserves_context_and_crops_reflected_window_padding(self):
        import torch
        calls = []
        def network(tensor):
            calls.append(tuple(tensor.shape))
            context = tensor.mean(dim=(2, 3), keepdim=True)
            return torch.nn.functional.interpolate(tensor * .5 + context * .5, scale_factor=4, mode='nearest')
        model = restoration.RestorationModel.__new__(restoration.RestorationModel)
        model.torch, model.model_id, model.device = torch, 'realesrgan-x4plus', torch.device('cpu')
        model.tile_size, model.halo, model.network = None, 16, network
        image = np.random.default_rng(29).integers(0, 256, (37, 71, 3), dtype=np.uint8)
        result = model.restore(image, output_size=(284, 148))
        padded = cv2.copyMakeBorder(image, 0, 3, 0, 1, cv2.BORDER_REFLECT_101).astype(np.float32) / 255
        expected = padded * .5 + padded.mean(axis=(0, 1), keepdims=True) * .5
        expected = np.repeat(np.repeat(expected, 4, axis=0), 4, axis=1)[:148, :284]
        expected = np.uint8(np.rint(np.clip(expected, 0, 1) * 255))
        self.assertEqual(calls, [(1, 3, 40, 72)])
        np.testing.assert_array_equal(result, expected)

    def test_exact_alignment_uses_the_shared_five_point_geometry(self):
        source = restoration.FFHQ_TEMPLATE * 2 + [20, -10]
        transform = restoration.five_point_alignment(source)
        mapped = source @ transform[:, :2].T + transform[:, 2]
        np.testing.assert_allclose(mapped, restoration.FFHQ_TEMPLATE, atol=1e-8)

    def test_degenerate_or_nonfinite_alignment_is_rejected(self):
        for points in [np.zeros((5, 2)), np.full((5, 2), np.nan), np.zeros((68, 2))]:
            with self.assertRaises(ValueError):
                restoration.five_point_alignment(points)

    def test_reference_is_not_ai_generated_and_down4_is_exact_area_sampling(self):
        image = np.random.default_rng(3).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        reference, valid, transform = restoration.aligned_reference(image, restoration.FFHQ_TEMPLATE)
        np.testing.assert_array_equal(reference, image)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(restoration.synthetic_down4(reference), cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA))

    def test_reflected_alignment_padding_is_excluded_from_the_score_mask(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        points = restoration.FFHQ_TEMPLATE + [120, 0]
        reference, valid, _ = restoration.aligned_reference(image, points)
        self.assertEqual(reference.shape, (512, 512, 3))
        self.assertFalse(valid[:, -100:].any())
        self.assertGreater(valid.mean(), 0.5)

    def test_identical_rgb_has_perfect_score_and_json_safe_psnr(self):
        image = np.full((64, 64, 3), 120, dtype=np.uint8)
        metrics = restoration.quality_metrics(image, image, perceptual=lambda *_: 0.0)
        self.assertTrue(metrics["exactMatch"])
        self.assertIsNone(metrics["psnrDb"])
        self.assertAlmostEqual(metrics["ssim"], 1)
        self.assertAlmostEqual(restoration.metric_quality_score(metrics), 100)

    def test_invalid_pixels_cannot_inflate_reconstruction_error(self):
        image = np.full((64, 64, 3), 120, dtype=np.uint8)
        output = image.copy()
        output[:10] = 0
        mask = np.ones((64, 64), dtype=bool)
        mask[:10] = False
        seen = []
        def perceptual(reference, prediction):
            seen.append(prediction)
            return float(np.mean(np.abs(reference.astype(float) - prediction)))
        metrics = restoration.quality_metrics(image, output, mask, perceptual)
        self.assertTrue(metrics["exactMatch"])
        self.assertAlmostEqual(metrics["ssim"], 1)
        self.assertEqual(metrics["lpips"], 0)
        np.testing.assert_array_equal(seen[0], image)

    def test_unequal_scale_empty_masks_and_unknown_lpips_are_not_rankable(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            restoration.quality_metrics(image, image[:32])
        with self.assertRaises(ValueError):
            restoration.quality_metrics(image, image, np.zeros((64, 64), dtype=bool))
        self.assertIsNone(restoration.metric_quality_score(restoration.quality_metrics(image, image)))

    def test_quality_order_cannot_change_with_execution_speed(self):
        metrics = {"exactMatch": False, "psnrDb": 30, "ssim": 0.95, "lpips": 0.05}
        self.assertEqual(restoration.metric_quality_score({**metrics, "latency": 1}),
                         restoration.metric_quality_score({**metrics, "latency": 9999}))

    def test_missing_or_modified_weights_fail_before_loading_neural_networks(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                restoration.verified_asset(temporary, "gfpgan-v1.4")
            (Path(temporary) / "GFPGANv1.4.pth").write_bytes(b"not-the-official-checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                restoration.verified_asset(temporary, "gfpgan-v1.4")

    def test_basic_sr_compatibility_alias_uses_the_official_same_function(self):
        restoration.compatibility_imports()
        import torchvision.transforms.functional as official
        import torchvision.transforms.functional_tensor as compatibility
        self.assertIs(compatibility.rgb_to_grayscale, official.rgb_to_grayscale)

    def test_offline_lpips_load_is_strict_and_makes_no_pretrained_network_request(self):
        assets = ROOT / "workspace" / ".vision-models" / "restoration"
        if not (assets / "alexnet-owt-7be5be79.pth").exists():
            self.skipTest("Optional official LPIPS cache unavailable")
        import torch
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("automatic download prohibited")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("automatic download prohibited")):
            metric = restoration.OfflineLpips(assets, device="cpu")
            image = np.full((64, 64, 3), 120, dtype=np.uint8)
            self.assertAlmostEqual(metric(image, image), 0, places=7)
            self.assertTrue(all(not parameter.requires_grad for parameter in metric.network.parameters()))


if __name__ == "__main__":
    unittest.main()
