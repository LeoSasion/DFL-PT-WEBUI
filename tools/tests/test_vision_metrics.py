import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "python"))
from vision_metrics import METRIC_IDS, PerceptualMetricSuite, metric_asset, prepare_metric_pair


def benchmark_module():
    spec = importlib.util.spec_from_file_location("metric_benchmark", ROOT / "tools" / "vision-metric-benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetricContract(unittest.TestCase):
    def test_same_scale_rgb_and_mask_are_required(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        for output in [image[:32], image.astype(np.float32), image[:, :, 0]]:
            with self.assertRaises(ValueError):
                prepare_metric_pair(image, output)
        with self.assertRaises(ValueError):
            prepare_metric_pair(image, image, np.zeros((64, 64), dtype=bool))

    def test_mask_normalization_keeps_original_pixels_and_does_not_mutate_input(self):
        reference = np.full((64, 64, 3), 100, dtype=np.uint8)
        prediction = np.full_like(reference, 120)
        valid = np.ones((64, 64), dtype=bool)
        valid[:16] = False
        _, normalized = prepare_metric_pair(reference, prediction, valid)
        np.testing.assert_array_equal(normalized[:16], reference[:16])
        np.testing.assert_array_equal(normalized[16:], prediction[16:])
        self.assertTrue((prediction == 120).all())

    def test_unverified_calibration_and_source_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                metric_asset(temporary, "dists-source")
            (Path(temporary) / "DISTS_pt.py").write_text("raise RuntimeError('must not execute')")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                metric_asset(temporary, "dists-source")

    def test_missing_human_preferences_never_become_correlation_or_quality_ranking(self):
        status = benchmark_module().evaluation_status()
        for field in ["humanPreferenceCorrelation", "metricQualityRanking", "restorationQualityRanking", "metricWinner"]:
            self.assertIsNone(status[field])

    def test_missing_or_duplicate_pairs_cannot_claim_full_candidate_coverage(self):
        module = benchmark_module()
        cases = [{"case": str(index)} for index in range(6)]
        protocol = {"caseCount": 6, "cases": cases, "models": list(module.RESTORERS)}
        restoration = {"models": [{"model": model, "cases": list(cases)} for model in module.RESTORERS]}
        module.validate_pair_manifest(protocol, restoration)
        restoration["models"][0]["cases"] = cases[:-1] + [cases[0]]
        with self.assertRaisesRegex(ValueError, "each locked case exactly once"):
            module.validate_pair_manifest(protocol, restoration)
        restoration["models"][0]["cases"] = cases
        restoration["models"][0]["model"] = module.RESTORERS[1]
        with self.assertRaisesRegex(ValueError, "three expected restorers"):
            module.validate_pair_manifest(protocol, restoration)


class RealOfficialMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = ROOT / "workspace" / ".vision-models" / "metrics"
        if not (assets / "vgg16-397923af.pth").exists():
            raise unittest.SkipTest("Optional official perceptual metric cache unavailable")
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("automatic pretrained download prohibited")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("automatic network request prohibited")):
            cls.suite = PerceptualMetricSuite(assets, device="cpu")

    def test_official_vgg_convolution_parameters_are_physically_shared(self):
        self.assertIs(self.suite.vgg.net.slice1[0].weight, self.suite.dists.stage1[0].weight)
        self.assertIs(self.suite.vgg.net.slice5._modules["24"].weight, self.suite.dists.stage5._modules["24"].weight)

    def test_all_calibration_and_feature_parameters_are_frozen(self):
        for network in [self.suite.alex.network, self.suite.vgg, self.suite.dists]:
            self.assertTrue(all(not parameter.requires_grad for parameter in network.parameters()))
            self.assertFalse(network.training)

    def test_identical_and_changed_pairs_are_actually_scored_by_each_network(self):
        reference = np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        identical = self.suite.evaluate(reference, reference)
        changed = self.suite.evaluate(reference, np.roll(reference, 5, axis=1))
        self.assertEqual(set(identical), set(METRIC_IDS))
        for key in METRIC_IDS:
            self.assertAlmostEqual(identical[key], 0, places=5)
            self.assertTrue(np.isfinite(changed[key]))
            self.assertGreater(changed[key], identical[key])

    def test_repeated_frozen_inference_has_no_random_calibration_leftovers(self):
        reference = np.full((64, 64, 3), 100, dtype=np.uint8)
        prediction = np.full_like(reference, 130)
        first, second = self.suite.evaluate(reference, prediction), self.suite.evaluate(reference, prediction)
        self.assertEqual(first, second)
        self.assertIsNone(self.suite.provenance["metricQualityRanking"])


if __name__ == "__main__":
    unittest.main()
