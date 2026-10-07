from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_internal/DeepFaceLab"))
from facelib import FaceType
from facelib.DetectorCandidates import CandidateDetector, validate_candidate_assets
from mainscripts import Extractor


class DetectorCandidateExtractionTests(unittest.TestCase):
    def fixed_entry(self, model_id="yolo11m-face"):
        catalog = json.loads((ROOT / "tools/vision-model-candidates.json").read_text(encoding="utf-8"))
        return next(asset.copy() for asset in catalog["detectorAssets"] if asset["id"] == model_id)

    def test_rect_adapter_sorts_by_area_and_preserves_color_flag(self):
        adapter = object.__new__(CandidateDetector)
        seen = []
        def extract(image, is_bgr):
            seen.append(is_bgr)
            return [(0., 0., 20., 20.), (0., 0., 80., 80.), (90., 0., 120., 30.)]
        adapter.detector = SimpleNamespace(extract=extract)
        image = np.zeros((128, 128, 3), np.uint8)
        self.assertEqual(adapter.extract(image, is_bgr=False),
                         [(0., 0., 80., 80.), (90., 0., 120., 30.), (0., 0., 20., 20.)])
        self.assertEqual(seen, [False])

    def test_overlap_option_remains_supported(self):
        adapter = object.__new__(CandidateDetector)
        adapter.detector = SimpleNamespace(extract=lambda *args, **kwargs:
                    [(0, 0, 80, 80), (10, 10, 30, 30), (90, 0, 120, 30)])
        self.assertEqual(adapter.extract(np.zeros((128, 128, 3), np.uint8), is_remove_intersects=True),
                         [(0, 0, 80, 80), (90, 0, 120, 30), (10, 10, 30, 30)])
        adapter.detector = SimpleNamespace(extract=lambda *args, **kwargs:
                    [(0, 0, 80, 80), (10, 10, 30, 30)])
        self.assertEqual(adapter.extract(np.zeros((128, 128, 3), np.uint8), is_remove_intersects=True),
                         [(0, 0, 80, 80)])

    def test_candidate_preflight_failure_precedes_output_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input"
            source.mkdir()
            output = root / "not-created"
            with patch("facelib.DetectorCandidates.validate_candidate_assets", side_effect=ValueError("missing pinned model")):
                with self.assertRaisesRegex(ValueError, "missing pinned model"):
                    Extractor.main(detector="yolo12l-face", input_path=source, output_path=output)
            self.assertFalse(output.exists())

    def test_missing_asset_explicit_failure_without_instantiation(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Optional detector yolo26s-face unavailable"):
                validate_candidate_assets("yolo26s-face", assets_root=temp)

    def test_runtime_version_mismatch_rejected(self):
        entry = self.fixed_entry()
        asset_module = SimpleNamespace(DEFAULT_ASSETS_ROOT=Path("."), verified_asset=lambda *args, **kwargs:
                (Path("model.pt"), entry))
        with patch("facelib.DetectorCandidates._local_module", return_value=asset_module), \
             patch("facelib.DetectorCandidates.importlib.metadata.version", return_value="different"):
            with self.assertRaisesRegex(ValueError, "runtime differs"):
                validate_candidate_assets("yolo11m-face")

    def test_mutated_task_and_weight_identity_rejected_before_runtime(self):
        for field, value in (("task", "pose"), ("sha256", "0" * 64), ("scale", "s")):
            entry = self.fixed_entry()
            entry[field] = value
            asset_module = SimpleNamespace(DEFAULT_ASSETS_ROOT=Path("."), verified_asset=lambda *args, **kwargs:
                    (Path("model.pt"), entry))
            with patch("facelib.DetectorCandidates._local_module", return_value=asset_module), \
                 patch("facelib.DetectorCandidates.importlib.metadata.version") as version:
                with self.assertRaisesRegex(ValueError, "differs from fixed catalog"):
                    validate_candidate_assets("yolo11m-face")
            version.assert_not_called()

    def test_broken_torchvision_fails_in_preflight(self):
        entry = self.fixed_entry()
        asset_module = SimpleNamespace(DEFAULT_ASSETS_ROOT=Path("."), verified_asset=lambda *args, **kwargs:
                (Path("model.pt"), entry))
        with patch("facelib.DetectorCandidates._local_module", return_value=asset_module), \
             patch("facelib.DetectorCandidates.importlib.metadata.version", return_value="8.4.142"), \
             patch("facelib.DetectorCandidates.importlib.import_module", side_effect=RuntimeError("broken native ABI")):
            with self.assertRaisesRegex(ValueError, "broken native ABI"):
                validate_candidate_assets("yolo11m-face")

    def test_selected_detector_reaches_worker_and_preserves_fan3d(self):
        config = Extractor.nn.DeviceConfig.CPU()
        processor = Extractor.ExtractSubprocessor([], "all", 256, 95, FaceType.HEAD, None,
                        device_config=config, detector="yolo12l-face")
        with patch.object(sys, "stdin", SimpleNamespace(fileno=lambda: 0)):
            _, _, client_dict = next(processor.process_info_generator())
        self.assertEqual(client_dict["detector"], "yolo12l-face")
        worker = object.__new__(Extractor.ExtractSubprocessor.Cli)
        worker.log_info = lambda *args: None
        with patch.object(Extractor.nn, "initialize"), \
             patch("facelib.DetectorCandidates.CandidateDetector") as candidate, \
             patch.object(Extractor.facelib, "FANExtractor") as fan, \
             patch.object(Extractor.facelib, "S3FDExtractor") as s3fd:
            worker.on_initialize(client_dict)
        candidate.assert_called_once_with("yolo12l-face", device="cpu")
        fan.assert_called_once_with(landmarks_3D=True, place_model_on_cpu=True)
        s3fd.assert_not_called()

    def test_default_worker_still_uses_s3fd_and_native_68(self):
        config = Extractor.nn.DeviceConfig.CPU()
        processor = Extractor.ExtractSubprocessor([], "all", 256, 95, FaceType.WHOLE_FACE, None,
                        device_config=config)
        with patch.object(sys, "stdin", SimpleNamespace(fileno=lambda: 0)):
            _, _, client_dict = next(processor.process_info_generator())
        worker = object.__new__(Extractor.ExtractSubprocessor.Cli)
        worker.log_info = lambda *args: None
        with patch.object(Extractor.nn, "initialize"), \
             patch("facelib.DetectorCandidates.CandidateDetector") as candidate, \
             patch.object(Extractor.facelib, "FANExtractor") as fan, \
             patch.object(Extractor.facelib, "S3FDExtractor") as s3fd:
            worker.on_initialize(client_dict)
        s3fd.assert_called_once_with(place_model_on_cpu=True)
        fan.assert_called_once_with(landmarks_3D=False, place_model_on_cpu=True)
        candidate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
