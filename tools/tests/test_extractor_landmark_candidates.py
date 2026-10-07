from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from facelib import FaceType
from facelib.LandmarkCandidates import TufaExtractor, load_source_map, sha256, validate_landmark_assets
from mainscripts import Extractor
from DFLIMG import DFLJPG


class LandmarkProductionTests(unittest.TestCase):
    def test_validated_restored_odd_width_preserves_forward_canvas_and_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'restored.png'
            image = np.zeros((194, 259, 3), np.uint8)
            image[:, -1] = 255
            self.assertTrue(cv2.imwrite(str(path), image))
            mapping = root / 'map.json'
            mapping.write_text(json.dumps({'outputs': [{'name': path.name, 'sourceName': 'original.jpg',
                'outputSha256': sha256(path), 'inputSize': [259, 194], 'outputSize': [259, 194]}]}), encoding='utf-8')
            validated = load_source_map(mapping, root)
            theta = np.linspace(np.pi, 0, 17)
            jaw = np.c_[np.cos(theta), np.sin(theta)] * [60, 60] + [130, 110]
            points68 = np.vstack([jaw, Extractor.LandmarksProcessor.landmarks_2D * 100 + [80, 40]]).astype(np.float32)
            self.assertEqual(points68.shape, (68, 2))
            for restored, expected_width in ((True, 259), (False, 258)):
                output = root / ('mapped' if restored else 'legacy')
                output.mkdir()
                worker = object.__new__(Extractor.ExtractSubprocessor.Cli)
                worker.type = 'all'
                worker.cached_image = (None, None)
                worker.source_map = validated if restored else {}
                worker.landmark_model = 'tufa'
                worker.face_type = FaceType.WHOLE_FACE
                worker.image_size = 256
                worker.jpeg_quality = 95
                worker.max_faces_from_image = 1
                worker.final_output_path = output
                worker.output_debug_path = None
                seen = []
                def rects(image, **kwargs):
                    seen.append(('rects', image.shape, int(image[0, -1, 0])))
                    return [[60, 20, 200, 180]]
                def landmarks(image, *args, **kwargs):
                    seen.append(('landmarks', image.shape, int(image[0, -1, 0])))
                    return [points68.copy()]
                def audit(image, rects):
                    seen.append(('audit', image.shape, int(image[0, -1, 0])))
                    return [{'points_original': np.tile([[100., 100.]], (98, 1)).tolist(),
                        'image_size_wh': [image.shape[1], image.shape[0]],
                        'point_definition': {'name': 'WFLW98', 'count': 98}}]
                worker.rects_extractor = SimpleNamespace(extract=rects)
                worker.tufa_extractor = SimpleNamespace(extract=landmarks, audit=audit, identity={'model': 'mock-native68'})
                worker.landmarks_extractor = worker.tufa_extractor
                result = worker.process_data(Extractor.ExtractSubprocessor.Data(path))
                self.assertEqual(len(result.final_output_files), 1)
                self.assertEqual([stage[1] for stage in seen], [(194, expected_width, 3)] * 3)
                self.assertEqual([stage[2] for stage in seen], [255 if restored else 0] * 3)
                dfl = DFLJPG.load(result.final_output_files[0])
                self.assertEqual(dfl.get_dict()['landmark_provenance']['processedCanvasWH'], [expected_width, 194])
                native = dfl.get_dict()['native_landmarks98']
                self.assertEqual(native['image_size_wh'], [expected_width, 194])
                np.testing.assert_array_equal(dfl.get_source_landmarks(), points68)
                np.testing.assert_allclose(Extractor.LandmarksProcessor.transform_points(points68, dfl.get_image_to_face_mat()), dfl.get_landmarks(), atol=1e-4)
                np.testing.assert_allclose(Extractor.LandmarksProcessor.transform_points(native['points_original'], dfl.get_image_to_face_mat()), native['points_aligned'], atol=1e-4)
                self.assertEqual(dfl.get_source_filename(), 'original.jpg' if restored else 'restored.png')

    def test_missing_tufa_assets_precedes_output_creation_or_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'input'
            source.mkdir()
            output = Path(temp) / 'aligned'
            for exists in (False, True):
                if exists:
                    output.mkdir()
                    (output / 'keep.jpg').write_bytes(b'preserve')
                with patch('facelib.LandmarkCandidates.validate_landmark_assets', side_effect=ValueError('missing TUFA asset')):
                    with self.assertRaisesRegex(ValueError, 'missing TUFA'):
                        Extractor.main(detector='s3fd', input_path=source, output_path=output, landmark_model='tufa')
                if exists:
                    self.assertEqual((output / 'keep.jpg').read_bytes(), b'preserve')
                else:
                    self.assertFalse(output.exists())

    def test_native68_adapter_preserves_color_and_never_truncates98(self):
        adapter = object.__new__(TufaExtractor)
        calls = []
        def predict(rgb, rect):
            calls.append(rgb[0, 0].tolist())
            return {'points_original': np.ones((68, 2)).tolist()}
        adapter.predictor = SimpleNamespace(predict=predict)
        image = np.full((4, 4, 3), [1, 2, 3], np.uint8)
        self.assertEqual(adapter.extract(image, [[0, 0, 4, 4]])[0].shape, (68, 2))
        self.assertEqual(calls, [[3, 2, 1]])
        adapter.predictor = SimpleNamespace(predict=lambda *args: {'points_original': np.ones((98, 2)).tolist()})
        with self.assertRaisesRegex(ValueError, 'Native TUFA68'):
            adapter.extract(image, [[0, 0, 4, 4]])

    def test_head_retains_fan3d_while_tufa_nonhead_uses_native68(self):
        for face_type in (FaceType.HEAD, FaceType.WHOLE_FACE):
            processor = Extractor.ExtractSubprocessor([], 'all', 256, 95, face_type, None,
                landmark_model='tufa', device_config=Extractor.nn.DeviceConfig.CPU())
            with patch.object(sys, 'stdin', SimpleNamespace(fileno=lambda: 0)):
                _, _, config = next(processor.process_info_generator())
            worker = object.__new__(Extractor.ExtractSubprocessor.Cli)
            worker.log_info = lambda *args: None
            with patch.object(Extractor.nn, 'initialize'), patch.object(Extractor.facelib, 'S3FDExtractor'), \
                 patch.object(Extractor.facelib, 'FANExtractor') as fan, \
                 patch('facelib.LandmarkCandidates.TufaExtractor') as tufa:
                worker.on_initialize(config)
                tufa.assert_called_once_with(device='cpu')
                if face_type == FaceType.HEAD:
                    fan.assert_called_once_with(landmarks_3D=True, place_model_on_cpu=True)
                else:
                    fan.assert_not_called()
                    self.assertIs(worker.landmarks_extractor, tufa.return_value)

    def test_audit_uses_original_canvas_and_canonical_rect_without_editing_it(self):
        adapter = object.__new__(TufaExtractor)
        seen = []
        adapter.audit_predictor = SimpleNamespace(predict=lambda image, rect: seen.append(rect) or {})
        rects = [(40, 35, 10, 5)]
        adapter.audit(np.zeros((50, 50, 3), np.uint8), rects)
        self.assertEqual(seen, [[10, 5, 40, 35]])
        self.assertEqual(rects, [(40, 35, 10, 5)])

    def test_final_metadata_and_hash_bound_sidecar_retain_distinct_topologies(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            data = Extractor.ExtractSubprocessor.Data(folder / 'restored.png', rects=[[0, 0, 256, 256]],
                landmarks=[np.full((68, 2), 100, dtype=np.float32)], manual=True)
            data.landmark_provenance = {'alignmentModel': 'tufa68', 'depthAvailable': False}
            data.restoration_provenance = {'sourceFilename': 'original.jpg'}
            points = np.arange(196, dtype=float).reshape(98, 2)
            data.native_landmarks98 = [{'points_original': points.tolist(), 'point_definition': {'name': 'WFLW98', 'count': 98}}]
            mat = np.array([[.5, 0, 10], [0, .5, 20]], dtype=np.float32)
            with patch.object(Extractor.LandmarksProcessor, 'get_transform_mat', return_value=mat):
                Extractor.ExtractSubprocessor.Cli.final_stage(data, np.zeros((256, 256, 3), np.uint8),
                    FaceType.WHOLE_FACE, 256, 95, final_output_path=folder)
            output = data.final_output_files[0]
            saved = DFLJPG.load(output)
            self.assertEqual(saved.get_landmarks().shape, (68, 2))
            self.assertEqual(saved.get_source_filename(), 'original.jpg')
            native = saved.get_dict()['native_landmarks98']
            np.testing.assert_allclose(native['points_aligned'], points * .5 + [10, 20])
            self.assertEqual(len(native['points_original']), 98)
            record = json.loads(Path(str(output) + '.landmarks.json').read_text(encoding='utf-8'))
            self.assertEqual(record['aligned_sha256'], sha256(output))
            self.assertEqual(record['native_landmarks98'], native)

    def test_source_map_rejects_paths_hash_mismatch_duplicate_and_resized_canvas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.png').write_bytes(b'input')
            mapping = root / 'map.json'
            entry = {'name': 'a.png', 'sourceName': 'a.jpg', 'inputSha256': 'a' * 64}
            for changed in ({'name': '../a.png'}, {'sourceName': 'C:\\a.jpg'}, {'outputSha256': 'b' * 64},
                            {'inputSize': [256, 256], 'outputSize': [512, 512]}, {'inputSha256': 'x' * 64}):
                mapping.write_text(json.dumps({'outputs': [{**entry, **changed}]}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_source_map(mapping, root)
            mapping.write_text(json.dumps({'outputs': [entry, entry]}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_source_map(mapping, root)
            mapping.write_text(json.dumps({'outputs': [entry]}), encoding='utf-8')
            saved = load_source_map(mapping, root)['a.png']
            self.assertFalse(saved['originalHashVerified'])
            self.assertEqual(saved['declaredInputSha256'], 'a' * 64)
            self.assertEqual(saved['processedSha256'], sha256(root / 'a.png'))

    def test_fan_hash_mutation_and_unknown_model_rejected(self):
        with patch('facelib.LandmarkCandidates.sha256', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'pinned release identity'):
                validate_landmark_assets('fan', 'head')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            validate_landmark_assets('unknown', 'head')

    def test_incomplete_tufa_worker_result_fails_instead_of_claiming_success(self):
        processor = Extractor.ExtractSubprocessor([Extractor.ExtractSubprocessor.Data(Path('pending.jpg'))],
            'all', 256, 95, FaceType.WHOLE_FACE, None, landmark_model='tufa', device_config=Extractor.nn.DeviceConfig.CPU())
        with self.assertRaisesRegex(RuntimeError, 'No fallback'):
            processor.get_result()


if __name__ == '__main__':
    unittest.main()
