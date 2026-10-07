"""Original-PTS and tail preservation requirements, independent of candidate scores."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('scene_detection_tested', ROOT / 'webui/python/scene_detection.py')
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)


def timeline(points, final_duration=31):
    return {'source': {'name': 'fixture.mkv', 'bytes': 42, 'sha256': 'a' * 64},
            'video': {}, 'frames': [{'sourceFrameIndex': index, 'pts': pts, 'timeBase': [1, 90000],
                                   'durationPts': final_duration if index == len(points) - 1 else None}
                                  for index, pts in enumerate(points)]}


class OriginalTimingTests(unittest.TestCase):
    def test_vfr_offset_and_integer_terminal_duration_are_retained(self):
        result = scene.assemble(timeline([180000, 180017, 180051, 180052]),
            [{'sourceFrameIndex': 2, 'score': .8}], 'fixture', {})
        self.assertEqual(result['cuts'][0]['pts'], 180051)
        self.assertEqual(result['scenes'][0]['end'], float(Fraction(51, 90000)))
        self.assertEqual(result['scenes'][1]['startPts'], 180051)
        self.assertEqual(result['scenes'][1]['endPts'], 180083)
        self.assertEqual(result['scenes'][1]['endFrameIndexExclusive'], 4)
        self.assertNotIn('frames', result)

    def test_ui_cap_retains_full_cut_inventory_and_final_source_tail(self):
        result = scene.assemble(timeline(list(range(700))),
            [{'sourceFrameIndex': i} for i in range(1, 700)], 'fixture', {})
        self.assertEqual(len(result['scenes']), 500)
        self.assertEqual(len(result['cuts']), 699)
        self.assertEqual(result['total'], 700)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['scenes'][-1]['startFrameIndex'], 499)
        self.assertEqual(result['scenes'][-1]['endFrameIndexExclusive'], 700)
        self.assertEqual(result['scenes'][-1]['endPts'], 730)

    def test_missing_tail_duration_is_not_guessed_from_previous_delta(self):
        with self.assertRaises(ValueError):
            scene.assemble(timeline([0, 13, 39], None), [], 'fixture', {})

    def test_fractional_or_out_of_range_candidate_index_is_rejected(self):
        for index in (-1, 2, 1.4):
            with self.subTest(index=index), self.assertRaises(ValueError):
                scene.assemble(timeline([0, 13]), [{'sourceFrameIndex': index}], 'fixture', {})

    def test_media_change_during_probe_or_inference_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'fixture.mp4'
            source.write_bytes(b'original')
            value = timeline([0, 3])
            value['source'].update(bytes=8, sha256=scene.file_sha256(source))
            def probe(*args):
                source.write_bytes(b'modified')
                value['source']['sha256'] = scene.file_sha256(source)
                return value
            with patch.object(scene, 'probe_timeline', probe), self.assertRaisesRegex(RuntimeError, 'during timeline'):
                scene.detect(source, 'ffmpeg', 'ffprobe')
            source.write_bytes(b'original')
            value['source']['sha256'] = scene.file_sha256(source)
            def infer(*args):
                source.write_bytes(b'changed!')
                return [], {}
            with patch.object(scene, 'probe_timeline', return_value=value), patch.object(scene, 'ffmpeg_cuts', infer), self.assertRaisesRegex(RuntimeError, 'during scene'):
                scene.detect(source, 'ffmpeg', 'ffprobe')


if __name__ == '__main__':
    unittest.main()
