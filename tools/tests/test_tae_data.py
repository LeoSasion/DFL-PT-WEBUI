"""Actual DFLJPEG and PAK/ZIP compatibility with artificial test images only."""

import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_internal" / "DeepFaceLab"))

from DFLIMG import DFLJPG
from tae.config import TrainConfig
from tae.data import TAEAlignedDataset, build_pair, read_aligned
from tae.data import _pretraining_holdout, _pretraining_group
from functools import partial
from tests.me_fixtures import make_aligned as _make_aligned
make_aligned = partial(_make_aligned, size=256)
from tests.test_me_data_features import pack_fixture


def config(**changes):
    return replace(TrainConfig(resolution=256, face_type="f", inter_dims=8,
                               hidden_dims=8, batch_size=2, random_warp=False,
                               random_flip=False), **changes).validate()


def assert_batch_equal(case, actual, expected):
    case.assertEqual(len(actual), 4)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


class DataTests(unittest.TestCase):
    def test_lightweight_checkpoint_only_copies_exposures_at_materialization(self):
        class ObservedCounters(list):
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / 'src', count=1)
            destination = make_aligned(Path(directory) / 'dst', count=1, offset=49)
            with build_pair(source, destination, config(batch_size=3), 19) as pair:
                before = pair.state_dict()
                for dataset in (pair.source_data, pair.destination_data):
                    dataset.exposures = ObservedCounters(dataset.exposures)
                with mock.patch.object(pair, 'state_dict', side_effect=AssertionError('eager pair snapshot')):
                    snapshot = pair.checkpoint_snapshot()
                    expected = pair.batch()
                self.assertEqual(pair.source_data.exposures, [3])
                self.assertEqual(pair.destination_data.exposures, [3])
                self.assertEqual(pair.source_data.exposures.iterations, 0)
                self.assertEqual(pair.destination_data.exposures.iterations, 0)
                self.assertEqual(snapshot.materialize(), before)
                self.assertEqual(pair.source_data.exposures.iterations, 1)
                self.assertEqual(pair.destination_data.exposures.iterations, 1)
                pair.close()
                self.assertEqual(snapshot.materialize(), before)
            with build_pair(source, destination, config(batch_size=3), 99) as resumed:
                resumed.load_state_dict(before)
                for actual, wanted in zip(resumed.batch(), expected):
                    assert_batch_equal(self, actual, wanted)

    def test_failed_pair_uses_delta_and_restores_the_previous_snapshot_window(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / 'src', count=2)
            destination = make_aligned(Path(directory) / 'dst', count=2, offset=49)
            with build_pair(source, destination, config(), 19) as pair:
                initial = pair.state_dict()
                initial_snapshot = pair.checkpoint_snapshot()
                pair.batch()
                after_first = pair.state_dict()
                snapshot = pair.checkpoint_snapshot()
                with mock.patch.object(pair, 'state_dict', side_effect=AssertionError('full counter snapshot')), \
                     mock.patch.object(pair.destination_data, 'batch', side_effect=ValueError('synthetic pair failure')):
                    with self.assertRaisesRegex(ValueError, 'synthetic pair failure'):
                        pair.batch()
                self.assertTrue(pair.source_data.closed and pair.destination_data.closed)
                self.assertEqual(pair.state_dict(), after_first)
                self.assertEqual(snapshot.materialize(), after_first)
                self.assertEqual(initial_snapshot.materialize(), initial)

    def test_checkpoint_bookmark_rejects_expiry_and_sampler_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / 'src', count=2)
            with TAEAlignedDataset(folder, config(), True, 19) as dataset:
                state = dataset.state_dict()
                expired = dataset.checkpoint_snapshot()
                dataset.batch()
                dataset.batch()
                with self.assertRaisesRegex(RuntimeError, 'snapshot expired'):
                    expired.materialize()
                reloaded = dataset.checkpoint_snapshot()
                dataset.load_state_dict(state)
                with self.assertRaisesRegex(RuntimeError, 'sampler restore'):
                    reloaded.materialize()
                self.assertEqual(dataset.state_dict(), state)

    def test_partial_exposure_commit_interrupt_restores_rng_and_duplicate_counts(self):
        class InterruptedCounters(list):
            writes = 0

            def __setitem__(self, key, value):
                self.writes += 1
                if self.writes == 2:
                    raise KeyboardInterrupt('synthetic interrupted counter commitment')
                super().__setitem__(key, value)

        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / 'src', count=1)
            destination = make_aligned(Path(directory) / 'dst', count=1, offset=49)
            with build_pair(source, destination, config(batch_size=3), 19) as pair:
                before = pair.state_dict()
                snapshot = pair.checkpoint_snapshot()
                pair.source_data.exposures = InterruptedCounters(pair.source_data.exposures)
                with self.assertRaisesRegex(KeyboardInterrupt, 'synthetic interrupted'):
                    pair.batch()
                self.assertEqual(pair.state_dict(), before)
                self.assertEqual(snapshot.materialize(), before)

    def test_pretraining_shared_pool_has_disjoint_holdout_and_independent_resumable_samplers(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / 'pool', count=6)
            held = next(f'group-{n}' for n in range(1000) if _pretraining_holdout('source-id:' + f'group-{n}'))
            trained = next(f'group-{n}' for n in range(1000) if not _pretraining_holdout('source-id:' + f'group-{n}'))
            paths = sorted(folder.glob('*.jpg'))
            for index, path in enumerate(paths):
                dfl = DFLJPG.load(str(path))
                dfl.get_dict()['source_id'] = held if index == 0 else trained
                dfl.save()
            shutil.copyfile(paths[4], paths[5])
            options = config(random_warp=True, random_flip=True)
            with build_pair(folder, folder, options, 13, pretraining=True) as pair:
                data = pair.source_data
                self.assertEqual(data.pretraining_receipt['trainingFaces'], 4)
                self.assertEqual(data.pretraining_receipt['validationFaces'], 1)
                self.assertEqual(data.pretraining_receipt['duplicateFiles'], 1)
                self.assertNotIn(data.validation_references[0], data._references)
                self.assertIsNot(data.exposures, pair.destination_data.exposures)
                self.assertIsNot(data.rng, pair.destination_data.rng)
                pair.batch()
                saved = copy.deepcopy(pair.state_dict())
                expected = pair.batch()
                expected_state = pair.state_dict()
            with build_pair(folder, folder, options, 99, pretraining=True) as pair:
                pair.load_state_dict(saved)
                actual = pair.batch()
                self.assertEqual(pair.state_dict(), expected_state)
                for left, right in zip(actual, expected):
                    assert_batch_equal(self, left, right)
            with self.assertRaisesRegex(ValueError, 'same SRC'):
                build_pair(folder, Path(directory) / 'different', options, 0, pretraining=True)

    def test_pretraining_groups_adjacent_named_frames_together(self):
        from types import SimpleNamespace
        reference = SimpleNamespace(packed_sample=None, path='fallback.jpg')
        groups = [_pretraining_group(reference, {'source_filename': name}) for name in
                  ('river-dogfight_003600.png', 'river-dogfight_003632.png')]
        self.assertEqual(groups[0], groups[1])

    def test_large_archive_opt_in_keeps_metadata_validation_and_default_limit(self):
        import importlib
        module = importlib.import_module('samplelib.PackedFaceset')
        from core.safe_pickle import loads, DataFormatError
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / 'pool', count=2)
            archive = pack_fixture(folder, 'pak')
            with mock.patch.object(module, 'MAX_CONFIG_BYTES', 8):
                with self.assertRaisesRegex(ValueError, 'metadata size'):
                    module.PackedFaceset.load(folder, archive_path=archive)
                loaded = module.PackedFaceset.load(folder, archive_path=archive, allow_large=True)
                self.assertEqual(len(loaded), 2)
        for profile in ('faceset', 'faceset-large'):
            with self.assertRaisesRegex(DataFormatError, 'Forbidden pickle global'):
                loads(b'cos\nsystem\n.', profile=profile)

    def test_native_aligned_dfl_masks_bgr_and_no_input_warp(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / "aligned", count=1)
            with TAEAlignedDataset(folder, config(batch_size=1), True, 17) as dataset:
                actual = dataset.batch()
                image, mask, encoded = read_aligned(dataset._references[0], dataset.config)
                assert_batch_equal(self, actual, tuple(value[None].transpose(0, 3, 1, 2)
                                   for value in (image, image, mask, encoded)))
                self.assertTrue(all(value.dtype == np.float32 for value in actual))
                self.assertGreater(actual[3].max(), 2.)
                self.assertEqual(dataset.exposures, [1])

    def test_input_degradation_and_warp_do_not_modify_clean_target_or_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / "aligned", count=2)
            with TAEAlignedDataset(folder, config(), True, 23) as clean:
                reference = clean.batch()
            for changes in ({"half_resolution_probability": 1.0}, {"random_warp": True}):
                with self.subTest(changes=changes):
                    with TAEAlignedDataset(folder, config(**changes), True, 23) as augmented:
                        actual = augmented.batch()
                    self.assertFalse(np.array_equal(reference[0], actual[0]))
                    for left, right in zip(reference[1:], actual[1:]):
                        np.testing.assert_array_equal(left, right)
                    self.assertGreaterEqual(actual[0].min(), 0)
                    self.assertLessEqual(actual[0].max(), 1)

    def test_horizontal_flip_applies_to_target_full_and_priority_masks_together(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / "aligned", count=1)
            with TAEAlignedDataset(folder, config(batch_size=1), True, 0) as clean:
                expected = clean.batch()
            found_flip = False
            for seed in range(12):
                with TAEAlignedDataset(folder, config(batch_size=1, random_flip=True), True, seed) as flipped:
                    actual = flipped.batch()
                if np.array_equal(actual[1], expected[1][..., ::-1]):
                    found_flip = True
                    for value, canonical in zip(actual, expected):
                        np.testing.assert_array_equal(value, canonical[..., ::-1])
                    break
                assert_batch_equal(self, actual, expected)
            self.assertTrue(found_flip, "Fixture seeds did not exercise the flip branch")

    def test_preflight_refuses_upscaled_target_or_wrong_face_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / "aligned", count=1)
            with self.assertRaisesRegex(ValueError, "below resolution|upsampl"):
                TAEAlignedDataset(folder, config(resolution=448), True, 0)
            with self.assertRaisesRegex(ValueError, "face type"):
                TAEAlignedDataset(folder, config(face_type="wf"), True, 0)

    def test_src_dst_duplicate_content_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / "src", count=2)
            destination = make_aligned(Path(directory) / "dst", count=2)
            with self.assertRaisesRegex(ValueError, "identical"):
                build_pair(source, destination, config(), 0)

    def test_resume_preserves_next_pair_rng_and_exposure_counts_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / "src", count=4)
            destination = make_aligned(Path(directory) / "dst", count=4, offset=49)
            options = config(random_warp=True, random_flip=True, half_resolution_probability=.5)
            with build_pair(source, destination, options, 9) as original:
                original.batch()
                state = copy.deepcopy(original.state_dict())
                expected = original.batch()
                expected_state = original.state_dict()
            with build_pair(source, destination, options, 99) as resumed:
                resumed.load_state_dict(state)
                actual = resumed.batch()
                self.assertEqual(resumed.state_dict(), expected_state)
                for left, right in zip(actual, expected):
                    assert_batch_equal(self, left, right)

    def test_rct_and_larger_batch_resume_legacy_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / "src", count=4)
            destination = make_aligned(Path(directory) / "dst", count=4, offset=49)
            original_config = config(batch_size=2)
            with build_pair(source, destination, original_config, 9) as original:
                original.batch()
                saved = copy.deepcopy(original.state_dict())
            changed = config(batch_size=3, ct_mode='rct')
            with build_pair(source, destination, changed, 99, previous_config=original_config) as resumed:
                # Simulate a checkpoint written before ct_mode existed.
                for side, dataset in (('src', resumed.source_data), ('dst', resumed.destination_data)):
                    saved[side]['fingerprint'] = dataset._legacy_fingerprint
                resumed.load_state_dict(saved)
                before = (list(resumed.source_data.exposures), list(resumed.destination_data.exposures))
                src, dst = resumed.batch()
                self.assertEqual(src[0].shape, (3, 3, 256, 256))
                self.assertEqual(dst[0].shape, (3, 3, 256, 256))
                self.assertNotEqual(before[0], resumed.source_data.exposures)
                self.assertFalse(np.array_equal(src[1], dst[1]))
                self.assertTrue(np.isfinite(src[0]).all())
            with build_pair(source, destination, config(batch_size=3), 99,
                            previous_config=original_config) as plain:
                plain.load_state_dict(saved)
                baseline, _ = plain.batch()
                self.assertFalse(np.array_equal(src[1][0], baseline[1][0]))
                np.testing.assert_array_equal(src[2][0], baseline[2][0])

    def test_rct_to_lct_resume_uses_native_transfer_and_preserves_masks_dst_and_replay(self):
        from me_backend.data_augmentation import transfer_color
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / 'src', count=4)
            destination = make_aligned(Path(directory) / 'dst', count=4, offset=49)
            old_config = config(batch_size=1, ct_mode='rct')
            with build_pair(source, destination, old_config, 9) as original:
                original.batch()
                saved = copy.deepcopy(original.state_dict())
            changed = config(batch_size=1, ct_mode='lct', training_mask_mode='expanded_feather')
            with build_pair(source, destination, changed, 99, previous_config=old_config) as resumed:
                resumed.load_state_dict(saved)
                self.assertEqual(resumed.state_dict(), saved)
                with mock.patch('tae.data.transfer_color', wraps=transfer_color) as transfer:
                    actual_src, actual_dst = resumed.batch()
                transfer.assert_called_once()
                args = transfer.call_args.args
                self.assertEqual(args[1], 'lct')
                expected = transfer_color(*args)[None].transpose(0, 3, 1, 2)
                np.testing.assert_array_equal(actual_src[0], expected)
                np.testing.assert_array_equal(actual_src[1], expected)
                expected_state = resumed.state_dict()
            with build_pair(source, destination, changed, 0) as replay:
                replay.load_state_dict(saved)
                replay_src, replay_dst = replay.batch()
                assert_batch_equal(self, replay_src, actual_src)
                assert_batch_equal(self, replay_dst, actual_dst)
                self.assertEqual(replay.state_dict(), expected_state)
            with build_pair(source, destination, old_config, 0) as rct:
                rct.load_state_dict(saved)
                old_src, old_dst = rct.batch()
                assert_batch_equal(self, actual_dst, old_dst)
                for index in (2, 3):
                    np.testing.assert_array_equal(actual_src[index], old_src[index])
                self.assertFalse(np.array_equal(actual_src[1], old_src[1]))

    def test_disable_warp_migrates_sampler_and_color_reference_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / 'src', count=4)
            destination = make_aligned(Path(directory) / 'dst', count=4, offset=49)
            original_config = config(random_warp=True, ct_mode='lct', training_mask_mode='expanded_feather')
            with build_pair(source, destination, original_config, 9) as original:
                original.batch()
                saved = copy.deepcopy(original.state_dict())
                expected = original.batch()
                expected_state = original.state_dict()
            changed = replace(original_config, random_warp=False)
            with build_pair(source, destination, changed, 99, previous_config=original_config) as resumed:
                for side, dataset in (('src', resumed.source_data), ('dst', resumed.destination_data)):
                    predecessor = next(iter(dataset._resume_fingerprints - {dataset._legacy_fingerprint}))
                    saved[side]['fingerprint'] = predecessor
                saved['src']['reference_fingerprint'] = saved['dst']['fingerprint']
                resumed.load_state_dict(saved)
                actual = resumed.batch()
                self.assertEqual(resumed.state_dict(), expected_state)
                for new_batch, old_batch in zip(actual, expected):
                    np.testing.assert_array_equal(new_batch[0], new_batch[1])
                    self.assertFalse(np.array_equal(new_batch[0], old_batch[0]))
                    for index in (1, 2, 3):
                        np.testing.assert_array_equal(new_batch[index], old_batch[index])

    def test_learning_rate_change_migrates_all_previous_sampler_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory)/'src', count=4)
            destination = make_aligned(Path(directory)/'dst', count=4, offset=49)
            original_config = config(ct_mode='lct', training_mask_mode='expanded_feather')
            with build_pair(source, destination, original_config, 9) as original:
                original.batch()
                saved = copy.deepcopy(original.state_dict())
                expected = original.batch()
                expected_state = original.state_dict()
            changed = replace(original_config, learning_rate=original_config.learning_rate/2)
            with build_pair(source, destination, changed, 99, previous_config=original_config) as resumed:
                for src_alias, dst_alias in zip(sorted(resumed.source_data._resume_fingerprints),
                                                sorted(resumed.destination_data._resume_fingerprints)):
                    state = copy.deepcopy(saved)
                    state['src']['fingerprint'] = src_alias
                    state['dst']['fingerprint'] = dst_alias
                    state['src']['reference_fingerprint'] = dst_alias
                    resumed.load_state_dict(state)
                    actual = resumed.batch()
                    self.assertEqual(resumed.state_dict(), expected_state)
                    for left, right in zip(actual, expected):
                        assert_batch_equal(self, left, right)

    def test_eyes_mouth_change_migrates_all_previous_sampler_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory)/'src', count=4)
            destination = make_aligned(Path(directory)/'dst', count=4, offset=49)
            original_config = config(ct_mode='lct', training_mask_mode='expanded_feather')
            with build_pair(source, destination, original_config, 9) as original:
                original.batch()
                saved = copy.deepcopy(original.state_dict())
                expected = original.batch()
                expected_state = original.state_dict()
            changed = replace(original_config, eyes_mouth_weight=1.5)
            with build_pair(source, destination, changed, 99, previous_config=original_config) as resumed:
                for src_alias, dst_alias in zip(sorted(resumed.source_data._resume_fingerprints),
                                                sorted(resumed.destination_data._resume_fingerprints)):
                    state = copy.deepcopy(saved)
                    state['src']['fingerprint'] = src_alias
                    state['dst']['fingerprint'] = dst_alias
                    state['src']['reference_fingerprint'] = dst_alias
                    resumed.load_state_dict(state)
                    actual = resumed.batch()
                    self.assertEqual(resumed.state_dict(), expected_state)
                    for left, right in zip(actual, expected):
                        assert_batch_equal(self, left, right)

    def test_source_balancing_groups_and_exposure_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / "aligned", count=4)
            for index, path in enumerate(sorted(folder.glob("*.jpg"))):
                dfl = DFLJPG.load(str(path))
                dfl.set_source_filename("rare.png" if index == 3 else "common.png")
                dfl.save()
            with TAEAlignedDataset(folder, config(batch_size=1), True, 37) as data:
                self.assertEqual(sorted(map(len, data.groups)), [1, 3])
                for _ in range(80):
                    data.batch()
                self.assertEqual(sum(data.exposures), 80)
                self.assertTrue(25 <= data.exposures[3] <= 55, data.exposures)

    def test_loose_pak_and_zip_match_including_person_directories(self):
        for extension in ("pak", "zip"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory) / "aligned"
                make_aligned(folder / "alice", count=2)
                make_aligned(folder / "bob", count=2, offset=37)
                options = config(random_warp=True, random_flip=True)
                with TAEAlignedDataset(folder, options, True, 17) as loose:
                    expected = loose.batch()
                    state = copy.deepcopy(loose.state_dict())
                    next_expected = loose.batch()
                archive = pack_fixture(folder, extension)
                self.assertFalse(list(folder.rglob("*.jpg")))
                with TAEAlignedDataset(folder, options, True, 17) as packed:
                    assert_batch_equal(self, packed.batch(), expected)
                with TAEAlignedDataset(archive, options, True, 99) as direct:
                    direct.load_state_dict(state)
                    assert_batch_equal(self, direct.batch(), next_expected)

    def test_explicit_source_id_survives_native_packing(self):
        for extension in ("pak", "zip"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                folder = make_aligned(Path(directory) / "aligned", count=3)
                for index, path in enumerate(sorted(folder.glob("*.jpg"))):
                    dfl = DFLJPG.load(str(path))
                    dfl.get_dict()["source_id"] = "source-a" if index < 2 else "source-b"
                    dfl.save()
                with TAEAlignedDataset(folder, config(), True, 31) as loose:
                    self.assertEqual(sorted(map(len, loose.groups)), [1, 2])
                    fingerprint = loose.fingerprint
                    expected = loose.batch()
                archive = pack_fixture(folder, extension)
                with TAEAlignedDataset(archive, config(), True, 31) as packed:
                    self.assertEqual(sorted(map(len, packed.groups)), [1, 2])
                    self.assertEqual(packed.fingerprint, fingerprint)
                    assert_batch_equal(self, packed.batch(), expected)

    def test_failed_destination_does_not_commit_source_exposures_or_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_aligned(Path(directory) / "src", count=2)
            destination = make_aligned(Path(directory) / "dst", count=2, offset=49)
            with build_pair(source, destination, config(), 19) as paired:
                before = copy.deepcopy(paired.state_dict())
                with mock.patch.object(paired.destination_data, "batch", side_effect=ValueError("injected decoder failure")):
                    with self.assertRaisesRegex(ValueError, "injected decoder failure"):
                        paired.batch()
                self.assertEqual(paired.state_dict(), before)

    def test_changed_sample_content_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = make_aligned(Path(directory) / "aligned", count=2)
            with TAEAlignedDataset(folder, config(), True, 1) as initial:
                initial.batch()
                state = initial.state_dict()
            make_aligned(folder, count=2, offset=29)
            with TAEAlignedDataset(folder, config(), True, 1) as changed:
                with self.assertRaisesRegex(ValueError, "dataset changed"):
                    changed.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
