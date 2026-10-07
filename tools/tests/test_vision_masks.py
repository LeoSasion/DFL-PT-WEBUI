import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'webui/python'))
from vision_assets import file_sha256
from vision_masks import FACE_PARTS, fixed_prompt, mask_statistics, validate_image, verified_identity, reject_foreign_sam2_modules


def test_no_truth_has_no_accuracy_or_winner():
    valid = np.array([[True, True], [False, False]])
    stats = mask_statistics(np.array([[1, 0], [1, 1]]), valid)
    assert stats['foregroundFraction'] == 0.5
    assert stats['iou'] is None and stats['dice'] is None


def test_truth_metrics_exclude_invalid_border():
    stats = mask_statistics(np.array([[1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[1, 1], [0, 1]]))
    assert stats['iou'] == 0.5 and stats['dice'] == 2/3
    with pytest.raises(ValueError):
        mask_statistics(np.zeros((2, 2)), np.zeros((2, 2)))


def test_prompt_and_semantic_subset_are_explicit():
    assert fixed_prompt((512, 512, 3))['point'] == [256, 245.76]
    assert not set(FACE_PARTS).intersection({0, 9, 14, 15, 16, 17, 18})
    validate_image(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        validate_image(np.zeros((32, 32, 3), dtype=np.float32))


def test_identity_hash_and_path_escape_blocked(tmp_path):
    file = tmp_path / 'weight.bin'
    file.write_bytes(b'official-test')
    item = {'path': file.name, 'sha256': file_sha256(file)}
    entry = {'id': 'test', 'weights': [item], 'sourceFiles': [], 'licenseFiles': []}
    registry = tmp_path / 'assets.json'
    registry.write_text(json.dumps({'schemaVersion': 1, 'models': [entry]}))
    assert verified_identity(tmp_path, 'test')['id'] == 'test'
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        verified_identity(tmp_path, 'test')
    entry['weights'][0]['path'] = '../outside.bin'
    registry.write_text(json.dumps({'schemaVersion': 1, 'models': [entry]}))
    with pytest.raises(ValueError, match='escapes'):
        verified_identity(tmp_path, 'test')


@pytest.mark.parametrize('source_root', ['../outside', '/absolute', 'altered-source'])
def test_registered_files_do_not_authorize_a_different_execution_tree(tmp_path, source_root):
    entry = {'id': 'sam2.1-hiera-large', 'sourceRoot': source_root, 'weights': [], 'sourceFiles': [], 'licenseFiles': []}
    (tmp_path / 'assets.json').write_text(json.dumps({'schemaVersion': 1, 'models': [entry]}))
    with pytest.raises(ValueError, match='(paths|escapes|sourceRoot)'):
        verified_identity(tmp_path, 'sam2.1-hiera-large')


def test_preimported_foreign_sam2_is_rejected(tmp_path, monkeypatch):
    module = types.ModuleType('sam2')
    module.__file__ = str(tmp_path / 'foreign' / '__init__.py')
    monkeypatch.setitem(sys.modules, 'sam2', module)
    with pytest.raises(ValueError, match='outside'):
        reject_foreign_sam2_modules(tmp_path / 'verified', [])
