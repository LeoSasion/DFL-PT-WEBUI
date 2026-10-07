import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal/DeepFaceLab'))
from mainscripts.Merger import select_preview_frames, validate_preview_output_paths


def test_preview_selects_explicit_window_without_silent_clamping():
    paths = [f'{index:05}.png' for index in range(1, 31)]
    assert select_preview_frames(paths, 3, 2) == ['00003.png', '00004.png']
    assert len(select_preview_frames(paths, 1, 20)) == 20
    for start, count in [(0, 2), (True, 2), (1, 21), (1, 0), (29, 3), (1, 2.0)]:
        with pytest.raises(ValueError):
            select_preview_frames(paths, start, count)


def test_preview_output_guard_rejects_formal_merge_and_other_workspace_before_creation(tmp_path):
    source = tmp_path / 'project' / 'data_dst'
    source.mkdir(parents=True)
    preview = source.parent / '.webui' / 'merge-previews' / 'preview-test-id'
    assert validate_preview_output_paths(source, preview / 'merged', preview / 'merged_mask') == preview
    assert not preview.exists()
    for merged, mask in [
        (source / 'merged', source / 'merged_mask'),
        (preview / 'merged', source / 'merged_mask'),
        (tmp_path / 'other' / '.webui' / 'merge-previews' / 'preview-test-id' / 'merged', preview / 'merged_mask'),
    ]:
        with pytest.raises(ValueError):
            validate_preview_output_paths(source, merged, mask)
    assert not (source / 'merged').exists()


def test_preview_never_reuses_a_populated_output_directory(tmp_path):
    source = tmp_path / 'data_dst'
    source.mkdir()
    preview = tmp_path / '.webui' / 'merge-previews' / 'preview-existing'
    (preview / 'merged').mkdir(parents=True)
    kept = preview / 'merged' / 'keep.png'
    kept.write_bytes(b'existing result')
    with pytest.raises(ValueError, match='unused'):
        validate_preview_output_paths(source, preview / 'merged', preview / 'merged_mask')
    assert kept.read_bytes() == b'existing result'
