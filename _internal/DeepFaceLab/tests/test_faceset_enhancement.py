"""Restoration publication must preserve originals and invalidate pixel-bound audits."""
import copy
from pathlib import Path
import numpy as np
import pytest
from DFLIMG import DFLJPG
from mainscripts import FacesetEnhancer
from tests.me_fixtures import make_aligned


class FixtureEnhancer:
    def __init__(self, model_id, **kwargs): self.provenance = {'id': model_id, 'fixture': True}
    def enhance(self, image, **kwargs): return np.clip(image * .9, 0, 1).astype(np.float32)


def test_selected_batch_publishes_complete_independent_copy_and_clears_native_audit(tmp_path, monkeypatch):
    aligned = make_aligned(tmp_path / 'aligned', count=3)
    paths = sorted(aligned.glob('*.jpg'))
    dfl = DFLJPG.load(paths[1]); data = copy.deepcopy(dfl.get_dict())
    data['native_landmarks98'] = {'pixelBoundAudit': True}; dfl.set_dict(data); dfl.save()
    originals = {p.name: p.read_bytes() for p in paths}
    for source in paths[:2]: source.with_suffix(source.suffix + '.landmarks.json').write_text('{"pixelAudit":"source-only"}', encoding='utf-8')
    monkeypatch.setattr(FacesetEnhancer, 'AlignedFaceEnhancer', FixtureEnhancer)
    receipt = FacesetEnhancer.process_folder(aligned, cpu_only=True, offset=1, limit=1)
    output = Path(receipt['facesetPath'])
    assert receipt['selectedRange'] == {'start': 2, 'end': 2, 'total': 3, 'limit': 500}
    assert len(list(output.glob('*.jpg'))) == 3 and receipt['needsReview']
    assert {p.name: p.read_bytes() for p in paths} == originals
    assert (output / paths[0].name).read_bytes() == originals[paths[0].name]
    assert (output / (paths[0].name + '.landmarks.json')).read_bytes() == paths[0].with_suffix(paths[0].suffix + '.landmarks.json').read_bytes()
    assert not (output / (paths[1].name + '.landmarks.json')).exists()
    restored = DFLJPG.load(output / paths[1].name)
    assert 'native_landmarks98' not in restored.get_dict()
    assert restored.get_dict()['enhancement_provenance']['annotationsRetainedForReview']
    np.testing.assert_array_equal(restored.get_landmarks(), data['landmarks'])
    (output / paths[0].name).write_bytes(b'independent edit')
    assert paths[0].read_bytes() == originals[paths[0].name]


def test_stale_input_blocks_publication_and_retains_failed_receipt(tmp_path, monkeypatch):
    aligned = make_aligned(tmp_path / 'aligned', count=1); source = next(aligned.glob('*.jpg'))
    class Stale(FixtureEnhancer):
        def enhance(self, image, **kwargs):
            source.write_bytes(source.read_bytes() + b'changed')
            return super().enhance(image, **kwargs)
    monkeypatch.setattr(FacesetEnhancer, 'AlignedFaceEnhancer', Stale)
    with pytest.raises(RuntimeError, match='changed'):
        FacesetEnhancer.process_folder(aligned, cpu_only=True)
    root = aligned.parent / 'aligned_enhanced'
    assert not list(root.glob('enhance-*'))
    assert len(list(root.glob('.*.pending/enhancement.receipt.json'))) == 1


@pytest.mark.parametrize('model', ['FaceEnhancer', 'swinir-psnr', 'unknown'])
def test_retired_models_are_rejected_before_any_output(tmp_path, model):
    aligned = make_aligned(tmp_path / 'aligned', count=1)
    with pytest.raises(ValueError, match='fixed model'):
        FacesetEnhancer.process_folder(aligned, model_id=model)
    assert not (aligned.parent / 'aligned_enhanced').exists()
