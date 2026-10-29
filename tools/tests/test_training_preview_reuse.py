"""Preview layout, transfer bounds, and atomic multi-destination publication."""
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal' / 'DeepFaceLab'))
from me_backend import web_bridge


@pytest.mark.parametrize('batch_size', [1, 4, 8])
def test_preview_copies_each_visible_output_once_and_preserves_panel_layout(monkeypatch, batch_size):
    size = 16
    src = np.zeros((batch_size, 3, size, size), np.float32)
    dst = np.full_like(src, .25)
    outputs = {key: torch.full((batch_size, channels, size, size), value)
               for key, channels, value in [('src', 3, .125), ('dst', 3, .375),
                                            ('swap', 3, .5), ('swap_mask', 1, .5), ('dst_mask', 1, .5)]}
    network = torch.nn.Identity()
    engine = SimpleNamespace(network=network, preview_outputs=lambda *_: outputs)
    copies = []
    cpu = torch.Tensor.cpu

    def record(tensor, *args, **kwargs):
        copies.append(tuple(tensor.shape))
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', record)
    preview = web_bridge.make_preview(engine, (src, src, src[:, :1], src[:, :1]),
                                      (dst, dst, dst[:, :1], dst[:, :1]))
    count = min(batch_size, 4)
    assert preview.shape == (count * size, size * 6, 3)
    for panel, value in enumerate([0., .125, .25, .375, .5, .25]):
        np.testing.assert_array_equal(preview[:, panel * size:(panel + 1) * size], value)
    assert copies == [(count, 1, size, size)] + [(count, 3, size, size)] * 3
    assert not network.training


def test_preview_publications_share_encoding_and_preserve_exact_bytes(tmp_path, monkeypatch):
    image = np.random.default_rng(8).uniform(-.2, 1.2, (48, 64, 3)).astype(np.float32)
    original = image.copy()
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    encoder = cv2.imencode
    expected = {suffix: encoder(suffix, pixels)[1].tobytes() for suffix in ('.png', '.jpg')}
    calls = []

    def encode(suffix, value):
        calls.append(suffix)
        return encoder(suffix, value)

    monkeypatch.setattr(cv2, 'imencode', encode)
    paths = [tmp_path / 'model/preview.png', tmp_path / 'runtime/preview.png', tmp_path / 'preview.jpg']
    web_bridge.atomic_images(paths, image)
    assert calls == ['.png', '.jpg']
    for path in paths:
        assert path.read_bytes() == expected[path.suffix]
        assert not list(path.parent.glob('*.tmp'))
    np.testing.assert_array_equal(image, original)
    assert web_bridge.atomic_image(tmp_path / 'single.png', image) is None
    assert (tmp_path / 'single.png').read_bytes() == expected['.png']


def test_failed_preview_replacement_retains_existing_destination(tmp_path, monkeypatch):
    path = tmp_path / 'preview.png'
    path.write_bytes(b'original')
    def fail(*_):
        raise PermissionError('synthetic replacement failure')
    monkeypatch.setattr(web_bridge.os, 'replace', fail)
    with pytest.raises(PermissionError, match='synthetic replacement'):
        web_bridge.atomic_images([path], np.zeros((16, 16, 3), np.float32))
    assert path.read_bytes() == b'original'
