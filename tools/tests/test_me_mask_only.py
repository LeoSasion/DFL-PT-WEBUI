"""ME merger mask-only decoding matches the full public decoder contract."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal' / 'DeepFaceLab'))
from core.leras import nn
from me_backend.config import MEConfig
from me_backend.network import MENetwork


@pytest.mark.parametrize('archi', ['df', 'df-ud', 'df-udt', 'df-utc',
                                  'liae', 'liae-ud', 'liae-udt', 'liae-udtc'])
def test_predict_preserves_full_forward_outputs_and_omits_destination_rgb(archi):
    torch.set_num_threads(2)
    torch.manual_seed(351)
    config = MEConfig(archi=archi, resolution=64, ae_dims=32, e_dims=16,
                      d_dims=16, d_mask_dims=16, batch_size=2)
    network = MENetwork(config).eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.inference_mode():
        expected = network(image, image)
    decoders = [network.decoder_src, network.decoder_dst] if archi.startswith('df') else [network.decoder]
    calls = [0] * len(decoders)
    handles = []
    for index, decoder in enumerate(decoders):
        def record(*_, slot=index):
            calls[slot] += 1
        handles.append(decoder.upscale0.register_forward_hook(record))
    try:
        with torch.inference_mode():
            actual = network.predict(image)
    finally:
        for handle in handles:
            handle.remove()
    for output, key in zip(actual, ('swap', 'swap_mask', 'dst_mask')):
        torch.testing.assert_close(output, expected[key], rtol=0, atol=0)
    assert calls == ([1, 0] if archi.startswith('df') else [1])


@pytest.mark.parametrize('options', ['', 'd', 't', 'dtc'])
def test_mask_only_keeps_weight_names_and_mask_gradients(options):
    torch.set_num_threads(2)
    nn.initialize(nn.DeviceConfig.CPU(), data_format='NCHW')
    decoder = nn.DeepFakeArchi(64, opts=options).Decoder(16, 16, 16, name='decoder')
    # Direct mask calls also honor ModelBase's lazy-build convention.
    latent = torch.rand(2, 16, 4 if 't' not in options else 2, 4 if 't' not in options else 2, requires_grad=True)
    actual = decoder.forward_mask(latent)
    names = tuple(decoder.state_dict())
    actual.sum().backward()
    gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                 for name, parameter in decoder.named_parameters()}
    input_gradient = latent.grad.clone()
    decoder.zero_grad(set_to_none=True)
    latent.grad = None
    _, expected = decoder(latent)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    torch.testing.assert_close(latent.grad, input_gradient, rtol=0, atol=0)
    for name, parameter in decoder.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert tuple(decoder.state_dict()) == names


@pytest.mark.parametrize('options', ['', 'dtc'])
def test_native_half_decoder_keeps_float32_mask_boundary(options):
    torch.set_num_threads(2)
    nn.initialize(nn.DeviceConfig.CPU(), data_format='NCHW')
    decoder = nn.DeepFakeArchi(64, use_fp16=True, opts=options).Decoder(16, 16, 16, name='decoder')
    size = 2 if 't' in options else 4
    latent = torch.rand(1, 16, size, size, dtype=torch.float16)
    with torch.no_grad():
        actual = decoder.forward_mask(latent)
        image, expected = decoder(latent)
    assert actual.dtype == expected.dtype == image.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
