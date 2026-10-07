"""Verify SAME transposed convolution against an independent forward adjoint."""
import pytest
import torch
import torch.nn.functional as F

from core.leras import nn


@pytest.fixture(autouse=True)
def cpu_layers():
    nn.initialize(nn.DeviceConfig.CPU(), data_format='NCHW')
    yield
    nn.set_data_format('NCHW')


def test_xseg_upsample_keeps_documented_same_origin():
    layer = nn.Conv2DTranspose(1, 1, 3, strides=2, padding='SAME', use_bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(1., 10.).reshape(1, 1, 3, 3))
    actual = layer(torch.tensor([[[[1., 2.], [3., 4.]]]]))
    expected = torch.tensor([[[[1., 2., 5., 4.], [4., 5., 14., 10.],
                               [10., 14., 36., 24.], [12., 15., 34., 20.]]]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('layout', ['NCHW', 'NHWC'])
@pytest.mark.parametrize('kernel', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('stride', [1, 2, 3])
def test_same_output_and_gradients_match_forward_convolution_adjoint(layout, kernel, stride):
    nn.set_data_format(layout)
    generator = torch.Generator().manual_seed(6408)
    layer = nn.Conv2DTranspose(2, 3, kernel, strides=stride, padding='SAME')
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator))
        layer.bias.copy_(torch.tensor([0.2, -0.3, 0.4]))
    x = torch.randn((2, 2, 4, 5), generator=generator, requires_grad=True)
    supplied = x if layout == 'NCHW' else x.permute(0, 2, 3, 1).contiguous()
    actual = layer(supplied)
    if layout == 'NHWC':
        actual = actual.permute(0, 3, 1, 2)

    # The transpose is independently derived as the gradient of a forward
    # convolution, including asymmetric SAME padding and non-square images.
    z = torch.zeros((2, 3, 4 * stride, 5 * stride), requires_grad=True)
    total = max(kernel - stride, 0)
    leading = total // 2
    forward = F.conv2d(F.pad(z, (leading, total-leading, leading, total-leading)),
                       layer.weight, stride=stride)
    reference = torch.autograd.grad((forward * x).sum(), z, create_graph=True)[0]
    reference = reference + layer.bias[None, :, None, None]
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=3e-6)
    direction = torch.randn(actual.shape, generator=generator)
    actual_gradients = torch.autograd.grad((actual * direction).sum(),
                                          (x, layer.weight, layer.bias), retain_graph=True)
    reference_gradients = torch.autograd.grad((reference * direction).sum(),
                                             (x, layer.weight, layer.bias))
    for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=2e-5, atol=5e-6)
