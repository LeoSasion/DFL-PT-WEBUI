"""FP8 convolutions / NVFP4 Inter linears, with FP16 boundaries and masks.

Quantization is an export-only transformation. Calibration observes clean DST
aligned faces; it never performs an optimizer update. NVFP4 weight packing uses
NVIDIA Model Optimizer. The graph follows TensorRT's dynamic double-quantization
recipe and keeps the standard float32 NHWC DFM interface.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from me_backend.tensor_checks import all_finite, scalar_metrics

from .exporter import attach_weights, trace_graph

RECIPE = 'dfm-fp8-conv-nvfp4-inter-fp16-sensitive-v1'


class ExportDense(nn.Module):
    def __init__(self, layer):
        super().__init__()
        # Explicit bias makes the ONNX linear a Gemm with [out,in] weights,
        # so block-16 quantization stays on the reduction dimension.
        self.linear = nn.Linear(layer.in_ch, layer.weight.shape[1], bias=True)
        self.linear.weight = nn.Parameter((layer.weight.detach().T *
            (layer.wscale if layer.use_wscale else 1)).contiguous(), requires_grad=False)
        self.linear.bias = nn.Parameter(layer.bias.detach().clone()
            if layer.bias is not None and layer.maxout_ch <= 1 else torch.zeros(layer.weight.shape[1]), requires_grad=False)
        self.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=False) if layer.bias is not None and layer.maxout_ch > 1 else None
        self.maxout_ch, self.out_ch = layer.maxout_ch, layer.out_ch

    def forward(self, x):
        x = self.linear(x)
        if self.maxout_ch > 1:
            x = x.reshape(-1, self.out_ch, self.maxout_ch).max(dim=-1)[0]
        return x + self.bias if self.bias is not None else x


def normalize_me_layers(module):
    """Convert only the private export copy; preserve padding, scaling, maxout."""
    from core.leras.layers.Conv2D import Conv2D
    from core.leras.layers.Dense import Dense
    for name, child in list(module.named_children()):
        if isinstance(child, Conv2D):
            padding = child.pad_4
            pad = (padding[2], padding[3], padding[0], padding[1]) if padding else (child.padding,) * 4
            conv = nn.Conv2d(child.in_ch, child.out_ch, child.kernel_size,
                stride=child.strides, dilation=child.dilations, bias=child.bias is not None)
            conv.weight = nn.Parameter(child.weight.detach() * (child.wscale if child.use_wscale else 1), requires_grad=False)
            if child.bias is not None:
                conv.bias = nn.Parameter(child.bias.detach().clone(), requires_grad=False)
            setattr(module, name, nn.Sequential(nn.ZeroPad2d(pad), conv))
        elif isinstance(child, Dense):
            setattr(module, name, ExportDense(child))
        else:
            normalize_me_layers(child)


def precision_plan(module):
    layers = dict(module.named_modules())
    protected = set()
    # Preserve both ends of each codec, even the patchified 16-channel input.
    for prefix in ('network.vae.encoder.', 'network.vae.decoder.', 'network.encoder.'):
        names = [name for name, layer in layers.items() if name.startswith(prefix) and isinstance(layer, nn.Conv2d)]
        if names:
            protected.update((names[0], names[-1]))
    result = {}
    for name, layer in layers.items():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        if name in protected or any(part in name for part in ('mask_head', 'upscalem', 'out_convm', 'to_latent')):
            result[name] = 'fp16'
        elif isinstance(layer, nn.Linear):
            result[name] = 'nvfp4' if 'inter' in name.lower() and layer.in_features % 16 == 0 and layer.out_features % 16 == 0 else 'fp16'
        else:
            result[name] = 'fp8' if layer.in_channels >= 16 and layer.in_channels % 16 == 0 and layer.out_channels % 16 == 0 and layer.groups == 1 else 'fp16'
    return result


def calibration_faces(directory, config, limit):
    from me_backend.data import _load_inventory, decode_aligned, read_aligned
    if not directory:
        raise ValueError('Extreme DFM export requires an aligned DST calibration faceset')
    if type(limit) is not int or not 8 <= limit <= 512:
        raise ValueError('calibration_samples must be 8..512')
    _, references, _ = _load_inventory(directory)
    if len(references) < 8:
        raise ValueError('Extreme DFM export requires at least 8 real aligned calibration faces')
    indexes = np.linspace(0, len(references)-1, min(limit, len(references)), dtype=int)
    for index in indexes:
        ref = references[int(index)]
        decoded = decode_aligned(ref)
        image = read_aligned(ref, config, decoded=decoded, include_masks=False)[0]
        yield torch.from_numpy(np.ascontiguousarray(image[None])), {
            'name': Path(ref.path).name, 'sha256': hashlib.sha256(decoded.payload).hexdigest()}


def tensor_proto(name, dtype, shape, raw):
    return onnx.TensorProto(name=name, data_type=dtype, dims=list(shape), raw_data=raw)


def quantize_fp8(weight, name):
    weight = weight.detach().float().cpu()
    scale = max(float(weight.abs().max()) / 448., 2**-24)
    packed = (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return tensor_proto(name, onnx.TensorProto.FLOAT8E4M3FN, weight.shape,
                        packed.view(torch.uint8).numpy().tobytes()), scale


def quantize_nvfp4(weight, name):
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    weight = weight.detach().float().cpu()
    # ModelOpt's explicit global scale avoids undefined zero/zero for all-zero
    # weights and keeps the two-level scale positive.
    global_scale = torch.tensor(max(float(weight.abs().max()) / (6*448), 2**-24), dtype=torch.float32)
    packed, block_scale, global_scale = NVFP4QTensor.quantize(weight, 16, weights_scaling_factor_2=global_scale)
    return (tensor_proto(name, onnx.TensorProto.FLOAT4E2M1, weight.shape,
                         packed._quantized_data.cpu().numpy().tobytes()),
            tensor_proto(name + '.block_scale', onnx.TensorProto.FLOAT8E4M3FN, block_scale.shape,
                         block_scale.view(torch.uint8).cpu().numpy().tobytes()), global_scale.cpu())


def apply_quantization(graph, parameters, plan, activation_max):
    """Transform explicit Conv/Gemm weight edges; no global graph constant fold.

    Working one weight at a time avoids protobuf serialization of the original
    multi-gigabyte model inside a graph optimizer.
    """
    nodes, converted, packed_weights = [], {}, {}
    parameters = dict(parameters)
    def constant(name, value, dtype=torch.float16):
        parameters[name] = torch.tensor(value, dtype=dtype)
        return name
    def node(op, inputs, output, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, [output], name=output, **attrs))
        return output
    for original in graph.graph.node:
        current = copy.deepcopy(original)
        weight_name = current.input[1] if current.op_type in ('Conv', 'Gemm') else None
        name = weight_name.removesuffix('.weight') if weight_name else None
        precision = plan.get(name, 'fp16')
        if precision == 'fp16' or name not in activation_max:
            nodes.append(current)
            continue
        maximum = activation_max[name]
        if not np.isfinite(maximum) or maximum <= 0:
            nodes.append(current)
            continue
        prefix = current.name + '/dfm_quant'
        # Shared modules (the LIAE decoder) have one weight and several input edges.
        if weight_name not in packed_weights:
            original_weight = parameters.pop(weight_name)
            if precision == 'fp8':
                packed, scale = quantize_fp8(original_weight, weight_name + '.fp8')
                parameters[packed.name] = packed
                s = constant(weight_name + '.scale', scale)
                restored = node('DequantizeLinear', [packed.name, s], weight_name + '.decoded')
            else:
                packed, blocks, scale = quantize_nvfp4(original_weight, weight_name + '.fp4')
                parameters[packed.name], parameters[blocks.name] = packed, blocks
                g = constant(weight_name + '.global_scale', float(scale), torch.float32)
                s32 = node('DequantizeLinear', [blocks.name, g], weight_name + '.scales_fp32')
                s16 = node('Cast', [s32], weight_name + '.scales_fp16', to=onnx.TensorProto.FLOAT16)
                restored = node('DequantizeLinear', [packed.name, s16], weight_name + '.decoded', axis=-1, block_size=16)
            packed_weights[weight_name] = restored
            converted[name] = precision
        current.input[1] = packed_weights[weight_name]
        if precision == 'fp8':
            s = constant(prefix + '.scale', max(maximum / 448., 2**-24))
            parameters[prefix + '.zero'] = tensor_proto(prefix + '.zero', onnx.TensorProto.FLOAT8E4M3FN, [], b'\0')
            q = node('QuantizeLinear', [current.input[0], s, prefix + '.zero'], prefix + '.q')
            current.input[0] = node('DequantizeLinear', [q, s], prefix + '.dq')
        else:
            s = constant(prefix + '.scale', max(maximum / (6*448), 2**-24), torch.float32)
            nodes.append(onnx.helper.make_node('TRT_FP4DynamicQuantize', [current.input[0], s],
                [prefix + '.q', prefix + '.block_scales'], name=prefix + '.dynamic', domain='trt',
                axis=-1, block_size=16, scale_type=onnx.TensorProto.FLOAT8E4M3FN))
            b = node('DequantizeLinear', [prefix + '.block_scales', s], prefix + '.scales_fp32')
            b = node('Cast', [b], prefix + '.scales_fp16', to=onnx.TensorProto.FLOAT16)
            current.input[0] = node('DequantizeLinear', [prefix + '.q', b], prefix + '.dq', axis=-1, block_size=16)
        nodes.append(current)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    # Quantized weights replace the original graph parameter inputs.
    bound = set(parameters) | set(packed_weights)
    kept = [x for x in graph.graph.input if x.name not in bound]
    del graph.graph.input[:]
    graph.graph.input.extend(kept)
    graph.opset_import.append(onnx.helper.make_opsetid('trt', 1))
    if not converted:
        raise ValueError('This model has no supported calibrated FP8/NVFP4 layers; normal export remains available')
    return graph, parameters, converted


def _calibrate(module, config, calibration_dir, limit, device, plan, *, retain_sample=False):
    """Observe one streaming pass; retain at most one owned CPU smoke input."""
    maxima, handles, records = {}, [], []
    first_sample = None
    def observe(name):
        def hook(layer, args):
            maximum = args[0].detach().abs().max()
            # Keep per-layer reductions on the device. maximum propagates
            # NaN/Inf, including shared layers called with several inputs.
            maxima[name] = torch.maximum(maxima[name], maximum) if name in maxima else maximum
        return hook
    for name, layer in module.named_modules():
        if plan.get(name) in ('fp8', 'nvfp4'):
            handles.append(layer.register_forward_pre_hook(observe(name)))
    try:
        with torch.inference_mode():
            for image, record in calibration_faces(calibration_dir, config, limit):
                result = module(image.to(device))
                # One host decision per sample, rather than one per layer and
                # output. Failure is still reported before another face loads.
                observed = (torch.stack(tuple(maxima.values())),) if maxima else ()
                if not bool(all_finite((*observed, *result))):
                    for name, maximum in maxima.items():
                        if not bool(torch.isfinite(maximum)):
                            raise FloatingPointError('Nonfinite calibration activation: ' + name)
                    raise FloatingPointError('Nonfinite FP16 calibration output')
                if retain_sample and first_sample is None:
                    first_sample = image.detach().cpu().clone()
                records.append(record)
        print(f'[DFM extreme] calibrated {len(records)} real aligned faces', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return scalar_metrics(maxima), records, first_sample


def export_extreme(module, config, example, path, calibration_dir, limit, device, threshold,
                   *, capture_calibration_sample=None):
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        raise ValueError('Extreme export requires a CUDA GPU; use normal export on CPU')
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (10, 12):
        raise ValueError('Extreme FP8/NVFP4 export requires a Blackwell GPU')
    # Import before calibration so missing optional dependencies fail early.
    from modelopt.torch.quantization.qtensor import NVFP4QTensor  # noqa: F401
    if module.family == 'ME':
        normalize_me_layers(module)
    module.half().to(device).eval()
    module.compute_dtype = torch.float16
    plan = precision_plan(module)
    maxima, records, sample = _calibrate(module, config, calibration_dir, limit, device, plan,
        retain_sample=capture_calibration_sample is not None)
    if capture_calibration_sample is not None:
        if sample is None:
            raise ValueError('Extreme DFM export received no calibration samples')
        capture_calibration_sample(sample)
    # Trace on CPU: export must not compete for extra GPU memory with calibration.
    module.cpu()
    graph, parameters = trace_graph(module, example, path, opset=20)
    # Existing float ops keep their semantics; FP4 Q/DQ require ONNX opset 23.
    graph.opset_import[0].version = 23
    graph, parameters, converted = apply_quantization(graph, parameters, plan, maxima)
    sidecars = attach_weights(graph, parameters, path, threshold=threshold)
    return graph, sidecars, {'recipe': RECIPE, 'runtime': 'TensorRT 10.16+ / Blackwell (SM100/SM120)',
        'counts': {value: sum(item == value for item in converted.values()) for value in ('fp8', 'nvfp4')},
        'layers': {name: converted.get(name, 'fp16') for name in plan},
        'calibration': records, 'calibration_count': len(records),
        'input_output': 'float32 NHWC BGR / probability masks',
        'speedup_measured': False, 'accuracy_certified': False}
