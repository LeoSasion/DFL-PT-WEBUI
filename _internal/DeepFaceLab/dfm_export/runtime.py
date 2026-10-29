"""TensorRT build and bounded inference for the extreme DFM export contract."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch


def _builder_optimization_level(network, trt, capability):
    # On SM120 / TRT 10.16, the fused NVFP4 Gemm -> dynamic-quantize
    # epilogue can corrupt every row after the first. Level 0 retains real
    # NVFP4 CUTLASS Gemms with separate quantization, verified across batches.
    if tuple(capability) == (12, 0) and str(trt.__version__).startswith('10.16.'):
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if (layer.type == trt.LayerType.DYNAMIC_QUANTIZE
                    and layer.get_output(0).dtype == trt.fp4):
                return 0
    return 3


def build_engine(dfm_path, device='cuda:0', *, max_batch=4):
    import tensorrt as trt
    path = Path(dfm_path)
    with torch.cuda.device(device):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(path)):
            raise RuntimeError('TensorRT rejected DFM: ' + '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        if network.num_inputs != 1 or network.get_input(0).name != 'in_face:0':
            raise ValueError('Unexpected DFM input contract')
        _, height, width, channels = network.get_input(0).shape
        profile = builder.create_optimization_profile()
        profile.set_shape('in_face:0', (1, height, width, channels),
                          (1, height, width, channels), (max_batch, height, width, channels))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
        capability = torch.cuda.get_device_capability(device)
        optimization_level = _builder_optimization_level(network, trt, capability)
        config.builder_optimization_level = optimization_level
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.add_optimization_profile(profile)
        print('[DFM extreme] building TensorRT engine for this GPU', flush=True)
        start = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT could not build the quantized DFM; no output was published')
        engine_path = path.with_suffix(path.suffix + '.engine')
        engine_path.write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError('TensorRT could not reload the exported engine')
        inspection = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        layers_path = path.with_suffix(path.suffix + '.layers.json')
        layers_path.write_text(json.dumps(inspection, ensure_ascii=False, indent=2), encoding='utf-8')
        return engine_path, layers_path, {'tensorrt_version': trt.__version__,
            'gpu': torch.cuda.get_device_name(device), 'compute_capability': list(capability),
            'build_seconds': time.perf_counter()-start, 'max_batch': max_batch,
            'builder_optimization_level': optimization_level,
            'nvfp4_output_fusion_workaround': optimization_level == 0,
            'engine_portability': 'built for this GPU / TensorRT version; rebuild from DFM on the target'}


class TensorRTRunner:
    """Run DFM NHWC BGR tensors without ONNX Runtime CPU fallback."""
    def __init__(self, path, device='cuda:0'):
        import tensorrt as trt
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None:
                raise ValueError('Cannot load TensorRT engine on this GPU/runtime')
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream(device=self.device)
            self.device = self.stream.device
            # Engine names and types are immutable; only resolved shapes vary
            # when callers change the batch. Keep no output/device buffer cache.
            output_specs = []
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                    output_specs.append((name, dtype))
            self._output_specs = tuple(output_specs)

    def __call__(self, value):
        producer = (torch.cuda.current_stream(value.device)
                    if isinstance(value, torch.Tensor) and value.device == self.device else None)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if producer is not None:
                # A same-device tensor may still be written asynchronously on
                # the caller's current stream. NumPy inputs need no such wait.
                self.stream.wait_stream(producer)
            x = torch.as_tensor(value, device=self.device, dtype=torch.float32).contiguous()
            if not self.context.set_input_shape('in_face:0', tuple(x.shape)):
                raise ValueError('Input shape is outside the exported TensorRT profile')
            tensors = {'in_face:0': x}
            for name, dtype in self._output_specs:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise ValueError('Unresolved TensorRT output shape')
                tensors[name] = torch.empty(shape, dtype=dtype, device=self.device)
            for name, tensor in tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('TensorRT execution failed')
            self.stream.synchronize()
            outputs = [tensors[name].cpu().numpy() for name in
                       ('out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0')]
        if not all(np.isfinite(out).all() for out in outputs):
            raise FloatingPointError('Nonfinite quantized DFM output')
        return outputs
