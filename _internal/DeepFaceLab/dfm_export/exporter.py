"""Bounded-memory ONNX serialization with recoverable DFM bundle publication.

Export the graph WITHOUT parameters first. Attaching external initializers only
after tracing avoids the protobuf 2 GiB limit in the exporter itself, not just in
the final save. No training checkpoint is modified.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import uuid

import onnx
import torch
from core.file_hash import file_sha256

from .publication import publish_bundle, recover_publication

INPUT = 'in_face:0'
OUTPUTS = ['out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0']
# Leave room for graph/metadata in the single protobuf. This is a packaging
# threshold, never an upper limit on model size.
EXTERNAL_THRESHOLD = 1536 * 1024**2
CHUNK_BYTES = 16 * 1024**2


class DFMNetwork(torch.nn.Module):
    """NHWC BGR [0,1] in; DST mask, swapped BGR, SRC mask out."""
    def __init__(self, network, family='ME'):
        super().__init__()
        self.network, self.family = network, family
        self.compute_dtype = torch.float32

    def forward(self, face):
        x = face.permute(0, 3, 1, 2).to(dtype=self.compute_dtype)
        if self.family == 'ME':
            swap, source_mask, destination_mask = self.network.predict(x)
        else:
            # Pure inference graph: training validation/freeze assertions are
            # outside the graph. Include the pinned encoder and SAVED decoder.
            n = self.network
            rgb = x[:, [2, 1, 0]]
            latent = n.vae.encoder(torch.cat((rgb, torch.ones_like(rgb[:, :1])), 1))
            ab = n.inter_AB(latent)
            swapped = n._features(latent, 'swap', ab=ab)
            decoded = n.vae.decoder(n.to_latent(swapped))[:, :3].float().clamp(0, 1)
            swap = decoded[:, [2, 1, 0]]
            size = (n.resolution, n.resolution)
            source_mask = n._mask(swapped, size)
            destination_mask = n._mask(n._features(latent, 'dst', ab=ab), size)
        return tuple(t.float().permute(0, 2, 3, 1) for t in (destination_mask, swap, source_mask))


def load_network(directory, family, weights=None):
    """Read inference state without allocating/restoring Adam on the GPU."""
    if family not in ('ME', 'TAE'):
        raise ValueError('DFM family must be ME or TAE')
    checkpoint = directory / ('me.pt' if family == 'ME' else 'tae.pt')
    if family == 'TAE':
        from tae.engine import TAEEngine
        engine = TAEEngine.load(checkpoint, 'cpu', weights=weights, training=False)
        return engine.network.eval(), engine.config, engine.iteration, checkpoint
    from me_backend.config import MEConfig
    from me_backend.network import MENetwork
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != 'me-pytorch' or payload.get('version') != 1:
        raise ValueError('Unsupported ME checkpoint')
    config = MEConfig.from_dict(payload['config'])
    network = MENetwork(config)
    network.load_state_dict(payload['model'], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in network.parameters()):
        raise ValueError('Nonfinite ME checkpoint')
    iteration = payload.get('iteration')
    if type(iteration) is not int or iteration < 0:
        raise ValueError('Invalid checkpoint iteration')
    return network.eval(), config, iteration, checkpoint


def trace_graph(module, example, path, *, opset=17):
    """Trace only structure. Parameter tensors never enter a protobuf here."""
    with torch.no_grad():
        torch.onnx.export(module, example, str(path), input_names=[INPUT],
            output_names=OUTPUTS, export_params=False, do_constant_folding=False,
            dynamic_axes={name: {0: 'batch'} for name in (INPUT, *OUTPUTS)},
            opset_version=opset, dynamo=False)
    graph = onnx.load(str(path), load_external_data=False)
    state = module.state_dict()
    used = {}
    for value in graph.graph.input:
        if value.name == INPUT:
            continue
        if value.name not in state:
            raise ValueError('Exported parameter cannot be bound: ' + value.name)
        used[value.name] = state[value.name].detach()
    return graph, used


def attach_weights(graph, weights, graph_path, *, threshold=EXTERNAL_THRESHOLD):
    """Stream weights; never serialize a >2 GiB ModelProto or TensorProto."""
    total = sum(len(t.raw_data) if isinstance(t, onnx.TensorProto) else t.numel() * t.element_size()
                for t in weights.values())
    external = total >= threshold
    data_path = graph_path.with_name(graph_path.name + '.' + uuid.uuid4().hex + '.data')
    stream = open(data_path, 'xb') if external else None
    try:
        for name, tensor in weights.items():
            if isinstance(tensor, onnx.TensorProto):
                initializer = onnx.TensorProto(name=name, data_type=tensor.data_type, dims=list(tensor.dims))
                raw = memoryview(tensor.raw_data)
            else:
                tensor = tensor.detach().cpu().contiguous()
                dtype = onnx.helper.np_dtype_to_tensor_dtype(tensor.numpy().dtype)
                initializer = onnx.TensorProto(name=name, data_type=dtype, dims=list(tensor.shape))
                raw = memoryview(tensor.numpy()).cast('B')
            if stream:
                offset = stream.tell()
                for start in range(0, len(raw), CHUNK_BYTES):
                    stream.write(raw[start:start+CHUNK_BYTES])
                initializer.data_location = onnx.TensorProto.EXTERNAL
                for key, value in {'location': data_path.name, 'offset': offset, 'length': len(raw)}.items():
                    item = initializer.external_data.add()
                    item.key, item.value = key, str(value)
            else:
                initializer.raw_data = raw.tobytes()
            graph.graph.initializer.append(initializer)
        kept = [value for value in graph.graph.input if value.name not in weights]
        del graph.graph.input[:]
        graph.graph.input.extend(kept)
        if stream:
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if stream:
            stream.close()
    return [data_path] if external else []


def _engine_smoke(runner, image, reference, *, max_batch=4, batch_atol=0.02):
    """Validate real output and dynamic batching, independently of quality."""
    import numpy as np
    if type(max_batch) is not int or not 1 <= max_batch <= 64:
        raise ValueError('TensorRT smoke max_batch must be 1..64')
    height, width = image.shape[1:3]

    def infer(batch):
        actual = runner(np.repeat(image, batch, axis=0))
        expected = [(batch, height, width, channels) for channels in (1, 3, 1)]
        if len(actual) != 3 or any(value.shape != shape for value, shape in zip(actual, expected)):
            raise ValueError('TensorRT DFM returned an incorrect output shape')
        if not all(np.isfinite(value).all() for value in actual):
            raise FloatingPointError('Nonfinite TensorRT DFM smoke output')
        return actual

    actual = infer(1)
    batches = [{'batch': 1, 'max_abs_error': [0., 0., 0.]}]
    for batch in [*dict.fromkeys([min(2, max_batch), max_batch]), 1]:
        values = infer(batch)
        errors = [float(np.max(np.abs(value - base))) for value, base in zip(values, actual)]
        if any(error > batch_atol for error in errors):
            raise RuntimeError(f'TensorRT DFM batch consistency failed at batch {batch}: {errors}')
        batches.append({'batch': batch, 'max_abs_error': errors})
    return {'finite': True, 'samples': 1,
        'reference': 'export FP16 network before quantization',
        'max_abs_error': [float(np.max(np.abs(a-b))) for a, b in zip(actual, reference)],
        'mean_abs_error': [float(np.mean(np.abs(a-b))) for a, b in zip(actual, reference)],
        'batch_consistency': {'absolute_tolerance': batch_atol, 'batches': batches}}


def export_dfm(model_directory, output=None, *, family='ME', mode='normal',
               calibration_dir=None, calibration_samples=64, device='cuda:0',
               weights=None, external_threshold=EXTERNAL_THRESHOLD, export_id=None):
    if mode not in ('normal', 'extreme'):
        raise ValueError('DFM mode must be normal or extreme')
    directory = Path(model_directory).resolve()
    output = Path(output).resolve() if output else directory / ('model.dfm' if mode == 'normal' else 'model_extreme.dfm')
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / ('me.pt' if family == 'ME' else 'tae.pt')
    if output == checkpoint or output.suffix.lower() not in ('.dfm', '.onnx'):
        raise ValueError('Output must be a separate .dfm or .onnx file')
    # Recover before loading/tracing a model. A locked or conflicting old bundle
    # must not trigger another expensive export or overwrite its recovery state.
    recover_publication(output)
    network, config, iteration, checkpoint = load_network(directory, family, weights)
    module = DFMNetwork(network, family).eval()
    example = torch.zeros(1, config.resolution, config.resolution, 3)
    metadata = {'model_type': family + '-PyTorch', 'face_type': config.face_type,
                'dfm_export_mode': mode, 'checkpoint_sha256': file_sha256(checkpoint)}
    # Staging is a sibling: external paths stay relative after publication.
    with tempfile.TemporaryDirectory(prefix='.dfm-export-', dir=output.parent) as staging:
        stage = Path(staging)
        graph_path = stage / output.name
        if mode == 'normal':
            graph, parameters = trace_graph(module, example, graph_path)
            sidecars = attach_weights(graph, parameters, graph_path, threshold=external_threshold)
            precision = {'recipe': 'fp32', 'runtime': 'ONNX Runtime / compatible DFM loader; external weights require loading by path'}
        else:
            from .quantization import export_extreme
            smoke_inputs = []
            graph, sidecars, precision = export_extreme(module, config, example, graph_path,
                calibration_dir, calibration_samples, device, external_threshold,
                capture_calibration_sample=smoke_inputs.append)
        for key, value in metadata.items():
            prop = graph.metadata_props.add()
            prop.key, prop.value = key, value
        onnx.save_model(graph, str(graph_path))
        # Always pass the pathname: checker(ModelProto) itself has a 2 GiB cap.
        onnx.checker.check_model(str(graph_path))
        extra_files = []
        if mode == 'extreme':
            from .runtime import build_engine, TensorRTRunner
            engine_path, layers_path, engine_info = build_engine(graph_path, device)
            runner = TensorRTRunner(engine_path, device)
            # A real aligned input validates the compiled engine, not only parsing.
            # Reuse the first calibrated CPU snapshot rather than rescanning and
            # decoding the dataset after a potentially long engine build.
            sample, = smoke_inputs
            with torch.inference_mode():
                reference = [x.cpu().numpy() for x in module(sample)]
            precision['engine'] = engine_info
            precision['engine_smoke'] = _engine_smoke(runner, sample.numpy(), reference,
                max_batch=engine_info.get('max_batch', 4))
            extra_files = [engine_path, layers_path]
        report = {'version': 1, 'family': family, 'mode': mode, 'iteration': iteration,
                  'export_id': export_id,
                  'checkpoint_sha256': metadata['checkpoint_sha256'], 'resolution': config.resolution,
                  'face_type': config.face_type, 'precision': precision,
                  'external_data': bool(sidecars), 'files': [
                      {'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_sha256(p)}
                      for p in [graph_path, *sidecars, *extra_files]]}
        report_name = output.name + '.export.json'
        (stage / report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        publish_bundle(output, [*sidecars, *extra_files, graph_path, stage / report_name])
    print(f'[DFM export] {output}; mode={mode}; external_data={bool(sidecars)}', flush=True)
    return output
