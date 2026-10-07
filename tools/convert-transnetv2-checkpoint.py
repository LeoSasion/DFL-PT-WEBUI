"""Maintainer-only official TransNetV2 checkpoint conversion without a TF runtime.

The optional pure Python tensorflow-checkpoint-reader source parses the binary
file format only. It is not part of any application runtime dependency profile.
The pinned official conversion functions supply all tensor remapping operations.
"""
import argparse
import ast
import collections
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(directory, reader_source):
    import numpy as np
    import torch
    directory, reader_source = directory.resolve(), reader_source.resolve()
    pins = {'transnetv2_pytorch.py': 'f7c1d437465579a8ec28a5add19853d2cb2755248ea4a4207678210a609428e1',
            'convert_weights.py': '3572d76ddccc92e7ccca13ac3c47aaf39c86f56ce4e392d4d0d6a60da05b42fa',
            'variables.index': '8b99e28b4ad11372d9a1ad9703298c2e370df14859da4245fdbe818e92dd403f',
            'variables.data-00000-of-00001': 'b8c9dc3eb807583e6215cabee9ca61737b3eb1bceff68418b43bf71459669367'}
    for name, expected in pins.items():
        if sha(directory / name) != expected:
            raise ValueError('Official source/checkpoint differs: ' + name)
    output = directory / 'transnetv2-pytorch-weights.pth'
    if output.exists():
        raise ValueError('Conversion output already exists; preserve it and use a fresh directory')
    # Confined compatibility for this one-shot binary-reader process. No files,
    # installed packages, model implementation, or production runtime are edited.
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    if not hasattr(np, 'string_'):
        np.string_ = np.bytes_
    if not hasattr(np, 'unicode_'):
        np.unicode_ = np.str_
    if not hasattr(inspect, 'ArgSpec'):
        inspect.ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')
        inspect.getargspec = lambda function: inspect.ArgSpec(*inspect.getfullargspec(function)[:4])
    if os.name == 'nt':
        original_open = os.open
        os.open = lambda path, flags, *args: original_open(path, flags | os.O_BINARY, *args)
        def pread(fd, size, offset):
            previous = os.lseek(fd, 0, os.SEEK_CUR)
            os.lseek(fd, offset, os.SEEK_SET)
            try:
                return os.read(fd, size)
            finally:
                os.lseek(fd, previous, os.SEEK_SET)
        os.pread = pread
    sys.path.insert(0, str(reader_source))
    from tensorflow_checkpoint_reader.py_checkpoint_reader import NewCheckpointReader
    from tensorflow_checkpoint_reader.pb.tensorflow.core.protobuf.trackable_object_graph_pb2 import TrackableObjectGraph
    reader = NewCheckpointReader(str(directory / 'variables'))
    graph = TrackableObjectGraph()
    # The reader's string dtype exposes C++ object pointers as numpy objects;
    # never dereference those. Decode the single string from the pinned binary
    # bundle format (varint length, four-byte length checksum, payload).
    entry = next(value for key, value in reader._reader._entries if key == b'_CHECKPOINTABLE_OBJECT_GRAPH')
    if entry.shard_id != 0 or entry.shape.dim:
        raise ValueError('Unsupported checkpoint object-graph shape')
    shard = (directory / 'variables.data-00000-of-00001').read_bytes()
    raw = shard[entry.offset:entry.offset + entry.size]
    length, position, shift = 0, 0, 0
    while position < len(raw) and shift < 64:
        byte = raw[position]
        length |= (byte & 127) << shift
        position += 1
        if byte < 128:
            break
        shift += 7
    if position + 4 + length != len(raw):
        raise ValueError('Invalid checkpoint object-graph string length')
    graph.ParseFromString(raw[position + 4:])
    functions = ast.parse((directory / 'convert_weights.py').read_text(encoding='utf-8'))
    functions.body = [node for node in functions.body if isinstance(node, ast.FunctionDef) and
                      node.name in ('remap_name', 'remap_tensor', 'check_and_fix_dicts')]
    namespace = {'np': np, 'torch': torch}
    exec(compile(functions, str(directory / 'convert_weights.py'), 'exec'), namespace)
    class TensorArray:
        def __init__(self, array):
            self.array = array
        def numpy(self):
            return self.array
    state, mappings = {}, []
    for node in graph.nodes:
        for attribute in node.attributes:
            if attribute.name != 'VARIABLE_VALUE':
                continue
            original = attribute.full_name
            original = original if original.endswith(':0') else original + ':0'
            name = namespace['remap_name'](original)
            if name in state:
                raise ValueError('Duplicate checkpoint variable mapping: ' + name)
            array = reader.get_tensor(attribute.checkpoint_key)
            if array is None or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('Unsupported/invalid original tensor: ' + attribute.checkpoint_key)
            tensor_entry = next(value for key, value in reader._reader._entries if key.decode() == attribute.checkpoint_key)
            if array.tobytes() != shard[tensor_entry.offset:tensor_entry.offset + tensor_entry.size]:
                raise ValueError('Reader tensor differs from original checkpoint bytes: ' + attribute.checkpoint_key)
            mapped = namespace['remap_tensor'](TensorArray(array.copy()))
            # Verify a lossless inverse permutation against every original value.
            inverse = mapped.numpy()
            if array.ndim == 5:
                inverse = inverse.transpose(2, 3, 4, 1, 0)
            elif array.ndim == 2:
                inverse = inverse.T
            if not np.array_equal(array, inverse):
                raise ValueError('Tensor conversion was not lossless: ' + name)
            state[name] = mapped
            mappings.append({'checkpointKey': attribute.checkpoint_key, 'originalName': original,
                'torchName': name, 'sourceShape': list(array.shape), 'torchShape': list(mapped.shape),
                'sourceFloat32Sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                'torchFloat32Sha256': hashlib.sha256(mapped.numpy().tobytes()).hexdigest()})
    spec = importlib.util.spec_from_file_location('_official_transnet_conversion', directory / 'transnetv2_pytorch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TransNetV2().eval()
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    if not namespace['check_and_fix_dicts'](state, shapes):
        raise ValueError('Official model/checkpoint shape or key mismatch')
    model.load_state_dict(state, strict=True)
    torch.save(model.state_dict(), output)
    reloaded = torch.load(output, map_location='cpu', weights_only=True)
    if any(not torch.equal(value, reloaded[name]) for name, value in model.state_dict().items()):
        raise ValueError('Saved PyTorch state did not round trip exactly')
    if 'tensorflow' in sys.modules:
        raise ValueError('A TensorFlow runtime was imported; conversion must parse data only')
    record = {'schemaVersion': 1, 'repository': 'https://github.com/soCzech/TransNetV2',
        'revision': '85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed', 'sourceLicense': 'MIT',
        'weightOrigin': 'official-repository-Git-LFS-checkpoint',
        'conversion': 'pinned-official-name-and-axis-remapping; pure-Python-binary-checkpoint-reader',
        'reader': {'package': 'tensorflow-checkpoint-reader', 'version': '0.2.2',
            'repository': 'https://github.com/shawwn/tensorflow-checkpoint-reader', 'license': 'MIT'},
        'files': [{'file': name, 'sha256': value, 'sizeBytes': (directory / name).stat().st_size}
                  for name, value in pins.items()] + [{'file': output.name, 'sha256': sha(output), 'sizeBytes': output.stat().st_size}],
        'mappedFloat32Variables': len(mappings), 'stateKeyCount': len(reloaded),
        'everyTensorInversePermutationExact': True, 'savedStateExact': True,
        'tensorflowRuntimeImported': False, 'tensorflowForwardEquivalenceTested': False,
        'torchForwardFunctionalTested': False, 'tensorMappings': mappings}
    (directory / 'identity.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in record.items() if key != 'tensorMappings'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-source', required=True, type=Path)
    parser.add_argument('--reader-source', required=True, type=Path)
    arguments = parser.parse_args()
    convert(arguments.model_source, arguments.reader_source)
