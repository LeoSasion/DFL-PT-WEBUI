"""Strict, CPU-only import of local non-RG TensorFlow ME network weights.

DFL's ``.npy`` files contain pickled NumPy mappings, not NumPy archives. Both
the original whole-dictionary and the later count + single-tensor records are
supported. The caller supplies the exact saved-model filename prefix and the
source configuration: activation options cannot be inferred from tensor shapes.
Legacy optimizer, discriminator, iteration, data and RNG state are not resumed.
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

from .config import MEConfig


class LegacyMEImportError(ValueError):
    pass


class _NumpyUnpickler(pickle.Unpickler):
    """Only reconstruct numeric NumPy arrays; never import checkpoint code."""
    def find_class(self, module, name):
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
            return np._core.multiarray._reconstruct
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            return np._core.numeric._frombuffer
        raise LegacyMEImportError(f'Unsupported pickle global: {module}.{name}')


class _HashingReader:
    def __init__(self, stream):
        self.stream, self.hash = stream, hashlib.sha256()

    def read(self, count=-1):
        data = self.stream.read(count)
        self.hash.update(data)
        return data

    def readline(self, count=-1):
        data = self.stream.readline(count)
        self.hash.update(data)
        return data


def legacy_me_components(archi):
    """The exact non-RG DeepFakeArchi component contract from Model_ME."""
    if not isinstance(archi, str):
        raise LegacyMEImportError('An explicit TF ME architecture is required')
    family, separator, options = archi.partition('-')
    if (family not in ('df', 'liae') or '-' in options
            or (separator and not options) or any(option not in 'udtc' for option in options)
            or len(set(options)) != len(options)):
        raise LegacyMEImportError('Unsupported legacy ME architecture; only non-RG df/liae with unique u/d/t/c options are supported')
    return ('encoder', 'inter', 'decoder_src', 'decoder_dst') if family == 'df' else ('encoder', 'inter_AB', 'inter_B', 'decoder')


def _named_parameters(model, prefix=''):
    model.build()
    for layer in model.layers:
        name = prefix + layer.name
        if hasattr(layer, 'layers'):
            yield from _named_parameters(layer, name + '/')
        else:
            for parameter_name, parameter in layer.named_parameters(recurse=False):
                yield f'{name}/{parameter_name}', layer, parameter_name, parameter


def _read_mapping(path, expected_count):
    try:
        with path.open('rb') as stream:
            reader = _HashingReader(stream)
            first = _NumpyUnpickler(reader).load()
            if type(first) is int:
                if first != expected_count:
                    raise LegacyMEImportError(f'{path.name}: tensor count {first} != {expected_count}')
                mapping = {}
                for _ in range(first):
                    record = _NumpyUnpickler(reader).load()
                    if not isinstance(record, dict) or len(record) != 1:
                        raise LegacyMEImportError(f'{path.name}: each tensor record must contain exactly one name')
                    name, value = next(iter(record.items()))
                    if not isinstance(name, str) or name in mapping:
                        raise LegacyMEImportError(f'{path.name}: invalid or duplicate tensor name')
                    mapping[name] = value
                container = 'counted-tensor-records'
            elif isinstance(first, dict):
                mapping, container = first, 'tensor-dictionary'
            else:
                raise LegacyMEImportError(f'{path.name}: expected a TF tensor mapping or tensor count')
            if reader.read(1):
                raise LegacyMEImportError(f'{path.name}: unexpected trailing data')
            if len(mapping) != expected_count:
                raise LegacyMEImportError(f'{path.name}: tensor count {len(mapping)} != {expected_count}')
            return mapping, container, reader.hash.hexdigest()
    except LegacyMEImportError:
        raise
    except (OSError, EOFError, pickle.UnpicklingError, TypeError, ValueError, OverflowError) as error:
        raise LegacyMEImportError(f'{path.name}: invalid TF weights file: {error}') from error


def _prepare_component(model, mapping, filename):
    named = list(_named_parameters(model))
    normalized = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise LegacyMEImportError(f'{filename}: tensor names must be strings')
        key = key.removesuffix(':0')
        if key.startswith(model.name + '/'):
            key = key[len(model.name) + 1:]
        if key in normalized:
            raise LegacyMEImportError(f'{filename}: duplicate normalized tensor name {key}')
        normalized[key] = value
    expected = {name for name, _, _, _ in named}
    if set(normalized) != expected:
        missing, unexpected = sorted(expected - set(normalized)), sorted(set(normalized) - expected)
        raise LegacyMEImportError(f'{filename}: TF weight names mismatch (missing={missing[:3]}, unexpected={unexpected[:3]})')
    if {id(parameter) for _, _, _, parameter in named} != {id(parameter) for parameter in model.get_weights()}:
        raise LegacyMEImportError(f'{filename}: unsupported or incomplete network parameter contract')
    prepared = []
    for name, layer, parameter_name, parameter in named:
        array = normalized[name]
        if not isinstance(array, np.ndarray) or array.dtype not in (np.dtype('float16'), np.dtype('float32')):
            raise LegacyMEImportError(f'{filename}: {name} must be a float16/float32 NumPy tensor')
        if not np.isfinite(array).all():
            raise LegacyMEImportError(f'{filename}: nonfinite TF tensor {name}')
        shape = tuple(parameter.shape)
        if layer.__class__.__name__ == 'Conv2D' and parameter_name == 'weight':
            expected_shape = (shape[2], shape[3], shape[1], shape[0])
            if tuple(array.shape) != expected_shape:
                raise LegacyMEImportError(f'{filename}: {name} TF shape {array.shape} != {expected_shape}')
            array = array.transpose(3, 2, 0, 1)  # HWIO -> OIHW.
        elif layer.__class__.__name__ in ('Conv2D', 'Dense') and parameter_name in ('weight', 'bias'):
            if tuple(array.shape) != shape:
                raise LegacyMEImportError(f'{filename}: {name} TF shape {array.shape} != {shape}')
            # DFL Dense already stores [input, output]; do not transpose it.
        else:
            raise LegacyMEImportError(f'{filename}: unsupported TF layer parameter {name}')
        tensor = torch.from_numpy(np.array(array, dtype=np.float32, order='C', copy=True))
        if tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all()):
            raise LegacyMEImportError(f'{filename}: invalid converted FP32 tensor {name}')
        prepared.append((parameter, tensor))
    return prepared


def import_tf_me_weights(source, name, config, output):
    """Validate every local TF network tensor, then atomically save native ME.

    ``name`` is the complete old saved-model prefix, e.g. ``my_model_ME``.
    The output contains provenance metadata in the same atomic checkpoint.
    This is a weights migration with a fresh optimizer, never an exact legacy
    training resume. No TensorFlow installation or execution is involved.
    """
    if isinstance(config, dict):
        config = MEConfig.from_dict(config)
    if not isinstance(config, MEConfig):
        raise LegacyMEImportError('config must be an explicit MEConfig or config dictionary')
    component_names = legacy_me_components(config.archi)
    if any(getattr(config, key, 0) for key in ('gan_power', 'true_face_power')):
        raise LegacyMEImportError('Import with GAN/true-face powers disabled; legacy discriminator and optimizer state are not imported')
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or any(character in name for character in '/\\\x00:*?"<>|')):
        raise LegacyMEImportError('name must be the exact legacy saved-model filename prefix')
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise LegacyMEImportError(f'TF weights source directory does not exist: {source}')
    files = {component: source / f'{name}_{component}.npy' for component in component_names}
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise LegacyMEImportError('Missing TF ME component files: ' + ', '.join(missing))
    output = Path(output).expanduser().resolve()
    if output in (path.resolve() for path in files.values()):
        raise LegacyMEImportError('Native output must not replace a source weights file')
    from .engine import MEEngine
    # Do not disturb a caller's CPU RNG. The imported checkpoint has its own
    # deterministic fresh training RNG, saved before this context is restored.
    with torch.random.fork_rng(devices=[]):
        engine = MEEngine(config, device='cpu', seed=0)
        prepared, reports = [], []
        for component in component_names:
            model, path = getattr(engine.network, component), files[component]
            named = list(_named_parameters(model))
            mapping, container, digest = _read_mapping(path, len(named))
            tensors = _prepare_component(model, mapping, path.name)
            prepared.extend(tensors)
            reports.append(dict(component=component, file=path.name, sha256=digest,
                                container=container, tensors=len(tensors)))
        if {id(parameter) for parameter, _ in prepared} != {id(parameter) for parameter in engine.network.parameters()}:
            raise LegacyMEImportError('The legacy components do not cover the complete native ME network')
        with torch.no_grad():
            for parameter, tensor in prepared:
                parameter.copy_(tensor)
        engine.optimizer.validate_state(iteration=0)
        metadata = dict(format='tf-me-network-import', version=1,
                        source=str(source), name=name, source_architecture=config.archi,
                        source_configuration='explicit-caller-config', components=reports,
                        network_only=True, optimizer_reset=True, iteration_reset=True,
                        legacy_optimizer_imported=False, legacy_discriminators_imported=False,
                        legacy_rng_imported=False, legacy_data_state_imported=False)
        engine.save(output, import_metadata=metadata)
    return dict(path=str(output), **metadata)
