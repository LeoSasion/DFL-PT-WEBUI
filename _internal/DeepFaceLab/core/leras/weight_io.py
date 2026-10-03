"""Strict loading for Torch helper networks and trusted DFL auxiliary weights.

The distributed S3FD, FAN and FaceEnhancer ``.npy`` files are pickled named
NumPy tensors, not NumPy archives. They do not contain TensorFlow programs.
Only complete, finite, shape-compatible mappings are applied to a network.
"""

import numpy as np
import torch


def _named_weights(model, prefix=""):
    if hasattr(model, "build"):
        model.build()
    for layer in model.layers:
        name = f"{prefix}{layer.name}"
        if hasattr(layer, "layers"):
            yield from _named_weights(layer, name + "/")
        else:
            for parameter_name, parameter in layer.named_parameters(recurse=False):
                yield f"{name}/{parameter_name}", layer, parameter_name, parameter


def _array(value):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Weights must be finite numeric tensors")
    return array


def _legacy_array(value, layer, parameter_name, parameter):
    array = _array(value)
    shape = tuple(parameter.shape)
    if parameter_name == "weight" and layer.__class__.__name__ in ("Conv2D", "Conv2DTranspose"):
        if array.ndim != 4:
            raise ValueError("A convolution kernel must have four dimensions")
        # Conv HWIO -> OIHW; transposed-conv HWOI -> IOHW.
        array = array.transpose(3, 2, 0, 1)
    elif parameter.ndim <= 1 and array.size == parameter.numel():
        # Old DFL stores some per-channel values as 1 x 1 x 1 x C.
        array = array.reshape(shape)
    if tuple(array.shape) != shape:
        raise ValueError(f"Weight shape mismatch: {array.shape} != {shape}")
    return np.ascontiguousarray(array)


def load_weight_mapping(model, mapping):
    """Validate the whole mapping before updating any parameter.

    ``param_N`` mappings are the Torch helper format. Named mappings use the
    exact legacy DFL layer paths, with optional ``:0`` and model-name prefix.
    Missing tensors must never turn a randomly initialized model into a
    seemingly loaded model.
    """
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("The weights file does not contain a tensor mapping")
    parameters = model.get_weights()
    if not parameters:
        raise ValueError("The network has no weights")
    prepared = []
    if any(str(key).startswith("param_") for key in mapping):
        expected = {f"param_{index}" for index in range(len(parameters))}
        if set(mapping) != expected:
            raise ValueError("Torch weight tensor names/count do not match the network")
        for index, parameter in enumerate(parameters):
            array = _array(mapping[f"param_{index}"])
            if tuple(array.shape) != tuple(parameter.shape):
                raise ValueError(f"Weight shape mismatch for param_{index}")
            prepared.append((parameter, np.ascontiguousarray(array)))
    else:
        if not hasattr(model, "layers"):
            raise ValueError("This object cannot load named DFL network weights")
        root = getattr(model, "name", None)
        normalized = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise ValueError("Named weight keys must be strings")
            key = key.removesuffix(":0")
            if root and key.startswith(root + "/"):
                key = key[len(root) + 1:]
            if key in normalized:
                raise ValueError("Duplicate normalized weight name")
            normalized[key] = value
        named = list(_named_weights(model))
        expected = {name for name, _, _, _ in named}
        if set(normalized) != expected or len(named) != len(parameters):
            missing = sorted(expected - set(normalized))
            unexpected = sorted(set(normalized) - expected)
            raise ValueError(f"DFL weight names mismatch (missing={missing[:3]}, unexpected={unexpected[:3]})")
        for name, layer, parameter_name, parameter in named:
            prepared.append((parameter, _legacy_array(normalized[name], layer, parameter_name, parameter)))
    # Allocate/conversion happens before mutation, including GPU placement.
    tensors = [(parameter, torch.from_numpy(array).to(device=parameter.device, dtype=parameter.dtype).reshape(parameter.shape))
               for parameter, array in prepared]
    with torch.no_grad():
        for parameter, tensor in tensors:
            parameter.copy_(tensor)
    return len(tensors)
