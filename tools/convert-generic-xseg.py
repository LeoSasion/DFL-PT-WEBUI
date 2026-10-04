"""Convert a SHA-pinned DFL XSeg tensor mapping for the PyTorch helper API.

This only copies inference tensors. It never trains, runs TensorFlow, or imports
an optimizer/training state. The original input and converted output hashes
belong in release/generic-xseg.json.
"""

import argparse
import hashlib
import io
import pickle
import sys
from pathlib import Path

import numpy as np


class TensorUnpickler(pickle.Unpickler):
    """Allow NumPy tensor construction only, never arbitrary pickle globals."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
        }
        try:
            return allowed[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; conversion never replaces an existing file")
    raw = args.input.read_bytes()
    original_sha = hashlib.sha256(raw).hexdigest()
    if original_sha != args.input_sha256.lower():
        parser.error("Original weight SHA-256 does not match the pinned source")
    mapping = TensorUnpickler(io.BytesIO(raw)).load()
    if not isinstance(mapping, dict) or not mapping:
        parser.error("Expected a non-empty tensor mapping")
    for name, value in mapping.items():
        if not isinstance(name, str) or not isinstance(value, np.ndarray):
            parser.error("Expected named NumPy tensors only")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            parser.error(f"Invalid numeric tensor: {name}")
    backend = Path(__file__).resolve().parents[1] / "_internal" / "DeepFaceLab"
    sys.path.insert(0, str(backend))
    from core.leras import nn
    from core.leras.weight_io import load_weight_mapping

    nn.initialize(nn.DeviceConfig.CPU(), data_format="NCHW")
    model = nn.XSeg(3, 32, 1, name="XSeg")
    count = load_weight_mapping(model, mapping)
    converted = {f"param_{index}": tensor.detach().cpu().numpy().copy()
                 for index, tensor in enumerate(model.get_weights())}
    payload = pickle.dumps(converted, protocol=4)
    # A second strict load proves the exported tensor count, shapes and values.
    saved = TensorUnpickler(io.BytesIO(payload)).load()
    load_weight_mapping(model, saved)
    for index, tensor in enumerate(model.get_weights()):
        if not np.array_equal(converted[f"param_{index}"], tensor.detach().cpu().numpy()):
            raise ValueError("Converted tensor changed during serialization")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(f"Converted {count} complete XSeg tensors; no training or inference performed")
    print(f"Original SHA-256: {original_sha}")
    print(f"Converted SHA-256: {hashlib.sha256(payload).hexdigest()}")
    print(f"Converted bytes: {len(payload)}")


if __name__ == "__main__":
    main()
