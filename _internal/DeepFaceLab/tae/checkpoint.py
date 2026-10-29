"""Atomic trusted-local checkpoints containing the trained decoder and head, but no frozen encoder."""
import hashlib
import os
from pathlib import Path
import random
import torch
from core.file_hash import file_sha256
from core.torch_checkpoint import load_weights_checkpoint

FORMAT = "tae-liae-pytorch"
VERSION = 1


def frozen_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path, *, mmap=False):
    # Inference can leave unused Adam storage on disk. Training retains writable
    # CPU optimizer state and must not pin the file it will later replace on Windows.
    payload = load_weights_checkpoint(path, mmap=mmap)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError("Not an TAE v1 checkpoint; DFL/SAEHD/ME weights cannot be loaded")
    return payload


def rng_state():
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact training resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
