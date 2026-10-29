"""CPU checkpoint loading with a narrow, weights-only mmap fallback."""
import os

import torch


def load_weights_checkpoint(path, *, mmap=False):
    # Torch can map only filesystem paths and modern ZIP storage. File-like
    # sources and native older torch.save streams retain eager loading support.
    mapped = bool(mmap and isinstance(path, (str, os.PathLike)))
    try:
        return torch.load(path, map_location='cpu', weights_only=True, mmap=mapped)
    except RuntimeError as error:
        # Do not retry damaged ZIPs, unsafe objects, or unrelated read errors.
        # In particular, a fallback must never turn off weights_only.
        if not mapped or not str(error).startswith(
                'mmap can only be used with files saved with '):
            raise
        return torch.load(path, map_location='cpu', weights_only=True, mmap=False)
