"""Frozen-TAE LIAE inference through the existing DFL masked merger.

The pixel/mask interface matches ME. Its network and ``tae.pt`` checkpoint
format are deliberately separate from ME/SAEHD weights.
"""
from pathlib import Path

import numpy as np

from .data import FACE_TYPES
from .web_bridge import validate_model_name


class TAEInferenceModel:
    def __init__(self, saved_models_path, force_model_name=None, force_gpu_idxs=None,
                 cpu_only=False, is_training=False, vae_weights=None, **kwargs):
        if is_training:
            raise ValueError('TAE training is provided by tae.py train')
        root = Path(saved_models_path)
        if not root.is_dir():
            raise FileNotFoundError(f'TAE model directory does not exist: {root}')
        if force_model_name:
            self.name = validate_model_name(force_model_name)
            self.directory = root / self.name
        elif (root / 'tae.pt').is_file():
            self.name, self.directory = root.name, root
        else:
            candidates = sorted(path for path in root.iterdir()
                                if path.is_dir() and (path / 'tae.pt').is_file())
            if len(candidates) != 1:
                raise ValueError('Select an existing TAE model explicitly')
            self.directory, self.name = candidates[0], candidates[0].name
        checkpoint = self.directory / 'tae.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'TAE checkpoint does not exist: {checkpoint}')
        if cpu_only:
            device = 'cpu'
        else:
            indexes = force_gpu_idxs
            if isinstance(indexes, str):
                indexes = [int(value.strip()) for value in indexes.split(',') if value.strip()]
            if indexes and (len(indexes) != 1 or type(indexes[0]) is not int or indexes[0] < 0):
                raise ValueError('TAE requires one nonnegative GPU index')
            device = f'cuda:{indexes[0]}' if indexes else 'cuda'
        from tae.engine import TAEEngine
        self.engine = TAEEngine.load(checkpoint, device=device, weights=vae_weights, training=False)
        from core.leras import nn
        nn.initialize(nn.DeviceConfig.CPU() if self.engine.device.type == 'cpu'
                      else nn.DeviceConfig.GPUIndexes([self.engine.device.index or 0]))

    def get_MergerConfig(self):
        from merger import MergerConfigMasked
        resolution = self.engine.config.resolution

        def predictor(face):
            value = np.asarray(face)
            if (value.shape != (resolution, resolution, 3)
                    or not np.issubdtype(value.dtype, np.floating)
                    or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1):
                raise ValueError('Expected finite aligned BGR HWC floats in [0,1] at TAE resolution')
            batch = np.ascontiguousarray(value.transpose(2, 0, 1)[None], dtype=np.float32)
            prediction = self.engine.predict(batch)
            shapes = ((1, 3, resolution, resolution), (1, 1, resolution, resolution),
                      (1, 1, resolution, resolution))
            if len(prediction) != 3 or any(np.shape(item) != shape or not np.isfinite(item).all()
                                           for item, shape in zip(prediction, shapes)):
                raise ValueError('TAE returned invalid BGR image or merger masks')
            swap, swap_mask, dst_mask = prediction
            return tuple(np.ascontiguousarray(np.clip(item, 0, 1), dtype=np.float32) for item in
                         (swap[0].transpose(1, 2, 0), swap_mask[0, 0], dst_mask[0, 0]))

        return predictor, (resolution, resolution, 3), MergerConfigMasked(
            face_type=FACE_TYPES[self.engine.config.face_type], default_mode='overlay')

    def get_iter(self):
        return self.engine.iteration

    def get_model_name(self):
        return self.name + '_TAE'

    def get_strpath_storage_for_file(self, filename):
        return str(self.directory / filename)

    def finalize(self):
        pass
