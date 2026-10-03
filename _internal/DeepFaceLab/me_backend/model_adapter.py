"""ME checkpoint adapter for the existing interactive merger tool."""
from pathlib import Path
import numpy as np
from .data import FACE_TYPES
from .engine import MEEngine


class MEInferenceModel:
    def __init__(self, saved_models_path, force_model_name=None, force_gpu_idxs=None,
                 cpu_only=False, is_training=False, **kwargs):
        if is_training:
            raise ValueError('ME training is provided by me.py web-train')
        root = Path(saved_models_path)
        if force_model_name:
            from .web_bridge import validate_model_name
            self.name = validate_model_name(force_model_name)
            self.directory = root / self.name
        elif (root / 'me.pt').is_file():
            self.name, self.directory = root.name, root
        else:
            candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / 'me.pt').is_file())
            if len(candidates) != 1:
                raise ValueError('Select an existing ME model explicitly')
            self.directory, self.name = candidates[0], candidates[0].name
        if cpu_only:
            device = 'cpu'
        else:
            indexes = force_gpu_idxs
            if isinstance(indexes, str):
                indexes = [int(value) for value in indexes.split(',') if value.strip()]
            if indexes and len(indexes) != 1:
                raise ValueError('ME currently uses one GPU')
            device = f'cuda:{indexes[0]}' if indexes else 'cuda'
        self.engine = MEEngine.load(self.directory / 'me.pt', device)
        from core.leras import nn
        nn.initialize(nn.DeviceConfig.CPU() if cpu_only else nn.DeviceConfig.GPUIndexes([self.engine.device.index or 0]))

    def get_MergerConfig(self):
        from merger import MergerConfigMasked
        def predictor(face):
            x = np.ascontiguousarray(face.transpose(2, 0, 1)[None])
            swap, src_mask, dst_mask = self.engine.predict(x)
            return swap[0].transpose(1, 2, 0), src_mask[0, 0], dst_mask[0, 0]
        resolution = self.engine.config.resolution
        return predictor, (resolution, resolution, 3), MergerConfigMasked(
            face_type=FACE_TYPES[self.engine.config.face_type], default_mode='overlay')

    def get_iter(self):
        return self.engine.iteration

    def get_model_name(self):
        return self.name + '_ME'

    def get_strpath_storage_for_file(self, filename):
        return str(self.directory / filename)

    def finalize(self):
        pass
