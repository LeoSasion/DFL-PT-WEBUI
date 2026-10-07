"""Data-only merger sessions and non-executable legacy session conversion."""
import copy
import hashlib
import math
import pickle
import re
import uuid
from pathlib import Path, PureWindowsPath

import numpy as np

from core.safe_pickle import DataRecord


def _state(value, name):
    if not isinstance(value, DataRecord) or value.type_name != name or not isinstance(value.state, dict):
        raise ValueError('Unsupported legacy merger session record')
    return value.state


def _config(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('Merger session configuration must be data')
    from .MergerConfig import MergerConfigMasked, mode_str_dict
    defaults = MergerConfigMasked().get_config()
    result = {key: raw.get(key, default) for key, default in defaults.items() if key != 'sharpen_dict'}
    integer_bounds = {'face_type': (0, 10), 'mask_mode': (0, 11), 'hist_match_threshold': (0, 255),
                      'erode_mask_modifier': (-400, 400), 'blur_mask_modifier': (0, 400),
                      'motion_blur_power': (0, 100), 'output_face_scale': (-50, 50),
                      'super_resolution_power': (0, 100), 'color_transfer_mode': (0, 10),
                      'image_denoise_power': (0, 500), 'bicubic_degrade_power': (0, 100),
                      'color_degrade_power': (0, 100), 'sharpen_mode': (0, 2), 'blursharpen_amount': (-100, 100),
                      'random_seed': (0, 4294967295), 'geometry_strength': (0, 100), 'blend_width': (0, 20)}
    for key, (minimum, maximum) in integer_bounds.items():
        value = result[key]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or not minimum <= value <= maximum:
            raise ValueError('Invalid session merger parameter: ' + key)
        result[key] = int(value)
    if result['face_type'] not in (0, 1, 2, 4, 10) or result['mode'] not in mode_str_dict or result['default_mode'] not in mode_str_dict:
        raise ValueError('Invalid session face type or mode')
    old_model = raw.get('super_resolution_model')
    if old_model not in ('mambairv2', 'realesrgan-x4plus'):
        if result['super_resolution_power'] > 0:
            raise ValueError('旧 FaceEnhancer/SwinIR 合成会话不可复用，请明确选择 MambaIRv2 或 Real-ESRGAN 增强模型')
        result['super_resolution_model'] = 'mambairv2'
    if result['super_resolution_model'] not in ('mambairv2', 'realesrgan-x4plus'):
        raise ValueError('Invalid session enhancement model')
    if type(result['masked_hist_match']) is not bool or result['geometry_mode'] not in ('off', 'bounded-center-scale') or result['blend_mode'] not in ('legacy', 'distance', 'multiband'):
        raise ValueError('Invalid session merger option')
    return result


def normalize_session(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('frames'), list) or not 1 <= len(payload['frames']) <= 100000:
        raise ValueError('Invalid merger session data')
    legacy = payload.get('sessionSchema') != 2
    frames = []
    for raw in payload['frames']:
        if legacy:
            state = _state(raw, 'merger.InteractiveMergerSubprocessor.InteractiveMergerSubprocessor.Frame')
            info = _state(state.get('frame_info'), 'merger.FrameInfo.FrameInfo')
            cfg = _state(state['cfg'], 'merger.MergerConfig.MergerConfigMasked') if state.get('cfg') is not None else None
            filename = PureWindowsPath(str(info.get('filepath'))).name
            raw = {'file': filename, 'sourceSha256': None, 'alignedIdentities': None,
                   'cfg': cfg, 'isDone': bool(state.get('is_done', False))}
        if not isinstance(raw, dict) or not isinstance(raw.get('file'), str) or Path(raw['file']).name != raw['file'] or '\\' in raw['file'] or '/' in raw['file']:
            raise ValueError('Plain session frame filenames required')
        frames.append({**raw, 'cfg': _config(raw.get('cfg'))})
    size = len(frames)
    indexes = payload.get('frames_idxs'), payload.get('frames_done_idxs')
    if any(not isinstance(array, list) or any(type(i) is not int or not 0 <= i < size for i in array) for array in indexes):
        raise ValueError('Invalid merger session frame indexes')
    combined = indexes[0] + indexes[1]
    if sorted(combined) != list(range(size)):
        raise ValueError('Merger session indexes must partition all frames once')
    iteration = payload.get('model_iter')
    if type(iteration) is not int or iteration < 0:
        raise ValueError('Invalid merger session model iteration')
    return {'sessionSchema': 2, 'frames': frames, 'frames_idxs': indexes[0],
            'frames_done_idxs': indexes[1], 'model_iter': iteration,
            'conversionSource': 'legacy-fixed-class-data-records' if legacy else payload.get('conversionSource')}, legacy


def restore_frames(payload, fresh_frames, current_cfg):
    from .MergerConfig import MergerConfigMasked
    if len(payload['frames']) != len(fresh_frames):
        return None, True
    result, recompute = [], False
    for saved, fresh in zip(payload['frames'], fresh_frames):
        if saved['file'] != fresh.frame_info.filepath.name:
            return None, True
        frame = copy.copy(fresh)
        frame.cfg = MergerConfigMasked(**saved['cfg']) if saved['cfg'] is not None else None
        if frame.cfg is not None:
            # These inputs are prepared once before workers start, so a saved
            # option cannot bypass today's reviewed-mask or geometry preflight.
            if frame.cfg.mask_mode in (10, 11) and current_cfg.mask_mode not in (10, 11):
                raise ValueError('Select reviewed DST mode before resuming this reviewed-mask session')
            if frame.cfg.super_resolution_power > 0 and frame.cfg.super_resolution_model != current_cfg.super_resolution_model:
                raise ValueError('Select the same enhancement model before resuming this session')
            if (frame.cfg.geometry_mode, frame.cfg.geometry_strength) != (current_cfg.geometry_mode, current_cfg.geometry_strength):
                raise ValueError('Select the same geometry candidate/strength before resuming this session')
        same_source = saved.get('sourceSha256') == getattr(fresh.frame_info, 'source_sha256', None) and saved.get('sourceSha256') is not None
        identities = getattr(fresh.frame_info, 'aligned_identities', None)
        # Old packed sessions recorded sha256=None. Equal missing digests do
        # not prove unchanged content, even when member filenames still match.
        complete_identities = isinstance(identities, list) and all(
            isinstance(item, dict) and isinstance(item.get('sha256'), str)
            and re.fullmatch(r'[0-9a-f]{64}', item['sha256']) for item in identities)
        same_aligned = saved.get('alignedIdentities') == identities and complete_identities
        frame.is_done = bool(saved.get('isDone')) and same_source and same_aligned
        recompute |= not same_source or not same_aligned
        frame.is_processing, frame.is_shown, frame.image = False, False, None
        result.append(frame)
    return result, recompute


def encode_session(frames, pending, completed, model_iter, conversion_source=None):
    return {'sessionSchema': 2, 'model_iter': int(model_iter), 'frames_idxs': pending,
            'frames_done_idxs': completed, 'conversionSource': conversion_source,
            'frames': [{'file': frame.frame_info.filepath.name,
                        'sourceSha256': getattr(frame.frame_info, 'source_sha256', None),
                        'alignedIdentities': getattr(frame.frame_info, 'aligned_identities', None),
                        'cfg': _config(frame.cfg.get_config()) if frame.cfg is not None else None,
                        'isDone': bool(frame.is_done)} for frame in frames]}


def save_session(path, payload, preserve_legacy=False):
    path = Path(path)
    if preserve_legacy and path.is_file():
        old = path.read_bytes()
        backup = path.with_name(path.name + '.legacy-' + hashlib.sha256(old).hexdigest()[:12] + '.bak')
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(old)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('xb') as stream:
        stream.write(pickle.dumps(payload, protocol=4))
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)
