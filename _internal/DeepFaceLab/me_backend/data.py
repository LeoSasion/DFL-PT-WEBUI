"""Aligned loose/packed facesets with reproducible CPU loading and prefetch.

Only consumed batches advance the checkpoint RNG. Workers receive immutable
sample plans, so their scheduling and already-prefetched results do not change
the next batch after save/resume.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import dataclass
import hashlib
import multiprocessing
from pathlib import Path
import cv2
import numpy as np
from DFLIMG import DFLIMG
from facelib import FaceType, LandmarksProcessor
from core.imagelib.warp import gen_warp_params, warp_by_params
from .data_augmentation import (
    CT_MODES, REFERENCE_CT_MODES, distort_input, rotate_lab_color, transfer_color,
)

FACE_TYPES = dict(h=FaceType.HALF, mf=FaceType.MID_FULL, f=FaceType.FULL,
                 wf=FaceType.WHOLE_FACE, head=FaceType.HEAD)


@dataclass(frozen=True)
class _ImageReference:
    path: str
    packed_sample: object = None


def read_aligned(path, config):
    packed = path.packed_sample if isinstance(path, _ImageReference) else (
        path if hasattr(path, 'read_raw_file') else None)
    name = path.path if isinstance(path, _ImageReference) else str(path)
    if packed is None:
        dfl = DFLIMG.load(name)
        if dfl is None or not dfl.has_data():
            raise ValueError(f'{name}: missing DFL aligned metadata')
        image = dfl.get_img()
        landmarks = dfl.get_landmarks()
        source_type = FaceType.fromString(dfl.get_face_type())
        full = dfl.get_xseg_mask()
        eyebrows = dfl.get_eyebrows_expand_mod()
    else:
        payload = packed.read_raw_file()
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        landmarks = packed.landmarks
        source_type = FaceType(packed.face_type)
        full = packed.get_xseg_mask()
        eyebrows = packed.eyebrows_expand_mod
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'{name}: invalid BGR image')
    image = image.astype(np.float32)/255.
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (68,2) or not np.isfinite(landmarks).all():
        raise ValueError(f'{name}: expected finite 68-point landmarks')
    if source_type == FaceType.MARK_ONLY:
        raise ValueError(f'{name}: mark_only is not an aligned crop')
    if full is None:
        full = LandmarksProcessor.get_image_hull_mask(image.shape,landmarks,eyebrows)
    else:
        if not np.isfinite(full).all() or full.ndim not in (2, 3) or (full.ndim == 3 and full.shape[2] != 1):
            raise ValueError(f'{name}: invalid XSeg mask')
        full = cv2.resize(full, image.shape[1::-1], interpolation=cv2.INTER_CUBIC)[...,None]
    full = np.clip(full,0,1)
    eyes = np.clip(LandmarksProcessor.get_image_eye_mask(image.shape,landmarks),0,1)
    mouth = np.clip(LandmarksProcessor.get_image_mouth_mask(image.shape,landmarks),0,1)
    eyes = np.where(eyes>.1,eyes+1,eyes)*(full!=0)
    mouth = np.where(mouth>.1,mouth+2,mouth)*(full!=0)
    encoded = np.where(eyes>1,eyes,full)
    encoded = np.where(mouth>2,mouth,encoded)
    target_type = FACE_TYPES[config.face_type]
    size = config.resolution
    if source_type == target_type:
        def align(x, interpolation, border):
            return cv2.resize(x,(size,size),interpolation=interpolation)
    else:
        matrix = LandmarksProcessor.get_transform_mat(landmarks,size,target_type)
        def align(x, interpolation, border):
            return cv2.warpAffine(x,matrix,(size,size),flags=interpolation,borderMode=border)
    image = np.clip(align(image,cv2.INTER_CUBIC,cv2.BORDER_REPLICATE),0,1)
    full = align(full,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT).reshape(size,size,1)
    encoded = align(encoded,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT).reshape(size,size,1)
    return image, full, encoded


def _load_inventory(directory):
    from samplelib.PackedFaceset import PackedFaceset, _member
    original = Path(directory).resolve()
    archive = original if original.is_file() and original.suffix.lower() in ('.pak', '.zip') else None
    root = archive.parent if archive else original
    if not root.is_dir():
        raise ValueError(f'Aligned dataset directory does not exist: {directory}')
    samples = PackedFaceset.load(root, archive_path=archive)
    if samples is not None:
        if not samples:
            raise ValueError(f'Packed aligned faceset is empty: {directory}')
        archive_path = Path(samples[0]._filename_offset_size[0]).resolve()
        samples = sorted(samples, key=_member)
        references = [_ImageReference(str(root.joinpath(*_member(sample).split('/'))), sample)
                      for sample in samples]
        info = archive_path.stat()
        inventory = [('packed', archive_path.name, info.st_size, info.st_mtime_ns,
                      tuple(_member(sample) for sample in samples))]
    else:
        if archive:
            raise FileNotFoundError(f'Packed aligned faceset not found: {archive}')
        paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg'))
        if not paths:
            raise ValueError(f'No aligned JPG images or faceset PAK/ZIP in {directory}')
        references = [_ImageReference(str(p)) for p in paths]
        # Flat directories retain the earlier checkpoint fingerprint exactly.
        inventory = [(p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    fingerprint = hashlib.sha256(repr(inventory).encode()).hexdigest()
    return root, references, fingerprint


def _yaw(reference):
    sample = reference.packed_sample
    if sample is not None:
        angles = sample.get_pitch_yaw_roll()
    else:
        dfl = DFLIMG.load(reference.path)
        if dfl is None or not dfl.has_data():
            raise ValueError(f'{reference.path}: missing DFL aligned metadata')
        landmarks = np.asarray(dfl.get_landmarks(), dtype=np.float32)
        if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
            raise ValueError(f'{reference.path}: expected finite 68-point landmarks')
        angles = dfl.get_dict().get('pitch_yaw_roll')
        if angles is None:
            angles = LandmarksProcessor.estimate_pitch_yaw_roll(landmarks, size=dfl.get_shape()[1])
    if np.asarray(angles).shape != (3,) or not np.isfinite(angles).all():
        raise ValueError(f'{reference.path}: invalid pitch/yaw/roll metadata')
    return -float(angles[1])


def _initialize_worker():
    cv2.setNumThreads(1)


def _process_sample(reference, config, source, seed, warp_seed, color_reference):
    image, full, encoded = read_aligned(reference, config)
    rnd, warp_rnd = np.random.RandomState(seed), np.random.RandomState(warp_seed)
    flip = config.random_src_flip if source else config.random_dst_flip
    params = gen_warp_params(config.resolution, flip, rotation_range=[-3, 3],
                             scale_range=[-.05, .05], rnd_state=rnd, warp_rnd_state=warp_rnd)
    def warp(value, random_warp=False, mask=False):
        return warp_by_params(params, value, random_warp, True, True, not mask,
                              cv2.INTER_LINEAR if mask else cv2.INTER_CUBIC)
    mode = getattr(config, 'ct_mode', 'none')
    if not source and mode not in ('fs-aug', 'cc-aug'):
        mode = 'none'
    reference_image = read_aligned(color_reference, config)[0] if color_reference is not None else None
    colored = transfer_color(image, mode, reference_image, seed)
    distorted = distort_input(colored, config, seed ^ 0x31415926)
    target_image, input_image = colored, distorted
    if config.random_hsv_power:
        amount = config.random_hsv_power
        hue = rnd.randint(-max(1, int(360 * amount * .5)), max(1, int(360 * amount * .5)) + 1)
        saturation, value = (rnd.random() - .5) * amount, (rnd.random() - .5) * amount
        def hsv(value_image):
            h, s, v = cv2.split(cv2.cvtColor(value_image, cv2.COLOR_BGR2HSV))
            return np.clip(cv2.cvtColor(cv2.merge(((h + hue) % 360,
                np.clip(s + saturation, 0, 1), np.clip(v + value, 0, 1))), cv2.COLOR_HSV2BGR), 0, 1)
        target_image = hsv(target_image)
        # Preserve local ME's destination policy: HSV only changes its target.
        if source:
            input_image = hsv(input_image)
    target = np.clip(warp(target_image), 0, 1)
    warped = np.clip(warp(input_image, config.random_warp), 0, 1)
    if getattr(config, 'random_color', False):
        target, warped = rotate_lab_color(target, seed), rotate_lab_color(warped, seed)
    result = (warped, target, np.clip(warp(full, mask=True), 0, 1), warp(encoded, mask=True))
    if not all(np.isfinite(value).all() for value in result):
        raise ValueError(f'{reference.path}: nonfinite ME batch sample')
    return result


def _stack_batch(samples):
    return tuple(np.ascontiguousarray(np.stack(items).transpose(0, 3, 1, 2), dtype=np.float32)
                 for items in zip(*samples))


class AlignedDataset:
    def __init__(self, directory, config, source, seed):
        self.config, self.source = config, bool(source)
        self.directory, self._references, self.fingerprint = _load_inventory(directory)
        self.paths = [Path(reference.path) for reference in self._references]
        mode = getattr(config, 'ct_mode', 'none')
        if mode not in CT_MODES:
            raise ValueError(f'Unsupported ME ct_mode: {mode}')
        self._requires_reference = self.source and mode in REFERENCE_CT_MODES
        self._color_reference = None
        self._saved_reference_fingerprint = None
        self._workers = getattr(config, 'data_workers', 0)
        if type(self._workers) is not int or not 0 <= self._workers <= 32:
            raise ValueError('data_workers must be an integer between 0 and 32')
        self._executor = None
        self._pending = deque()
        self._scheduled_rng = None
        self.closed = False
        self.rng = np.random.default_rng(seed)
        self.yaw_groups = None
        if getattr(config, 'uniform_yaw', False):
            bins = {}
            boundaries = np.linspace(-1.2, 1.2, 128)
            for index, reference in enumerate(self._references):
                bin_id = int(np.clip(np.searchsorted(boundaries, _yaw(reference), side='right') - 1, 0, 127))
                bins.setdefault(bin_id, []).append(index)
            self.yaw_groups = tuple(tuple(bins[key]) for key in sorted(bins))

    def state_dict(self):
        state = dict(fingerprint=self.fingerprint, rng=copy.deepcopy(self.rng.bit_generator.state))
        if self._requires_reference:
            state['reference_fingerprint'] = (self._color_reference.fingerprint if self._color_reference
                                               else self._saved_reference_fingerprint)
        return state

    def load_state_dict(self, state):
        if state['fingerprint'] != self.fingerprint:
            raise ValueError('Aligned dataset changed since checkpoint; use --reset-data-state intentionally')
        reference_fingerprint = state.get('reference_fingerprint')
        if self._requires_reference and self._color_reference is not None and reference_fingerprint not in (None, self._color_reference.fingerprint):
            raise ValueError('DST color reference dataset changed since checkpoint')
        next_rng = np.random.default_rng(0)
        next_rng.bit_generator.state = copy.deepcopy(state['rng'])
        self._stop_prefetch()
        self.rng = next_rng
        self._saved_reference_fingerprint = reference_fingerprint
        self.closed = False

    def set_color_reference(self, dataset):
        if self._saved_reference_fingerprint not in (None, dataset.fingerprint):
            raise ValueError('DST color reference dataset changed since checkpoint')
        if self._color_reference is not dataset:
            self._stop_prefetch()
            self._color_reference = dataset

    def _plan_batch(self, rng):
        if self._requires_reference and self._color_reference is None:
            raise ValueError('Source color transfer requires a DST reference dataset; use PairDataLoader')
        plan = []
        for _ in range(self.config.batch_size):
            if self.yaw_groups is None:
                index = int(rng.integers(len(self._references)))
            else:
                group = self.yaw_groups[int(rng.integers(len(self.yaw_groups)))]
                index = group[int(rng.integers(len(group)))]
            seed, warp_seed = int(rng.integers(2**31)), int(rng.integers(2**31))
            reference = None
            if self._requires_reference:
                reference = self._color_reference._references[int(rng.integers(len(self._color_reference._references)))]
            plan.append((self._references[index], self.config, self.source, seed, warp_seed, reference))
        return plan

    def _schedule(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self._workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker)
            self._scheduled_rng = np.random.default_rng(0)
            self._scheduled_rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        # At most two batches hold arrays or in-flight work, even for 32 workers.
        while len(self._pending) < 2:
            plan = self._plan_batch(self._scheduled_rng)
            futures = [self._executor.submit(_process_sample, *arguments) for arguments in plan]
            self._pending.append((futures, copy.deepcopy(self._scheduled_rng.bit_generator.state)))

    def batch(self):
        if self.closed:
            raise RuntimeError('AlignedDataset is closed')
        if not self._workers:
            next_rng = np.random.default_rng(0)
            next_rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
            plan = self._plan_batch(next_rng)
            result = _stack_batch([_process_sample(*arguments) for arguments in plan])
            self.rng.bit_generator.state = copy.deepcopy(next_rng.bit_generator.state)
            return result
        try:
            self._schedule()
            futures, next_state = self._pending.popleft()
            result = _stack_batch([future.result() for future in futures])
            self.rng.bit_generator.state = next_state
            self._schedule()
            return result
        except BaseException:
            self.close()
            raise

    @property
    def worker_pids(self):
        processes = getattr(self._executor, '_processes', None) or {}
        return tuple(process.pid for process in processes.values())

    def _stop_prefetch(self):
        for futures, _ in self._pending:
            for future in futures:
                future.cancel()
        self._pending.clear()
        executor, self._executor = self._executor, None
        self._scheduled_rng = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def close(self):
        self.closed = True
        self._stop_prefetch()

    def __enter__(self):
        if self.closed:
            raise RuntimeError('AlignedDataset is closed')
        return self

    def __exit__(self, *_):
        self.close()


class PairDataLoader:
    """Coordinate SRC color references while preserving each side's RNG state."""
    def __init__(self, source_data, destination_data):
        if not source_data.source or destination_data.source:
            raise ValueError('PairDataLoader requires SRC followed by DST datasets')
        if (source_data.config.resolution, source_data.config.batch_size) != (
                destination_data.config.resolution, destination_data.config.batch_size):
            raise ValueError('SRC and DST batch sizes and resolutions must match')
        self.source_data, self.destination_data = source_data, destination_data
        source_data.set_color_reference(destination_data)

    def batch(self):
        source_rng = copy.deepcopy(self.source_data.rng.bit_generator.state)
        destination_rng = copy.deepcopy(self.destination_data.rng.bit_generator.state)
        try:
            return self.source_data.batch(), self.destination_data.batch()
        except BaseException:
            # The trainer did not receive a pair. Save/resume must therefore
            # still point at its first unconsumed batch on both sides.
            try:
                self.close()
            finally:
                self.source_data.rng.bit_generator.state = source_rng
                self.destination_data.rng.bit_generator.state = destination_rng
            raise

    def state_dict(self):
        return dict(src=self.source_data.state_dict(), dst=self.destination_data.state_dict())

    def load_state_dict(self, state):
        self.source_data.load_state_dict(state['src'])
        self.destination_data.load_state_dict(state['dst'])

    def close(self):
        try:
            self.source_data.close()
        finally:
            self.destination_data.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
