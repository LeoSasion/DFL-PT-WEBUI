"""Deterministic CPU photometric transforms for the ME input pipeline.

The distortion ranges and FS/CC/LAB transforms follow the local ME
SampleProcessor. Every random draw belongs to a supplied sample seed;
neither NumPy's nor Python's global RNG is changed.
"""
import random

import cv2
import numpy as np
from scipy.stats import special_ortho_group

from core.imagelib.blursharpen import LinearMotionBlur
from core.imagelib.color_transfer import linear_color_transfer


CT_MODES = ('none', 'rct', 'lct', 'mkl', 'idt', 'sot', 'mix', 'fs-aug', 'cc-aug')
REFERENCE_CT_MODES = frozenset(('rct', 'lct', 'mkl', 'idt', 'sot', 'mix'))


def _finite_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError('Color augmentation produced an invalid BGR image')
    return np.clip(image, 0, 1)


def _fs_cc_color(image, seed, strong):
    face = np.clip(image * 255, 0, 255).astype(np.uint8)
    # The original functions reset their local seed for CLAHE and LAB.
    rnd = random.Random(int(seed))
    if rnd.random() <= .5:
        grid_size = 1 + int(rnd.random() * 2)
        clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(grid_size, grid_size))
        for channel in range(3):
            face[..., channel] = clahe.apply(face[..., channel])
    rnd = random.Random(int(seed))
    amounts = (.45, .20, .20) if strong else (.30, .08, .08)
    adjustments = [rnd.random() * amount * 2 - amount for amount in amounts]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.
    for channel, adjustment in enumerate(adjustments):
        value = lab[..., channel]
        lab[..., channel] = ((1 - value) * adjustment + value if adjustment >= 0
                            else value * (1 + adjustment))
    face = cv2.cvtColor(np.clip(lab * 255, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if strong:
        factor = np.random.RandomState(int(seed)).uniform(.7, 1.1)
        face = np.clip(face * factor, 0, 255).astype(np.uint8)
        mean = face.mean(axis=(0, 1), keepdims=True)
        factor = np.random.RandomState(int(seed)).uniform(.7, 1.1)
        face = np.clip((face - mean) * factor + mean, 0, 255).astype(np.uint8)
        factor = np.random.RandomState(int(seed)).uniform(.6, 1.2)
        face = np.clip(face * factor, 0, 255).astype(np.uint8)
    return face.astype(np.float32) / 255.


def _sliced_transfer(source, reference, rnd):
    """SOT with a local RNG, matching the existing CPU color-transfer solver."""
    height, width, channels = source.shape
    result = source.copy()
    for _ in range(10):
        advect = np.zeros((height * width, channels), dtype=np.float32)
        for _ in range(5):
            direction = rnd.normal(size=channels).astype(np.float32)
            direction /= np.linalg.norm(direction)
            src_projection = (result * direction).sum(axis=-1).reshape(-1)
            dst_projection = (reference * direction).sum(axis=-1).reshape(-1)
            src_order, dst_order = np.argsort(src_projection), np.argsort(dst_projection)
            difference = dst_projection[dst_order] - src_projection[src_order]
            advect[src_order] += difference[:, None] * direction
        result += advect.reshape(source.shape) / 5
    return source + cv2.bilateralFilter(result - source, 0, 5., 16.)


def _iterative_transfer(source, reference, rnd):
    """IDT with explicit rotations so worker scheduling never affects colors."""
    original_shape = source.shape
    current = source.reshape(-1, 3).T.copy()
    target = reference.reshape(-1, 3).T
    for _ in range(20):
        rotation = special_ortho_group.rvs(3, random_state=rnd).astype(np.float32)
        projected, projected_target = rotation @ current, rotation @ target
        matched = np.empty_like(current)
        for channel in range(3):
            # A collapsed distribution has one value, so histogram interpolation
            # has no inverse. Translate its mean instead of extrapolating to 256.
            if np.ptp(projected[channel]) < 1e-6 or np.ptp(projected_target[channel]) < 1e-6:
                matched[channel] = (projected[channel] - projected[channel].mean()
                                    + projected_target[channel].mean())
                continue
            lower = min(projected[channel].min(), projected_target[channel].min())
            upper = max(projected[channel].max(), projected_target[channel].max())
            src_hist, edges = np.histogram(projected[channel], 256, range=(lower, upper))
            dst_hist, _ = np.histogram(projected_target[channel], 256, range=(lower, upper))
            src_cdf, dst_cdf = src_hist.cumsum().astype(np.float32), dst_hist.cumsum().astype(np.float32)
            src_cdf /= src_cdf[-1]
            dst_cdf /= dst_cdf[-1]
            mapped = np.interp(src_cdf, dst_cdf, edges[1:])
            matched[channel] = np.interp(projected[channel], edges[1:], mapped,
                                         left=mapped[0], right=mapped[-1])
        current += np.linalg.solve(rotation, matched - projected) / 20
    return current.T.reshape(original_shape)


def _reinhard_transfer(source, reference):
    source_lab = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_BGR2LAB).astype(np.float64)
    target_lab = cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_BGR2LAB).astype(np.float64)
    source_mean, target_mean = source_lab.mean((0, 1)), target_lab.mean((0, 1))
    ratio = target_lab.std((0, 1)) / np.maximum(source_lab.std((0, 1)), 1e-6)
    result = (source_lab - source_mean) * ratio + target_mean
    result[..., 0] = np.clip(result[..., 0], 0, 100)
    result[..., 1:] = np.clip(result[..., 1:], -127, 127)
    return cv2.cvtColor(result.astype(np.float32), cv2.COLOR_LAB2BGR)


def _monge_transfer(source, reference):
    # Float64 centering avoids amplifying float32 mean rounding in flat crops.
    current, target = source.reshape(-1, 3).astype(np.float64), reference.reshape(-1, 3).astype(np.float64)
    current_mean, target_mean = current.mean(0), target.mean(0)
    current, target = current - current_mean, target - target_mean
    covariance = current.T @ current / max(1, len(current) - 1)
    target_covariance = target.T @ target / max(1, len(target) - 1)
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(np.maximum(values, 1e-6))) @ vectors.T
    inverse_root = (vectors / np.sqrt(np.maximum(values, 1e-6))) @ vectors.T
    middle = root @ target_covariance @ root
    values, vectors = np.linalg.eigh(middle)
    middle_root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    transform = inverse_root @ middle_root @ inverse_root
    return (current @ transform + target_mean).reshape(source.shape).astype(np.float32)


def _mixed_transfer(source, reference, rnd):
    """Legacy lightness/chroma mix, with SOT driven by the sample RNG."""
    source_lab = cv2.cvtColor(np.clip(source * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(np.clip(reference * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    lightness = np.clip(linear_color_transfer(source_lab[..., :1].astype(np.float32) / 255.,
                                             target_lab[..., :1].astype(np.float32) / 255.)[..., 0]
                        * 255, 0, 255).astype(np.uint8)
    source_lab[..., 0] = target_lab[..., 0] = 100
    source_chroma = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    target_chroma = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    matched = np.clip(_sliced_transfer(source_chroma, target_chroma, rnd), 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(matched, cv2.COLOR_BGR2LAB)
    matched[..., 0] = lightness
    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.


def transfer_color(image, mode, reference, seed):
    if mode not in CT_MODES:
        raise ValueError(f'Unsupported ME ct_mode: {mode}')
    if mode == 'none':
        return image
    if mode in ('fs-aug', 'cc-aug'):
        return _finite_image(_fs_cc_color(image, seed, mode == 'cc-aug'))
    if reference is None:
        raise ValueError(f'ME ct_mode {mode} requires a DST reference dataset; use PairDataLoader')
    reference = _finite_image(reference)
    if reference.shape != image.shape:
        reference = cv2.resize(reference, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    rnd = np.random.RandomState(int(seed))
    if mode == 'sot':
        result = _sliced_transfer(image, reference, rnd)
    elif mode == 'idt':
        result = _iterative_transfer(image, reference, rnd)
    elif mode == 'rct':
        result = _reinhard_transfer(image, reference)
    elif mode == 'mkl':
        result = _monge_transfer(image, reference)
    elif mode == 'mix':
        result = _mixed_transfer(image, reference, rnd)
    else:
        result = linear_color_transfer(image, reference)
    return _finite_image(result)


def distort_input(image, config, seed):
    """Corrupt the encoder input, never the reconstruction target or masks."""
    flags = {name: bool(getattr(config, f'random_{name}', False))
             for name in ('blur', 'noise', 'jpeg', 'downsample')}
    if not any(flags.values()):
        return image
    rnd = np.random.RandomState(int(seed))
    order = ['blur', 'noise', 'jpeg', 'downsample']
    rnd.shuffle(order)
    image = image.copy()
    height, width = image.shape[:2]
    for distortion in order:
        if not flags[distortion]:
            continue
        if distortion == 'blur':
            if rnd.randint(2) == 0:
                image = LinearMotionBlur(image, int(rnd.randint(10, 20)), 360 * rnd.random())
            else:
                sigma = 5 * rnd.random() + 3
                size = int((2.9 if sigma < 5 else 2.6) * sigma)
                size += size % 2 == 0
                image = cv2.GaussianBlur(image, (size, size), sigma)
        elif distortion == 'noise':
            noise_type = int(rnd.randint(3))
            scale = 20 * rnd.random() + 20
            if noise_type == 0:
                noise = rnd.normal(scale=scale, size=image.shape)
            elif noise_type == 1:
                noise = rnd.laplace(scale=scale, size=image.shape)
            else:
                noise = rnd.poisson(lam=15 * rnd.random() + 15, size=image.shape)
            image += noise.astype(np.float32) / 255.
        elif distortion == 'jpeg':
            ok, encoded = cv2.imencode('.jpg', np.clip(image * 255, 0, 255).astype(np.uint8),
                                       [cv2.IMWRITE_JPEG_QUALITY, int(rnd.randint(50, 85))])
            if not ok:
                raise IOError('ME random JPEG encoding failed')
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise IOError('ME random JPEG decoding failed')
            image = decoded.astype(np.float32) / 255.
        else:
            size = int(rnd.randint(max(1, int(.125 * min(height, width))),
                                   max(2, int(.25 * min(height, width)))))
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return _finite_image(image)


def rotate_lab_color(image, seed):
    lab = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2LAB)
    rotation = special_ortho_group.rvs(2, random_state=np.random.RandomState(int(seed)))
    lab[..., 1:] = lab[..., 1:] @ rotation
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    lab[..., 1:] = np.clip(lab[..., 1:], -127, 127)
    return _finite_image(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
