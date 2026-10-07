"""Interpretable face-detail proxies and metadata coverage, without a model.

Scores compare detail at one fixed 256-pixel scale. They are neither identity
quality nor landmark accuracy. Eye/mouth states are geometry buckets only;
there is no inference of occlusion or expression ground truth.
"""
import math

import cv2
import numpy as np


METRICS = ('foreground_tenengrad', 'foreground_brenner')
# Locked Astra review found identical ordering in the two candidates on the
# limited controlled corpus. Conventional Sobel gradients break the tie;
# this is not a universal perceptual-quality winner or an accuracy score.
DEFAULT_METRIC = 'foreground_tenengrad'
REFERENCE_SIZE = 256


def _points(landmarks):
    value = np.asarray(landmarks, dtype=np.float64)
    if value.shape != (68, 2) or not np.isfinite(value).all():
        raise ValueError('Face quality needs 68 finite existing landmarks')
    if min(np.ptp(value, axis=0)) <= 1:
        raise ValueError('Face quality needs non-degenerate existing landmarks')
    return value


def _masked_energy(values, support):
    data = values[support]
    if data.size < 64:
        raise ValueError('Face quality foreground support is too small')
    # Limit a few isolated bright edges; the mask is never multiplied into
    # pixels, so its silhouette cannot create artificial sharpness.
    cap = float(np.quantile(data, .98))
    return float(np.minimum(data, cap).mean())


def face_detail_signals(image, landmarks, *, foreground=None):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16):
        raise ValueError('Face quality needs a readable uint8 BGR image')
    points = _points(landmarks)
    height, width = image.shape[:2]
    if foreground is None:
        foreground = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(foreground, cv2.convexHull(np.rint(points).astype(np.int32)), 1)
    else:
        foreground = np.asarray(foreground, dtype=np.float32).squeeze()
        if foreground.shape != (height, width) or not np.isfinite(foreground).all():
            raise ValueError('Face quality foreground must match the image')
        foreground = (foreground >= .5).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (REFERENCE_SIZE, REFERENCE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    support = cv2.resize(foreground, (REFERENCE_SIZE, REFERENCE_SIZE), interpolation=cv2.INTER_NEAREST)
    # Both samples of Brenner and the full Sobel stencil stay inside the
    # foreground. Constant borders, hair and background edges cannot win.
    support = cv2.erode(support, np.ones((7, 7), np.uint8)).astype(bool)
    if np.count_nonzero(support) < 64:
        raise ValueError('Face quality foreground support is too small')
    clean = cv2.medianBlur(gray, 3)
    clean = cv2.GaussianBlur(clean, (0, 0), .9)
    residual = gray - clean
    residual_center = float(np.median(residual[support]))
    noise = float(np.median(np.abs(residual[support] - residual_center)) / .67448975)
    dx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    dy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    tenengrad = _masked_energy(dx * dx + dy * dy, support)
    horizontal = clean[:, 2:] - clean[:, :-2]
    vertical = clean[2:, :] - clean[:-2, :]
    brenner = (_masked_energy(horizontal * horizontal, support[:, 1:-1])
               + _masked_energy(vertical * vertical, support[1:-1, :])) / 2
    # Residual is a high-frequency contamination proxy, which can also
    # include real texture. It is reported explicitly for human review.
    penalty = 1 / (1 + (noise / .025) ** 2)
    scores = {name: float(energy / (energy + .001) * penalty)
              for name, energy in zip(METRICS, (tenengrad, brenner))}
    mean = float(gray[support].mean())
    return {'schemaVersion': 1, 'referenceSize': REFERENCE_SIZE,
            'foregroundPixels': int(np.count_nonzero(support)),
            'rawEnergy': {'foreground_tenengrad': tenengrad, 'foreground_brenner': brenner},
            'highFrequencyResidual': noise, 'noisePenalty': penalty,
            'scores': scores, 'brightness': mean,
            'exposure': float(max(0, 1 - abs(mean - .5) / .5)),
            'policy': 'same-scale-denoised-foreground-proxies; no-identity-or-annotation-accuracy'}


def metadata_coverage(landmarks, pose, *, canvas_size):
    points = _points(landmarks)
    angles = np.asarray(pose, dtype=np.float64)
    if angles.shape != (3,) or not np.isfinite(angles).all() or np.max(np.abs(angles)) > math.pi / 2 + 1e-6:
        raise ValueError('Coverage needs finite bounded existing-landmark pose estimates')
    if not isinstance(canvas_size, (int, float)) or canvas_size < 16:
        raise ValueError('Coverage needs a valid aligned canvas')
    def eye_ratio(start):
        eye = points[start:start + 6]
        denominator = 2 * np.linalg.norm(eye[0] - eye[3])
        if denominator < .5:
            return None
        return float((np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / denominator)
    eyes = [eye_ratio(36), eye_ratio(42)]
    mouth_width = float(np.linalg.norm(points[48] - points[54]))
    mouth = float(np.linalg.norm(points[62] - points[66]) / mouth_width) if mouth_width >= .5 else None
    valid_eyes = [value for value in eyes if value is not None and 0 <= value <= 2]
    eye = float(np.mean(valid_eyes)) if len(valid_eyes) == 2 else None
    mouth = mouth if mouth is not None and 0 <= mouth <= 2 else None
    pitch, yaw, roll = map(float, angles)
    pitch_bin = int(np.clip(math.floor((pitch + math.pi / 2) / math.radians(20)), 0, 8))
    yaw_bin = int(np.clip(math.floor((yaw + math.pi / 2) / math.radians(15)), 0, 11))
    eye_bin = 'unknown' if eye is None else 'closed-like' if eye < .20 else 'open-like'
    mouth_bin = 'unknown' if mouth is None else 'closed-like' if mouth < .12 else 'open-like'
    outside = float(np.any((points < 0) | (points >= canvas_size), axis=1).mean())
    return {'source': 'existing-68-landmarks-geometric-proxies',
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'pitchBin': pitch_bin, 'yawBin': yaw_bin,
            'eyeAspectRatio': eye, 'mouthAspectRatio': mouth,
            'eyeBucket': eye_bin, 'mouthBucket': mouth_bin,
            'landmarksOutsideFraction': outside,
            'bucket': [yaw_bin, pitch_bin, eye_bin, mouth_bin],
            'occlusionAvailable': False, 'annotationAccuracyAvailable': False}


def select_quality_coverage(samples, target_count):
    """Exact-count selection sharing the budget across pose/eye/mouth bins.

    Rows use Sorter indices (path, score, unused, yaw, pitch, evidence).
    Score orders representatives within a bucket. Across buckets the next
    representative maximizes distance to already represented coverage, so
    profiles and closed eyes do not receive an intrinsic quality penalty.
    """
    if type(target_count) is not int or target_count < 1:
        raise ValueError('Quality target count must be a positive integer')
    if not samples:
        return [], []
    buckets = {}
    for sample in samples:
        if len(sample) < 6 or not math.isfinite(float(sample[1])):
            raise ValueError('Quality selection needs finite audited scores')
        key = tuple(sample[5]['coverage']['bucket'])
        buckets.setdefault(key, []).append(sample)
    for rows in buckets.values():
        rows.sort(key=lambda row: (-float(row[1]), str(row[0]).casefold()))
    remaining = set(buckets)
    first = min(remaining, key=lambda key: (-float(buckets[key][0][1]), key))
    order = [first]; remaining.remove(first)
    distances = {key: float('inf') for key in remaining}
    while remaining:
        chosen = order[-1]
        for key in remaining:
            distance = ((key[0] - chosen[0]) / 11) ** 2 + ((key[1] - chosen[1]) / 8) ** 2
            distance += .25 * (key[2] != chosen[2]) + .25 * (key[3] != chosen[3])
            distances[key] = min(distances[key], distance)
        next_key = min(remaining, key=lambda key: (-distances[key], -float(buckets[key][0][1]), key))
        order.append(next_key); remaining.remove(next_key)
    selected = []
    while len(selected) < min(target_count, len(samples)):
        for key in order:
            if buckets[key]:
                row = buckets[key].pop(0)
                row[5]['selectionReason'] = 'coverage-representative' if not any(tuple(x[5]['coverage']['bucket']) == key for x in selected) else 'remaining-bucket-detail-rank'
                selected.append(row)
                if len(selected) == min(target_count, len(samples)):
                    break
    rejected = [row for key in order for row in buckets[key]]
    for row in rejected:
        row[5]['selectionReason'] = 'outside-target-after-coverage-and-detail-ranking'
    return selected, rejected
