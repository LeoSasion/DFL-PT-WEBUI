"""Conservative geometry-only tracks on verified rational source timestamps.

These are associations, not identity recognition. Ambiguity terminates tracks.
No landmark point is filtered or overwritten.
"""
import json
import re
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .quality_merge import sha256


def bind_timeline(paths):
    marker = Path(paths[0]).parent / 'frames.timeline.json'
    if not marker.is_file():
        return [None] * len(paths), {'status': 'missing-timeline', 'motionDisabled': True}
    payload = json.loads(marker.read_text(encoding='utf-8'))
    if payload.get('schemaVersion') != 1 or payload.get('kind') != 'extracted-frames':
        raise ValueError('Unsupported merge source timeline')
    source = payload.get('source', {})
    if not isinstance(source, dict) or not isinstance(source.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', source['sha256']):
        raise ValueError('Merge timeline must record its original source SHA256')
    frames = payload.get('frames', [])
    if not isinstance(frames, list):
        raise ValueError('Invalid merge timeline entries')
    by_name = {frame['file']: frame for frame in frames}
    if len(by_name) != len(frames):
        raise ValueError('Duplicate source timeline filenames')
    bound, previous = [], None
    for path in paths:
        path = Path(path)
        frame = by_name.get(path.name)
        if not frame or sha256(path) != frame.get('imageSha256'):
            raise ValueError('Merge source frame differs from its timing manifest: ' + path.name)
        pts, tb, index = frame.get('pts'), frame.get('timeBase'), frame.get('sourceFrameIndex')
        if type(pts) is not int or not isinstance(tb, list) or len(tb) != 2 or any(type(x) is not int or x <= 0 for x in tb) or type(index) is not int or index < 0:
            raise ValueError('Original integer PTS, positive time base and source index required')
        timestamp = Fraction(pts) * Fraction(*tb)
        if previous is not None and timestamp <= previous:
            raise ValueError('Merge source timestamps must strictly increase')
        previous = timestamp
        bound.append({**frame, 'seconds': float(timestamp)})
    return bound, {'status': 'bound', 'path': str(marker), 'sha256': sha256(marker),
                   'sourceSha256': payload.get('source', {}).get('sha256'), 'motionDisabled': False}


def associate_geometry(records):
    """Each record has seconds/index/segment, optional cut and per-face geometry."""
    tracks, next_id, previous = {}, 0, None
    times = [record['seconds'] for record in records]
    if any(not np.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Finite strictly increasing source times required')
    median_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0
    for record in records:
        faces = record['faces']
        for face in faces:
            face['trackId'], face['breakReason'] = None, 'initial'
        boundary = None
        if previous is not None:
            dt = record['seconds'] - previous['seconds']
            if record.get('cutBefore') or record.get('shotId') != previous.get('shotId') or record.get('segmentIndex') != previous.get('segmentIndex'):
                boundary = 'shot-change'
            elif record['sourceFrameIndex'] != previous['sourceFrameIndex'] + 1:
                boundary = 'source-index-gap'
            elif median_dt > 0 and dt > 4 * median_dt:
                boundary = 'pts-gap'
            elif not previous['faces'] or not faces:
                boundary = 'missing-face'
        if previous is not None and boundary is None:
            old = previous['faces']
            eligible = np.zeros((len(old), len(faces)), bool)
            for i, a in enumerate(old):
                for j, b in enumerate(faces):
                    scale_ratio = b['scale'] / a['scale']
                    distance = np.linalg.norm(np.asarray(b['center']) - a['center']) / max(a['scale'], b['scale'])
                    eligible[i, j] = .7 <= scale_ratio <= 1.43 and distance <= .35
            for j, face in enumerate(faces):
                incoming = np.flatnonzero(eligible[:, j])
                if len(incoming) == 1 and np.count_nonzero(eligible[incoming[0]]) == 1 and old[incoming[0]]['trackId'] is not None:
                    face['trackId'] = old[incoming[0]]['trackId']
                    face['breakReason'] = None
                elif len(incoming) > 0:
                    face['breakReason'] = 'ambiguous-geometry'
                else:
                    face['breakReason'] = 'geometry-discontinuity'
        for face in faces:
            # Duplicate/current overlaps are ambiguous even on the initial frame.
            duplicates = sum(np.linalg.norm(np.asarray(face['center']) - other['center']) < .05 * max(face['scale'], other['scale'])
                             and .9 < face['scale'] / other['scale'] < 1.11 for other in faces)
            if duplicates > 1:
                face['trackId'], face['breakReason'] = None, 'duplicate-geometry'
                continue
            if face['trackId'] is None:
                face['trackId'] = next_id
                next_id += 1
                if boundary:
                    face['breakReason'] = boundary
            tracks.setdefault(face['trackId'], []).append((record, face))
        previous = record
    return tracks


def stabilize_geometry(records, strength=0):
    if not 0 <= strength <= 100:
        raise ValueError('Geometry strength must be 0..100')
    tracks = associate_geometry(records)
    for record in records:
        for face in record['faces']:
            face.update({'rawCenter': list(face['center']), 'rawScale': face['scale'],
                         'usedCenter': list(face['center']), 'usedScale': face['scale'],
                         'contributors': [], 'fallbackReason': 'ambiguous-geometry',
                         'motionPower': 0., 'motionDegrees': 0., 'velocityPixelsPerSecond': [0., 0.],
                         'motionDeltaSeconds': None, 'motionExposureSeconds': None})
    for track in tracks.values():
        for i, (record, face) in enumerate(track):
            center, scale = np.asarray(face['center'], np.float64), float(face['scale'])
            face['rawCenter'], face['rawScale'] = center.tolist(), scale
            face['usedCenter'], face['usedScale'] = center.tolist(), scale
            face['contributors'] = []
            face['fallbackReason'] = 'disabled' if strength == 0 else 'insufficient-neighbors'
            window = track[max(0, i - 2):i + 3]
            if strength and len(window) >= 3:
                dt = np.array([r['seconds'] - record['seconds'] for r, _ in window])
                # Fit at actual uneven PTS; exact constant velocity/log-scale is preserved.
                design = np.column_stack((np.ones(len(dt)), dt))
                values = np.array([f['center'] + [np.log(f['scale'])] for _, f in window])
                fit = np.linalg.lstsq(design, values, rcond=None)[0][0]
                delta = fit[:2] - center
                limit = scale * .025
                norm = np.linalg.norm(delta)
                if norm > limit:
                    delta *= limit / norm
                factor = strength / 100.
                face['usedCenter'] = (center + factor * delta).tolist()
                face['usedScale'] = float(scale * np.exp(factor * np.clip(fit[2] - np.log(scale), -.025, .025)))
                face['contributors'] = [r['sourceFrameIndex'] for r, _ in window]
                face['fallbackReason'] = None
            face['velocityPixelsPerSecond'] = [0., 0.]
            face['motionDeltaSeconds'], face['motionExposureSeconds'] = None, None
            neighbors = track[max(0, i - 1):i + 2]
            if len(neighbors) >= 2:
                before, after = neighbors[0], neighbors[-1]
                duration = after[0]['seconds'] - before[0]['seconds']
                velocity = (np.asarray(after[1]['center']) - before[1]['center']) / duration
                face['velocityPixelsPerSecond'] = velocity.tolist()
                face['motionDeltaSeconds'] = duration
                duration_pts = record.get('durationPts')
                exposure = float(Fraction(duration_pts) * Fraction(*record['timeBase'])) if type(duration_pts) is int and duration_pts > 0 else duration / (len(neighbors) - 1)
                face['motionExposureSeconds'] = exposure
                face['motionPower'] = float(np.linalg.norm(velocity) * exposure)
                face['motionDegrees'] = float(-np.degrees(np.arctan2(velocity[1], velocity[0])))
            else:
                face['motionPower'], face['motionDegrees'] = 0., 0.
    return records


def adjusted_affine(matrix, geometry):
    if not geometry or geometry['usedCenter'] == geometry['rawCenter'] and geometry['usedScale'] == geometry['rawScale']:
        return matrix
    ratio = geometry['rawScale'] / geometry['usedScale']
    translation = np.asarray(geometry['rawCenter']) - ratio * np.asarray(geometry['usedCenter'])
    correction = np.array([[ratio, 0, translation[0]], [0, ratio, translation[1]], [0, 0, 1.]])
    return (np.vstack((matrix, [0, 0, 1])) @ correction)[:2].astype(np.float32)


def hard_cut_thumbnail(previous, current):
    """Conservative supplemental hard-cut guard, recorded rather than claimed GT."""
    a, b = cv2.resize(previous, (48, 48)).astype(np.float32) / 255., cv2.resize(current, (48, 48)).astype(np.float32) / 255.
    return float(np.mean(np.abs(a - b))) > .30
