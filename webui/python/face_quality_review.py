"""Explainable review signals from pixels and existing DFL metadata only."""
import math

import cv2
import numpy as np

# Selected after the locked 12-case Astra review: Tenengrad and Brenner had
# identical ordering; Sobel gradients are the conventional tie-break. Retain
# both candidates and the previous sharpness signal for transparent review.
REVIEW_DETAIL_METRIC = 'foreground_tenengrad'


def _field(dfl_image, method):
    try:
        return getattr(dfl_image, method)()
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _points(value):
    try:
        points = np.asarray(value, dtype=np.float64)
        if points.shape == (68, 2) and np.isfinite(points).all():
            return points
    except (TypeError, ValueError):
        pass
    return None


def quality_review(image, metrics, dfl_image=None, pose_estimator=None):
    height, width = image.shape[:2]
    flags = []
    source_face = None
    alignment = {"available": False, "source": "existing-dfl-metadata", "residualFraction": None}
    pose = {"available": False, "source": "existing-68-landmarks-pnp-estimate", "pitch": None, "yaw": None, "roll": None}
    if dfl_image is not None:
        try:
            rect = np.asarray(_field(dfl_image, "get_source_rect"), dtype=np.float64).reshape(-1)
            if (rect.shape == (4,) and np.isfinite(rect).all() and rect[2] > rect[0] and rect[3] > rect[1]
                    and math.isfinite(float(rect[2]) - float(rect[0]))
                    and math.isfinite(float(rect[3]) - float(rect[1]))):
                source_width, source_height = float(rect[2] - rect[0]), float(rect[3] - rect[1])
                source_face = {"available": True, "source": "existing-dfl-source-rect",
                               "width": source_width, "height": source_height,
                               "minimumDimension": min(source_width, source_height)}
                if source_face["minimumDimension"] < 64:
                    flags.append("small_source_face")
                if min(width, height) / source_face["minimumDimension"] > 4:
                    flags.append("strong_source_upscale")
            elif rect.size and not (rect.size == 1 and np.isnan(rect).all()):
                flags.append("invalid_source_rect")
        except (TypeError, ValueError):
            flags.append("invalid_source_rect")
        landmarks = _points(_field(dfl_image, "get_landmarks"))
        source_landmarks = _points(_field(dfl_image, "get_source_landmarks"))
        if landmarks is None:
            flags.append("invalid_or_missing_landmarks")
        else:
            spans = np.ptp(landmarks, axis=0)
            if np.min(spans) <= 1.0:
                flags.append("degenerate_landmarks")
            else:
                outside = np.any((landmarks < 0) | (landmarks >= np.array([width, height])), axis=1)
                if float(outside.mean()) > 0.1:
                    flags.append("landmarks_outside_crop")
                if pose_estimator is not None and width == height:
                    try:
                        angles = np.asarray(pose_estimator(landmarks.astype(np.float32), size=width), dtype=float)
                        if angles.shape == (3,) and np.isfinite(angles).all() and np.max(np.abs(angles)) <= math.pi / 2 + 1e-6:
                            pose.update({"available": True, "pitch": math.degrees(float(angles[0])),
                                         "yaw": math.degrees(float(-angles[1])), "roll": math.degrees(float(angles[2]))})
                            if abs(pose["roll"]) > 60:
                                flags.append("extreme_roll_estimate")
                    except (ValueError, TypeError, cv2.error):
                        flags.append("pose_estimate_failed")
            try:
                matrix = np.asarray(_field(dfl_image, "get_image_to_face_mat"), dtype=np.float64)
                if matrix.shape == (2, 3) and np.isfinite(matrix).all():
                    determinant = float(np.linalg.det(matrix[:, :2]))
                    if not math.isfinite(determinant) or abs(determinant) < 1e-8:
                        flags.append("degenerate_alignment_transform")
                    elif source_landmarks is not None:
                        projected = source_landmarks @ matrix[:, :2].T + matrix[:, 2]
                        residual = float(np.median(np.linalg.norm(projected - landmarks, axis=1)) / max(width, height))
                        if math.isfinite(residual):
                            alignment.update({"available": True, "residualFraction": residual})
                            if residual > 0.02:
                                flags.append("source_alignment_mismatch")
                        else:
                            flags.append("invalid_alignment_transform")
            except (TypeError, ValueError, np.linalg.LinAlgError):
                flags.append("invalid_alignment_transform")
    sharpness = float(metrics["sharpness"])
    baseline_sharpness = sharpness
    detail = {"available": False, "reason": "requires-valid-existing-68-landmarks"}
    if dfl_image is not None:
        landmarks = _points(_field(dfl_image, "get_landmarks"))
        if landmarks is not None:
            try:
                from core.imagelib.face_quality import face_detail_signals
                from facelib import LandmarksProcessor
                foreground = LandmarksProcessor.get_image_hull_mask(image.shape, landmarks)[..., 0]
                signals = face_detail_signals(image, landmarks, foreground=foreground)
                if REVIEW_DETAIL_METRIC is not None:
                    sharpness = signals["scores"][REVIEW_DETAIL_METRIC]
                detail = {"available": True, "selectedMetric": REVIEW_DETAIL_METRIC, **signals}
            except (ValueError, TypeError, cv2.error):
                detail = {"available": False, "reason": "invalid-existing-geometry-or-foreground-support"}
    exposure = float(max(0.0, 1.0 - abs(metrics["brightness"] - 0.5) / 0.5))
    components = {"sharpness": {"available": True, "score": sharpness, "weight": 0.55},
                  "exposure": {"available": True, "score": exposure, "weight": 0.20},
                  "sourceResolution": {"available": source_face is not None,
                                       "score": min(source_face["minimumDimension"] / 256.0, 1.0) if source_face else None, "weight": 0.15},
                  "alignmentConsistency": {"available": alignment["available"],
                                           "score": max(0.0, 1.0 - alignment["residualFraction"] / 0.08) if alignment["available"] else None, "weight": 0.10}}
    available = [part for part in components.values() if part["available"]]
    score = sum(part["score"] * part["weight"] for part in available) / sum(part["weight"] for part in available)
    return {"schemaVersion": 2, "method": "denoised-foreground-and-existing-metadata-v2", "score": score,
            "components": components, "sourceFace": source_face or {"available": False, "source": "existing-dfl-source-rect"},
            "detailCandidates": detail, "baselineSharpness": baseline_sharpness,
            "alignment": alignment, "pose": pose, "flags": flags,
            "occlusion": {"available": False, "reason": "requires-validated-model-or-human-labels"},
            "policy": "review-signals-only; pose-does-not-penalize-profile-coverage; unavailable-components-omitted"}
