"""Image-loss coverage, separate from the supervised face/merge mask."""
import cv2
import numpy as np
from .config import TRAINING_MASK_MODES


def expansion_geometry(resolution):
    """Pixel radius rounded to nearest integer; feather spans the whole radius."""
    radius = max(1, int(resolution * 0.1 + 0.5))
    return radius, radius


def image_loss_mask(face_mask, mode="face"):
    """Build N1HW float weights without altering the native mask supervision.

    Expansion uses Euclidean distance from the native mask's 0.5 contour.
    Weights are 1 inside that contour and fall linearly to 0 across the entire
    added radius. There is no full-weight plateau outside the original face.
    """
    mask = np.asarray(face_mask, dtype=np.float32)
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[2] != mask.shape[3]:
        raise ValueError("Expected square N1HW face masks")
    if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("Face mask weights must be finite and within [0,1]")
    if mode not in TRAINING_MASK_MODES:
        raise ValueError("Unknown TAE training mask mode")
    if mode == "face":
        return mask.copy()
    if mode == "full":
        return np.ones_like(mask)
    radius, feather = expansion_geometry(mask.shape[-1])
    result = np.empty_like(mask)
    # OpenCV's IPP precise transform can vary between identical calls
    # (opencv/opencv#24082). Bypass it only for this distance calculation.
    # OpenCV 4.12 stores useIPP in CoreTLSData: this setting is thread-local,
    # and restoring it also preserves acceleration for later work on this thread.
    use_ipp = cv2.ipp.useIPP()
    try:
        if use_ipp:
            cv2.ipp.setUseIPP(False)
        for index, plane in enumerate(mask[:, 0]):
            core = plane >= 0.5
            if not core.any():
                raise ValueError("Expanded training mask needs a face region with weight >= 0.5")
            distance = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            transition = np.clip((radius - distance) / feather, 0, 1)
            result[index, 0] = transition
    finally:
        if use_ipp:
            cv2.ipp.setUseIPP(True)
    return result
