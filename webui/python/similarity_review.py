"""Weight-free, same-scale SSIM and conservative non-transitive grouping.

SSIM follows Wang et al. (2004): Gaussian 11x11, sigma 1.5,
C1=(0.01 L)^2, C2=(0.03 L)^2, L=1; valid-window mean.
https://ece.uwaterloo.ca/~z70wang/research/ssim/
This verifies image structure, not identity, occlusion or semantic equivalence.
"""
import cv2
import numpy as np

SSIM_SIZE = 128


def ssim_features(image):
    scaled = cv2.resize(image, (SSIM_SIZE, SSIM_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    mean = cv2.GaussianBlur(gray, (11, 11), 1.5)
    variance = np.maximum(cv2.GaussianBlur(gray * gray, (11, 11), 1.5) - mean * mean, 0)
    return {"gray": gray, "mean": mean, "variance": variance,
            "aspect": float(image.shape[1] / image.shape[0])}


def same_scale_ssim(left, right):
    # Resizing two different crops to a square must not manufacture a duplicate.
    if abs(left["aspect"] / right["aspect"] - 1.0) > 0.01:
        return 0.0
    if np.array_equal(left["gray"], right["gray"]):
        return 1.0
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    covariance = cv2.GaussianBlur(left["gray"] * right["gray"], (11, 11), 1.5) - left["mean"] * right["mean"]
    numerator = (2 * left["mean"] * right["mean"] + c1) * (2 * covariance + c2)
    denominator = (left["mean"] ** 2 + right["mean"] ** 2 + c1) * (left["variance"] + right["variance"] + c2)
    score = float(np.mean((numerator / denominator)[5:-5, 5:-5]))
    return float(np.clip(score, -1.0, 1.0)) if np.isfinite(score) else -1.0


def consistent_groups(accepted, order):
    """Disjoint complete-link partition; A~B and B~C cannot imply A~C.

    Visit best-quality samples first. A sample may join only a group in which
    it passed the explicit pair verification against every existing member.
    Conservative partitions may leave a bridging sample ungrouped.
    """
    groups = []
    for index in order:
        for group in groups:
            if all(bool(accepted[index, other]) for other in group):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups
