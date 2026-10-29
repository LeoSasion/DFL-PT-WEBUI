"""Pixel/structure/detail objectives through the frozen decoder, without clamp."""
import torch
from torch.nn import functional as F
from me_backend.tensor_checks import all_finite


def masked_mean(error, mask):
    """Normalize each face separately; large masks do not dominate the batch."""
    weights = mask.float().expand_as(error)
    areas = weights.flatten(1).sum(1)
    return _weighted_mean(error, weights, areas)


def _weighted_mean(error, weights, areas):
    sums = (error.float() * weights).flatten(1).sum(1)
    return (sums / areas.clamp_min(1e-8)).mean()


def dssim_map(prediction, target):
    # Local statistics, [0,1] data range, no cross-image/batch coupling.
    x, y = prediction.float(), target.float()
    average = lambda z: F.avg_pool2d(z, 7, stride=1, padding=3, count_include_pad=False)
    mx, my = average(x), average(y)
    vx, vy = (average(x*x)-mx*mx).clamp_min(0), (average(y*y)-my*my).clamp_min(0)
    cov = average(x*y)-mx*my
    ssim = ((2*mx*my+0.01**2)*(2*cov+0.03**2) /
            ((mx.square()+my.square()+0.01**2)*(vx+vy+0.03**2)))
    return (1-ssim).clamp_min(0)*0.5


def edge_loss(prediction, target, mask):
    dx = (prediction[..., 1:]-prediction[..., :-1])-(target[..., 1:]-target[..., :-1])
    dy = (prediction[..., 1:, :]-prediction[..., :-1, :])-(target[..., 1:, :]-target[..., :-1, :])
    return 0.5*(masked_mean(dx.abs(), mask[..., 1:]*mask[..., :-1])+
                masked_mean(dy.abs(), mask[..., 1:, :]*mask[..., :-1, :]))


def reconstruction_loss(image, predicted_mask, batch, config):
    target, mask, roi = batch["target"], batch["mask"], batch["eyes_mouth"]
    coverage = batch.get("image_mask", mask)
    if config.training_mask_mode != "face" and "image_mask" not in batch:
        raise ValueError("Expanded/full reconstruction needs a prepared image-loss mask")
    if image.shape != target.shape or predicted_mask.shape != mask.shape:
        raise ValueError("Model image/mask shape differs from aligned target")
    finite, nonempty = torch.stack((
        all_finite((image, predicted_mask, target, mask, roi)),
        (mask.flatten(1).sum(1) > 0).all(),
    )).cpu().tolist()
    if not finite:
        raise FloatingPointError("Nonfinite loss input")
    if not nonempty:
        raise ValueError("Training face masks must be nonempty")
    error = torch.sqrt((image.float()-target.float()).square()+1e-6)-0.001
    # RGB and DSSIM use the same coverage, channel expansion and per-face
    # denominator. Reuse only this target-derived work; keep independent image
    # graphs and their backward accumulation order.
    weights = coverage.float().expand_as(error)
    areas = weights.flatten(1).sum(1)
    terms = {
        "rgb": _weighted_mean(error, weights, areas),
        "dssim": _weighted_mean(dssim_map(image, target), weights, areas),
        "edge": edge_loss(image, target, coverage),
        "eyes_mouth": masked_mean(error, roi*mask)+edge_loss(image, target, roi*mask),
        "mask": F.binary_cross_entropy(predicted_mask.float(), mask.float()),
    }
    total = sum(getattr(config, name+"_weight")*value for name, value in terms.items())
    return total, terms

