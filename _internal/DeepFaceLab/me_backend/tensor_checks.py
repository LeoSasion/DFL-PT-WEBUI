"""Batch device reductions without a host synchronization for every tensor."""
import torch


def all_finite(tensors):
    """Return one device boolean for a nonempty collection on the same device."""
    return torch.stack([torch.isfinite(value).all() for value in tensors]).all()


def finite_unit_interval(value):
    """One device flag; callers validate nonempty shape before reducing."""
    return torch.isfinite(value).all() & (value.amin() >= 0) & (value.amax() <= 1)


def gradients_are_finite(parameters):
    """Reject disconnected gradients before reading one aggregate device flag."""
    grads = [parameter.grad for parameter in parameters]
    return bool(grads) and all(grad is not None for grad in grads) and bool(all_finite(grads))


def scalar_metrics(metrics):
    """Copy detached scalar metrics together; preserve their names and values."""
    if not metrics:
        return {}
    packed = torch.stack([value.detach() for value in metrics.values()])
    return dict(zip(metrics, packed.cpu().tolist()))
