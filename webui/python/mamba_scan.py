"""FP32 selective recurrence implemented using PyTorch affine prefix scans.

The recurrence is h[t] = exp(delta[t]*A)*h[t-1] + delta[t]*B[t]*u[t].
Chunking bounds temporary memory; state is carried across every chunk. This is
inference-only and must be checked against the pinned official serial reference.
"""
import torch
import torch.nn.functional as F


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    if torch.is_grad_enabled() and any(x.requires_grad for x in (u, delta, A, B, C)):
        raise ValueError('The portable selective scan supports inference only')
    if A.is_complex() or u.ndim != 3 or A.ndim != 2 or delta.shape != u.shape:
        raise ValueError('Real FP32 selective scan contract required')
    original_dtype = u.dtype
    u, delta, A, B, C = (value.float() for value in (u, delta, A, B, C))
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, channels, length = u.shape
    states = A.shape[1]
    if A.shape[0] != channels:
        raise ValueError('Selective scan channel mismatch')
    def terms(value, start, stop):
        if value.ndim == 2:
            return value[None, :, :, None]
        if value.ndim == 3:
            return value[:, None, :, start:stop]
        if value.ndim == 4 and value.shape[1] == 1:
            return value[:, 0, None, :, start:stop]
        if value.ndim == 4 and channels % value.shape[1] == 0:
            return value[:, :, :, start:stop].repeat_interleave(channels // value.shape[1], dim=1)
        raise ValueError('Unsupported selective scan coefficient shape')
    state = torch.zeros((batch, channels, states), dtype=torch.float32, device=u.device)
    outputs = []
    for start in range(0, length, 512):
        stop = min(length, start + 512)
        dt = delta[:, :, None, start:stop]
        aa = torch.exp(dt * A[None, :, :, None])
        bb = dt * terms(B, start, stop) * u[:, :, None, start:stop]
        offset = 1
        while offset < stop - start:
            next_b = torch.cat((bb[..., :offset], bb[..., offset:] + aa[..., offset:] * bb[..., :-offset]), dim=-1)
            aa = torch.cat((aa[..., :offset], aa[..., offset:] * aa[..., :-offset]), dim=-1)
            bb = next_b
            offset *= 2
        history = bb + aa * state[..., None]
        state = history[..., -1]
        outputs.append((history * terms(C, start, stop)).sum(dim=2))
    result = torch.cat(outputs, dim=-1)
    if D is not None:
        result = result + u * D.float()[None, :, None]
    if z is not None:
        result = result * F.silu(z.float())
    result = result.to(original_dtype)
    return (result, state) if return_last_state else result


selective_scan_ref = selective_scan_fn
