"""Discrete query-rotation retrieval at one scene-mean scale.

The gallery stays north-aligned. Each query is scaled to the scene mean and
encoded at equally spaced rotations. Location scores sum the cosine responses
over those rotations. The heading is the circular mean of the responses at the
winning location, returned in the same convention as ``rotdeg_fm_north_anticlock``.
"""
import math

import numpy as np
import torch


_IMPLEMENTED_RETRIEVAL_MODES = ("north_aligned", "query_rotation_search")
_RESERVED_RETRIEVAL_MODES = {
    "reference_rotation_search": (
        "reference_rotation_search is reserved: rotating satellite crops is not implemented"
    ),
}


def resolve_retrieval_mode(mode):
    if mode is None or str(mode).strip() == "":
        return "north_aligned"
    name = str(mode).strip().lower()
    if name in _RESERVED_RETRIEVAL_MODES:
        raise NotImplementedError(_RESERVED_RETRIEVAL_MODES[name])
    if name not in _IMPLEMENTED_RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval_mode: {mode!r}")
    return name


def require_scene_mean_scale(gallery_scale_mode):
    mode = "scene_mean" if gallery_scale_mode is None else str(gallery_scale_mode).strip().lower()
    if mode != "scene_mean":
        raise ValueError(
            "query_rotation_search uses each scene's mean scale; "
            f"val_gallery_scale_mode must be scene_mean, got {gallery_scale_mode!r}"
        )
    return mode


def rotation_steps_rad(step_deg):
    """Warp angles 0, step, ... strictly below 360 degrees, in radians."""
    step = float(step_deg)
    if not math.isfinite(step) or step <= 0.0 or step > 360.0:
        raise ValueError(f"query_rotation_step_deg must be in (0, 360], got {step_deg!r}")
    turns = 360.0 / step
    count = int(round(turns))
    if count < 1 or abs(turns - count) > 1e-6:
        raise ValueError(f"query_rotation_step_deg must divide 360, got {step_deg!r}")
    degrees = torch.arange(count, dtype=torch.float64) * step
    return torch.deg2rad(degrees).to(dtype=torch.float32)


def circular_heading_from_responses(responses, warp_angles_rad):
    """Estimate heading at one location from cosine responses.

    ``responses`` is ``[B, A]``. Cosines may be negative, so weights are the
    responses shifted to a minimum of zero. A flat response uses equal weights.
    The returned heading is the negated circular mean of the warp angles, matching
    the north-alignment warp ``rot = -heading``.
    """
    if responses.ndim != 2:
        raise ValueError(f"responses must have shape [B, A], got {tuple(responses.shape)}")
    angles = torch.as_tensor(warp_angles_rad, dtype=responses.dtype, device=responses.device).reshape(-1)
    if responses.shape[-1] != angles.numel():
        raise ValueError(
            f"responses have {responses.shape[-1]} angles, warp grid has {angles.numel()}"
        )
    shifted = responses - responses.min(dim=-1, keepdim=True).values
    denom = shifted.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(shifted, 1.0 / responses.shape[-1])
    weights = torch.where(denom > 0, shifted / denom.clamp_min(torch.finfo(shifted.dtype).tiny), uniform)
    sin_sum = (weights * torch.sin(angles)).sum(dim=-1)
    cos_sum = (weights * torch.cos(angles)).sum(dim=-1)
    warp_hat = torch.atan2(sin_sum, cos_sum)
    return torch.atan2(torch.sin(-warp_hat), torch.cos(-warp_hat))


def absolute_heading_error_rad(estimated, target):
    delta = torch.as_tensor(estimated) - torch.as_tensor(target)
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def rotation_search_ranking(responses, warp_angles_rad, top_k):
    """Rank gallery locations by the sum of per-angle cosine responses.

    ``responses`` is ``[B, A, G]``. The heading uses only the winning location.
    """
    if responses.ndim != 3:
        raise ValueError(f"responses must have shape [B, A, G], got {tuple(responses.shape)}")
    scores = responses.sum(dim=1)
    k = min(int(top_k), scores.shape[-1])
    if k < 1:
        raise ValueError("top_k must be positive")
    indices = torch.topk(scores, k=k, dim=-1, largest=True).indices
    chosen = responses.gather(
        2, indices[:, :1].unsqueeze(1).expand(-1, responses.shape[1], 1)
    ).squeeze(-1)
    heading = circular_heading_from_responses(chosen, warp_angles_rad)
    return indices, heading


def summarize_heading_errors(labels, errors_rad):
    labels = np.asarray(labels, dtype=object)
    errors = torch.as_tensor(errors_rad, dtype=torch.float64).detach().cpu().reshape(-1)
    if errors.numel() != len(labels):
        raise ValueError("Heading errors do not match the number of queries")
    degrees = errors * (180.0 / math.pi)
    summary = {}
    for name in ("val", "test"):
        selected = torch.from_numpy(labels == name)
        if not bool(selected.any()):
            continue
        values = degrees[selected]
        summary[name] = {
            "heading_error_deg_mean": float(values.mean()),
            "heading_error_deg_median": float(values.median()),
        }
    return summary


def attach_heading_errors(result, summary):
    """Add heading stats beside Recall@K for each split present in ``result``."""
    if not summary:
        return result
    nested = "recall@1" not in result and any(name in result for name in summary)
    if nested:
        for name, stats in summary.items():
            result[name].update(stats)
        return result
    result.update(summary.get("test", next(iter(summary.values()))))
    return result
