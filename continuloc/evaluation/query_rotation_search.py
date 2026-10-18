"""Discrete query-rotation retrieval with independently configured scales.

The gallery stays north-aligned, with one or multiple scales. Each query is
optionally scale-normalized and encoded at equally spaced rotations. Each angle contributes
``exp(-distance / temperature)`` with the same temperature used for heading.
Each gallery candidate score is the sum of its angle responses. Heading at the winning
location is their softmax, averaged as a local deviation from the peak angle,
then returned in the same convention as ``rotdeg_fm_north_anticlock``.
"""
import math

import numpy as np
import torch





def resolve_forward_variant_limit(max_forward_variants, val_chunk_size=1024):
    """Images per network forward. An unset cap keeps the existing chunk size."""
    if max_forward_variants is None or str(max_forward_variants).strip() == "":
        limit = int(val_chunk_size)
    else:
        limit = int(max_forward_variants)
    if limit < 1:
        raise ValueError(f"max_forward_variants must be positive, got {max_forward_variants!r}")
    return limit


def normalized_base_grid(height, width, dtype=torch.float32, device=None):
    """Output sampling grid matching ``affine_grid(..., align_corners=False)``."""
    height = int(height)
    width = int(width)
    if height < 1 or width < 1:
        raise ValueError(f"sampling grid size must be positive, got {(height, width)}")
    ys = torch.linspace(-1, 1, height, device=device, dtype=dtype) * ((height - 1) / height)
    xs = torch.linspace(-1, 1, width, device=device, dtype=dtype) * ((width - 1) / width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def rotation_scale_grid(base_grid, rot_rad, scale_f):
    """Apply rotation and scale to a precomputed ``[H, W, 2]`` sampling grid.

    The result is ``[B, H, W, 2]`` in the same convention as ``_warp_uav_imgs``:
    ``x' = scale * (cos x + sin y)``, ``y' = scale * (-sin x + cos y)``.
    """
    if base_grid.shape[-1] != 2:
        raise ValueError(f"base grid must end with xy, got {tuple(base_grid.shape)}")
    rot = torch.as_tensor(rot_rad).to(dtype=base_grid.dtype, device=base_grid.device).reshape(-1)
    scale = torch.as_tensor(scale_f).to(dtype=base_grid.dtype, device=base_grid.device).reshape(-1)
    if rot.numel() != scale.numel():
        raise ValueError("rotation and scale must describe the same number of samples")
    if rot.numel() == 0:
        return base_grid.new_empty((0, *base_grid.shape))
    x = base_grid[..., 0]
    y = base_grid[..., 1]
    cos_v = torch.cos(rot)[:, None, None]
    sin_v = torch.sin(rot)[:, None, None]
    scale = scale[:, None, None]
    x_s = scale * (cos_v * x + sin_v * y)
    y_s = scale * (-sin_v * x + cos_v * y)
    return torch.stack((x_s, y_s), dim=-1)


def variant_coordinates(flat_index, n_angles):
    """Map a flattened query-major variant index to ``(query, angle)``."""
    n_angles = int(n_angles)
    if n_angles < 1:
        raise ValueError("n_angles must be positive")
    flat_index = torch.as_tensor(flat_index)
    query_index = torch.div(flat_index, n_angles, rounding_mode="floor")
    angle_index = flat_index - query_index * n_angles
    return query_index, angle_index


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


def resolve_heading_temperature(value, default=0.02):
    """Softmax temperature on feature distance. Smaller values stay nearer the peak."""
    if value is None or str(value).strip() == "":
        value = default
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"heading_temperature must be positive and finite, got {value!r}")
    return temperature


def feature_chord_distance(cosine):
    """Euclidean distance between L2-normalized features, from their cosine."""
    clipped = torch.as_tensor(cosine).clamp(-1.0, 1.0)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * clipped, min=0.0))


def exponential_heading_from_responses(responses, warp_angles_rad, temperature=0.02):
    """Estimate heading at one location from cosine responses.

    ``responses`` is ``[B, A]``. Distance is ``sqrt(max(0, 2 - 2 cos))``.
    Weights are ``softmax(-distance / temperature)``. The estimate is the peak
    warp plus the weighted local deviation, then negated to match the
    north-alignment warp ``rot = -heading``.
    """
    if responses.ndim != 2:
        raise ValueError(f"responses must have shape [B, A], got {tuple(responses.shape)}")
    temperature = resolve_heading_temperature(temperature)
    angles = torch.as_tensor(warp_angles_rad, dtype=responses.dtype, device=responses.device).reshape(-1)
    if responses.shape[-1] != angles.numel():
        raise ValueError(
            f"responses have {responses.shape[-1]} angles, warp grid has {angles.numel()}"
        )
    distance = feature_chord_distance(responses)
    weights = torch.softmax(-distance / responses.new_tensor(temperature), dim=-1)
    peak_angle = angles.index_select(0, responses.argmax(dim=-1))
    delta = angles - peak_angle[:, None]
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    warp_hat = peak_angle + (weights * delta).sum(dim=-1)
    return torch.atan2(torch.sin(-warp_hat), torch.cos(-warp_hat))


def absolute_heading_error_rad(estimated, target):
    delta = torch.as_tensor(estimated) - torch.as_tensor(target)
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def exponential_location_scores(responses, temperature=0.02):
    """Sum ``exp(-distance / temperature)`` over angles.

    ``responses`` is ``[B, A, G]`` cosine similarities. The returned score is
    ``logsumexp(-distance / temperature)`` over angles, which ranks locations
    identically to the sum and stays finite when the temperature is small.
    """
    if responses.ndim != 3:
        raise ValueError(f"responses must have shape [B, A, G], got {tuple(responses.shape)}")
    temperature = resolve_heading_temperature(temperature)
    distance = feature_chord_distance(responses).to(dtype=torch.float64)
    return torch.logsumexp(-distance / temperature, dim=1)


def resolve_rot_retrieval_mode(mode):
    if mode not in ('rot_weighting', 'rot_argmax', 'rot_xy_argmax'):
        raise ValueError(f'Unknown rot_retrieval_mode: {mode!r}')
    return mode


def rotation_search_ranking(responses, warp_angles_rad, top_k, temperature=0.02,
                            mode='rot_weighting', return_all_headings=False):
    """Rank gallery locations by the summed exponential angle responses.

    ``responses`` is ``[B, A, G]``. A small distance at any angle raises that
    location's score. The heading uses only the winning location.
    """
    if responses.ndim != 3:
        raise ValueError(f"responses must have shape [B, A, G], got {tuple(responses.shape)}")
    mode = resolve_rot_retrieval_mode(mode)
    angles = torch.as_tensor(warp_angles_rad, device=responses.device, dtype=responses.dtype).reshape(-1)
    if angles.numel() != responses.shape[1]:
        raise ValueError('Angle grid does not match responses')
    if mode == 'rot_weighting':
        scores = exponential_location_scores(responses, temperature)
    elif mode == 'rot_argmax':
        scores, best_angles = responses.max(dim=1)
    else:
        scores = responses.flatten(1)  # angle-major [A, G]; repeated xy is intentional.
    k = min(int(top_k), scores.shape[-1])
    if k < 1:
        raise ValueError("top_k must be positive")
    indices = torch.topk(scores, k=k, dim=-1, largest=True).indices
    if mode == 'rot_weighting':
        chosen = responses.gather(2, indices.unsqueeze(1).expand(-1, responses.shape[1], -1))
        heading = exponential_heading_from_responses(
            chosen.transpose(1, 2).reshape(-1, responses.shape[1]), angles, temperature,
        ).reshape(indices.shape)
    else:
        if mode == 'rot_xy_argmax':
            angle_indices = indices // responses.shape[2]
            indices = indices % responses.shape[2]
        else:
            angle_indices = best_angles.gather(1, indices)
        warp = angles[angle_indices]
        heading = torch.atan2(torch.sin(-warp), torch.cos(-warp))
    return indices, heading if return_all_headings else heading[:, 0]


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
