import math
import torch
import torch.nn.functional as F


def _get_gaussian_kernel_3d(kernel_size=3, sigma=1.0, channels=1, device='cpu'):
    """Generate a 3D Gaussian kernel."""
    # Create grid coordinates.
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Compute the 2D Gaussian in XY.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Extend to Z/rotation; isotropic here, though dimensions could be configured separately.
    # Construct a grid over all three dimensions.
    range_vec = torch.arange(kernel_size, dtype=torch.float32, device=device) - mean
    xx, yy, zz = torch.meshgrid(range_vec, range_vec, range_vec, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * variance))

    # Normalize to preserve total mass.
    kernel = kernel / torch.sum(kernel)

    # Reshape to conv3d weights: [out_channels, in_channels, k_d, k_h, k_w].
    # Here in_channels=1 and out_channels=1.
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    # Repeat for multichannel inputs if needed; probabilities here are usually single-channel.
    if channels > 1:
        kernel = kernel.repeat(channels, 1, 1, 1, 1)

    return kernel.to(device)


def _apply_gaussian_smoothing_3d(preds, kernel_size=3, sigma=1.0):
    """Smooth a probability volume with a 3D Gaussian and periodic rotation padding.

    Args:
        preds: Probabilities [B, H, W, Rot].
    Returns:
        Smoothed probabilities [B, H, W, Rot]."""
    B, H, W, Rot = preds.shape
    device = preds.device

    # 1. Arrange dimensions for conv3d: [B, C, D, H, W].
    # Map H, W, Rot to convolution depth, height, width.
    # Input needs to be [B, 1, H, W, Rot]
    x = preds.unsqueeze(1)

    # 2. Pad the volume.
    # Use replicated edges or constant zeros for H and W.
    # Rotation requires circular padding to preserve periodic UAV headings.
    pad = kernel_size // 2

    # F.pad lists dimensions in reverse: left/right, top/bottom, front/back.
    # These pairs correspond to (Rot, W, H).

    # Apply circular padding to rotation, the last dimension, first.
    # Some PyTorch versions have limited support for 5D circular padding.
    # Concatenate rotation boundaries manually or pad dimensions separately.

    # Strategy: replicate W/H boundaries and wrap rotation boundaries.
    # Tensor layout here is [B, 1, H, W, Rot].

    # Step 2.1: Pad W and H (spatial) using replicate
    # pad_W_left, pad_W_right, pad_H_top, pad_H_bottom
    # conv3d padding must account for all three spatial dimensions.
    # Handle circular padding manually for compatibility.

    # Concatenate the end and beginning of the rotation axis.
    left_slice = x[..., -pad:]
    right_slice = x[..., :pad]
    x_padded_rot = torch.cat([left_slice, x, right_slice], dim=-1)  # [B, 1, H, W, Rot + 2*pad]

    # Step 2.2: Pad H and W (spatial)
    # Pad the second- and third-last dimensions.
    # F.pad for 5D input pads the last 3 dimensions (D, H, W).
    # Our mapping is (H->D, W->H, Rot->W) for conv3d names, but let's just stick to indices.
    # Rotation has already been padded; only H and W remain.
    # Use replicated boundaries.
    # For 5D tensors, F.pad specifies pairs for the last, second-last, and third-last axes.
    # With the last axis already padded, pad W and H next.
    # Mixing manual padding and F.pad requires care with the changed tensor shape.
    # An alternative is to allocate the fully padded tensor directly.

    # A single circular F.pad call is possible only when 5D circular padding is supported.
    # For versions without that support, use the following approach:
    # 1. Wrap the rotation dimension manually.
    # 2. Replicate H/W boundaries with F.pad.

    # Rotation was already padded above in x_padded_rot.
    # Replicate-pad H (dimension 2) and W (dimension 3) of x_padded_rot.
    # Padding order: left/right, top/bottom, front/back.
    # Corresponding dimensions: padded rotation, W, H.
    # Set rotation padding to zero because that axis is already padded.
    x_fully_padded = F.pad(x_padded_rot, (0, 0, pad, pad, pad, pad), mode='replicate')

    # 3. Generate the kernel.
    kernel = _get_gaussian_kernel_3d(kernel_size, sigma, channels=1, device=device)

    # 4. Convolve.
    # x_fully_padded: [B, 1, H+2p, W+2p, Rot+2p]
    out = F.conv3d(x_fully_padded, kernel, padding=0)

    return out.squeeze(1)


def get_batch_adaptive_indices(scores_flat, top_p=0.95, prob_threshold=0.0, min_k=1, normalize=True):
    """Select an adaptive number of top-ranked indices per batch item.

    Args:
        scores_flat: Raw logits or probabilities [B, N].
        top_p: Cumulative probability threshold, e.g. 0.95.
        prob_threshold: Absolute probability threshold, e.g. 0.001.
        min_k: Minimum candidates retained per item.
        normalize: Normalize inputs; recommended unless probabilities already sum to 1.

    Returns:
        indices: Sorted indices [B, Max_K].
        mask: Boolean mask [B, Max_K], True for valid entries, False for padding.
        probs: Corresponding probabilities [B, Max_K]."""
    B, N = scores_flat.shape

    # --- 1. Normalize safely ---
    if normalize:
        # Negative values usually indicate logits; use softmax.
        if scores_flat.min() < 0 or scores_flat.max() > 1.0:
            probs_flat = F.softmax(scores_flat, dim=-1)
        else:
            # For nonnegative values not summing to one, use L1 normalization.
            probs_flat = scores_flat / (scores_flat.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        probs_flat = scores_flat

    # --- 2. Sort all candidates ---
    sorted_probs, sorted_indices = torch.sort(probs_flat, descending=True, dim=-1)

    # --- 3. Compute the top-p mask ---
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # mask_p selects entries whose cumulative probability is below top_p.
    # Shift right to retain the candidate that first crosses top_p.
    mask_p = cumulative_probs < top_p
    mask_p = torch.cat([torch.ones(B, 1, device=scores_flat.device, dtype=torch.bool),
                        mask_p[:, :-1]], dim=1)

    # --- 4. Apply an additional absolute probability threshold ---
    # Discard tiny probabilities even if top-p would retain every point.
    if prob_threshold > 0:
        mask_t = sorted_probs >= prob_threshold
        final_mask = mask_p & mask_t
    else:
        final_mask = mask_p

    # --- 5. Enforce minimum K ---
    if min_k > 0:
        final_mask[:, :min_k] = True

    # --- 6. Truncate dynamically ---
    lengths = final_mask.sum(dim=1)  # [B]
    max_k = lengths.max().item()

    # Limit max_k to N in case min_k exceeds the candidate count.
    max_k = min(max_k, N)

    selected_indices = sorted_indices[:, :max_k]
    selected_probs = sorted_probs[:, :max_k]
    selected_mask = final_mask[:, :max_k]

    return selected_indices, selected_mask, selected_probs, lengths


def filter_batch_by_dynamic_threshold(scores_flat, thresholds, min_topN=64):
    """Filter with a per-item threshold while retaining at least min_topN candidates.

    Args:
        scores_flat: Scores or probabilities [B, N], in the same units as thresholds.
        thresholds: Per-item threshold tensor [B].
        min_topN: Minimum number retained even below the threshold.

    Returns:
        indices: Sorted, padded indices [B, Max_K].
        mask: [B, Max_K], True for selected entries and False for padding/rejections.
        values: Corresponding scores [B, Max_K]."""
    B, N = scores_flat.shape

    # 1. Sort all candidates in descending order.
    # Sorting supports efficient ragged-batch handling for top-K and threshold selection.
    sorted_scores, sorted_indices = torch.sort(scores_flat, descending=True, dim=-1)

    # 2. Construct the dynamic mask with vectorized operations.
    # Expand thresholds for broadcasting: [B] -> [B, 1].
    thresholds_exp = thresholds.view(B, 1)

    # Condition A: scores >= threshold.
    # Broadcasting compares each row against its own threshold.
    mask_threshold = sorted_scores >= thresholds_exp

    # Condition B: always retain the first min_topN candidates.
    # Keep the leading min_topN entries regardless of the threshold.
    if min_topN > 0:
        # Create a mask that is True only in the first min_topN positions.
        # Since scores are sorted, the first min_topN positions contain the largest values.
        # Use a broadcast [1, N] mask instead of [B, N] to save memory.
        mask_min = torch.arange(N, device=scores_flat.device).expand(B, N) < min_topN

        # Combine masks with logical OR.
        # Keep entries above the threshold or among the leading min_topN.
        final_mask = mask_threshold | mask_min
    else:
        final_mask = mask_threshold

    # 3. Truncate dynamically.
    # Find the maximum retained length across the batch.
    lengths = final_mask.sum(dim=1)  # [B]
    max_k = lengths.max().item()

    # Clamp max_k defensively to avoid out-of-bounds slicing.
    max_k = min(max_k, N)

    # 4. Slice all outputs consistently.
    # Slice to max_k; PyTorch handles the tensor data.
    selected_indices = sorted_indices[:, :max_k]
    selected_scores = sorted_scores[:, :max_k]
    selected_mask = final_mask[:, :max_k]

    return selected_indices, selected_mask, selected_scores
