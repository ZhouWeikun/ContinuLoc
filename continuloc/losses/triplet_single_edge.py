"""Hardest positive/negative triplet with explicit gate and row-selection controls."""
import math

import torch
from torch import nn
from torch.nn import functional as F

class tripletLoss_singleEdge(nn.Module):
    """Select one hardest positive and negative per query.

    Input is a [queries, references] distance or similarity matrix. The second
    argument can be a bool positive mask or floating positive weights. With one
    weight matrix, weights must be in [0, 1] and values > 0.5 are positive. With
    two weight matrices, pos_weight > neg_weight is positive (ties are negative).
    Weights define membership only; they never multiply the loss.

    Invalid rows (missing positives or negatives) are skipped. Row filtering,
    when enabled, additionally keeps only delta > 0. Mean divides by the number
    of retained rows; sum does not divide. Empty selections return graph-connected
    zero. Gate 'hard' uses ReLU(delta); 'softplus' uses softplus(alpha*delta)/alpha.
    Learnable alpha affects only the softplus gate and is positive via exp(log_alpha).
    """

    def __init__(self, alpha=10.0, margin=0.0, gate='softplus',
                 enable_row_filter=False, learnable_alpha=False,
                 metric='dist', reduction='mean'):
        super().__init__()
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        if not math.isfinite(margin):
            raise ValueError('margin must be finite')
        if gate not in {'hard', 'softplus'}:
            raise ValueError("gate must be 'hard' or 'softplus'")
        if reduction not in {'mean', 'sum'}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.margin = float(margin)
        self.gate = gate
        self.enable_row_filter = bool(enable_row_filter)
        self.learnable_alpha = bool(learnable_alpha)
        self.metric = self._metric(metric)
        self.reduction = reduction
        value = torch.tensor(math.log(alpha), dtype=torch.float32)
        if learnable_alpha:
            self.log_alpha = nn.Parameter(value)
        else:
            self.register_buffer('log_alpha', value)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @staticmethod
    def _metric(metric):
        if metric in {'dist', 'distance', 'l2', 'euclidean'}:
            return 'dist'
        if metric in {'sim', 'similarity', 'cos', 'cosine'}:
            return 'similarity'
        raise ValueError('metric must be distance or similarity')

    @staticmethod
    def _positive_mask(matrix, targets, negative_weights):
        if not torch.is_tensor(targets) or targets.shape != matrix.shape:
            raise ValueError('mask/weights must be a tensor with the same shape as feat_mat')
        targets = targets.to(device=matrix.device)
        if targets.dtype == torch.bool:
            if negative_weights is not None:
                raise ValueError('negative weights cannot accompany a boolean mask')
            return targets
        if not targets.is_floating_point():
            raise ValueError('use a bool mask or floating-point weights; integer masks are ambiguous')
        if not torch.isfinite(targets).all():
            raise ValueError('positive weights must be finite')
        if negative_weights is None:
            if ((targets < 0) | (targets > 1)).any():
                raise ValueError('a single weight matrix must lie in [0, 1]')
            return targets > 0.5
        if (not torch.is_tensor(negative_weights)
                or negative_weights.shape != matrix.shape
                or not negative_weights.is_floating_point()):
            raise ValueError('negative weights must be floating-point and match feat_mat shape')
        negative_weights = negative_weights.to(device=matrix.device)
        if not torch.isfinite(negative_weights).all():
            raise ValueError('negative weights must be finite')
        return targets > negative_weights

    def forward(self, feat_mat, pos_mask_or_weight, neg_weight=None, *,
                metric=None, enable_row_filter=None):
        if feat_mat.ndim != 2 or not feat_mat.is_floating_point():
            raise ValueError('feat_mat must be a floating-point [queries, references] matrix')
        if not torch.isfinite(feat_mat).all():
            raise ValueError('feat_mat must contain finite values')
        mode = self.metric if metric is None else self._metric(metric)
        row_filter = self.enable_row_filter if enable_row_filter is None else bool(enable_row_filter)
        mask = self._positive_mask(feat_mat, pos_mask_or_weight, neg_weight)
        # Promote half precision before softplus and reductions. Index invalid rows
        # out before hard mining, avoiding infinities in their autograd paths.
        values = feat_mat.float() if feat_mat.dtype in {torch.float16, torch.bfloat16} else feat_mat
        valid = mask.any(dim=1) & (~mask).any(dim=1)
        values = values[valid]
        mask = mask[valid]
        alpha = self.alpha.to(device=values.device, dtype=values.dtype)
        if values.shape[0] == 0:
            return values.sum() + alpha * 0.0
        if mode == 'dist':
            positive = values.masked_fill(~mask, -torch.inf).max(dim=1).values
            negative = values.masked_fill(mask, torch.inf).min(dim=1).values
            delta = positive - negative + self.margin
        else:
            positive = values.masked_fill(~mask, torch.inf).min(dim=1).values
            negative = values.masked_fill(mask, -torch.inf).max(dim=1).values
            delta = negative - positive + self.margin
        if row_filter:
            delta = delta[delta > 0]
        if delta.numel() == 0:
            return delta.sum() + alpha * 0.0
        losses = (F.relu(delta) + alpha * 0.0 if self.gate == 'hard'
                  else F.softplus(alpha * delta) / alpha)
        return losses.mean() if self.reduction == 'mean' else losses.sum()
