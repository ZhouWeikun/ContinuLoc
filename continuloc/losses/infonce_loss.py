import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """InfoNCE for Stage-1 UAV query to satellite reference matching."""

    def __init__(
        self,
        temperature=0.1,
        negative_mode="batch_and_explicit",
        learnable_temperature=False,
        gate="soft",
    ):
        super().__init__()
        temperature = float(temperature)
        self.learnable_temperature = bool(learnable_temperature)
        self.negative_mode = str(negative_mode).lower()
        self.gate = str(gate).strip().lower()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if self.negative_mode not in {"batch", "explicit", "batch_and_explicit"}:
            raise ValueError(
                f"Unsupported negative_mode={negative_mode!r}. "
                "Expected one of: batch, explicit, batch_and_explicit."
            )
        if self.gate not in {"soft", "hard"}:
            raise ValueError(
                f"Unsupported gate={gate!r}. Expected one of: soft, hard."
            )
        log_temperature = torch.tensor(math.log(temperature), dtype=torch.float32)
        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature, persistent=False)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = f"{prefix}log_temperature"
        has_saved_temperature = key in state_dict
        if not self.learnable_temperature and has_saved_temperature:
            state_dict = dict(state_dict)
            state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # Old checkpoints predate learnable temperature. Keep the configured
        # initialization when resuming them with the new switch enabled.
        if self.learnable_temperature and not has_saved_temperature and key in missing_keys:
            missing_keys.remove(key)

    def forward(
            self,
            query,
            positive_key,
            explicit_negative_keys=None,
            metric="similarity",
            normalize=True,
    ):
        if query.dim() != 2 or positive_key.dim() != 2:
            raise ValueError("query and positive_key must be [N, D].")

        if query.shape != positive_key.shape:
            raise ValueError(
                f"query and positive_key must have the same shape, "
                f"got {query.shape} and {positive_key.shape}"
            )

        if explicit_negative_keys is not None:
            if explicit_negative_keys.dim() != 2:
                raise ValueError("explicit_negative_keys must be [M, D] when provided.")

            if explicit_negative_keys.shape[1] != query.shape[1]:
                raise ValueError(
                    f"explicit_negative_keys must have feature dim {query.shape[1]}, "
                    f"got {explicit_negative_keys.shape[1]}"
                )

        metric = str(metric).lower()
        if metric in {"sim", "similarity", "cos", "cosine", "simility"}:
            metric = "similarity"
        elif metric in {"dist", "distance", "l2", "euclidean"}:
            metric = "dist"
        else:
            raise ValueError(
                f"Unsupported metric={metric!r}. "
                "Expected one of: 'similarity', 'dist'."
            )

        normalize = bool(normalize)

        if normalize:
            query = F.normalize(query, dim=-1)
            positive_key = F.normalize(positive_key, dim=-1)

            if explicit_negative_keys is not None:
                explicit_negative_keys = F.normalize(explicit_negative_keys, dim=-1)

        neg_logits = []

        if metric == "similarity":
            # similarity logit: q^T k
            # If normalize=True, this is cosine similarity.
            # If normalize=False, this is dot-product similarity.
            pos_logits = torch.sum(query * positive_key, dim=-1, keepdim=True)

            if self.negative_mode in {"batch", "batch_and_explicit"} and query.shape[0] > 1:
                batch_logits = query @ positive_key.transpose(0, 1)

                self_mask = torch.eye(
                    query.shape[0],
                    device=query.device,
                    dtype=torch.bool,
                )

                batch_neg_logits = batch_logits.masked_select(~self_mask).view(
                    query.shape[0],
                    -1,
                )
                neg_logits.append(batch_neg_logits)

            if (
                    self.negative_mode in {"explicit", "batch_and_explicit"}
                    and explicit_negative_keys is not None
                    and explicit_negative_keys.shape[0] > 0
            ):
                explicit_logits = query @ explicit_negative_keys.transpose(0, 1)
                neg_logits.append(explicit_logits)

        else:
            # distance logit: -||q - k||_2
            # Smaller distance should correspond to larger logit.
            pos_logits = -torch.linalg.norm(
                query - positive_key,
                dim=-1,
                keepdim=True,
            )

            if self.negative_mode in {"batch", "batch_and_explicit"} and query.shape[0] > 1:
                batch_dist = torch.cdist(
                    query,
                    positive_key,
                    p=2,
                )
                batch_logits = -batch_dist

                self_mask = torch.eye(
                    query.shape[0],
                    device=query.device,
                    dtype=torch.bool,
                )

                batch_neg_logits = batch_logits.masked_select(~self_mask).view(
                    query.shape[0],
                    -1,
                )
                neg_logits.append(batch_neg_logits)

            if (
                    self.negative_mode in {"explicit", "batch_and_explicit"}
                    and explicit_negative_keys is not None
                    and explicit_negative_keys.shape[0] > 0
            ):
                explicit_dist = torch.cdist(
                    query,
                    explicit_negative_keys,
                    p=2,
                )
                explicit_logits = -explicit_dist
                neg_logits.append(explicit_logits)

        if not neg_logits:
            raise ValueError("InfoNCE needs at least one negative sample.")

        logits = torch.cat([pos_logits] + neg_logits, dim=1) / self.temperature

        scaled_deficits = logits[:, 1:] - logits[:, :1]
        aggregate_scaled_deficit = torch.logsumexp(scaled_deficits, dim=1)
        aggregate_deficit = aggregate_scaled_deficit * self.temperature.detach()

        # Keep detached gate statistics available to the trainer without
        # changing the scalar-loss interface.
        self.last_gate_stats = {
            "num_negatives": scaled_deficits.shape[1],
            "aggregate_deficit_mean": aggregate_deficit.detach().mean(),
            "aggregate_deficit_min": aggregate_deficit.detach().min(),
            "aggregate_deficit_max": aggregate_deficit.detach().max(),
            "active_fraction": (aggregate_deficit.detach() > 0).float().mean(),
            "soft_lambda_mean": torch.sigmoid(aggregate_scaled_deficit.detach()).mean(),
        }

        if self.gate == "hard":
            # Phi_hard = [tau * logsumexp(d / tau)]_+.
            # At tau=1 this is the LiftedStruct scalarization used in the paper.
            return F.relu(aggregate_deficit).mean()

        labels = torch.zeros(
            query.shape[0],
            device=query.device,
            dtype=torch.long,
        )

        # Phi_soft = tau * log(1 + sum_i exp(d_i / tau)).
        return F.cross_entropy(logits, labels) * self.temperature.detach()

    # def forward(self, query, positive_key, explicit_negative_keys=None):
    #     if query.dim() != 2 or positive_key.dim() != 2:
    #         raise ValueError("query and positive_key must be [N, D].")
    #     if query.shape != positive_key.shape:
    #         raise ValueError(
    #             f"query and positive_key must have the same shape, got {query.shape} and {positive_key.shape}"
    #         )
    #     if explicit_negative_keys is not None and explicit_negative_keys.dim() != 2:
    #         raise ValueError("explicit_negative_keys must be [M, D] when provided.")
    #
    #     query = F.normalize(query, dim=-1)
    #     positive_key = F.normalize(positive_key, dim=-1)
    #     pos_logits = torch.sum(query * positive_key, dim=-1, keepdim=True)
    #
    #     neg_logits = []
    #     if self.negative_mode in {"batch", "batch_and_explicit"} and query.shape[0] > 1:
    #         batch_logits = query @ positive_key.transpose(0, 1)
    #         self_mask = torch.eye(query.shape[0], device=query.device, dtype=torch.bool)
    #         batch_neg_logits = batch_logits.masked_select(~self_mask).view(query.shape[0], -1)
    #         neg_logits.append(batch_neg_logits)
    #
    #     if (
    #         self.negative_mode in {"explicit", "batch_and_explicit"}
    #         and explicit_negative_keys is not None
    #         and explicit_negative_keys.shape[0] > 0
    #     ):
    #         explicit_negative_keys = F.normalize(explicit_negative_keys, dim=-1)
    #         neg_logits.append(query @ explicit_negative_keys.transpose(0, 1))
    #
    #     if not neg_logits:
    #         raise ValueError("InfoNCE needs at least one negative sample.")
    #
    #     logits = torch.cat([pos_logits] + neg_logits, dim=1) / self.temperature
    #     labels = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
    #     return F.cross_entropy(logits, labels)
