import torch
import torch.nn.functional as TF
import torch.nn.functional as F

def _compute_metric_from_query_and_points(
        self,
        query_feats,
        ref_points,
        temperature=10.0,
        metric='possibility',
        coord_space='raw',
        chunk_size=2048,
        feat_type='projector',
    ):
        """Compute distances or probabilities between query features and reference points.

        Args:
            query_feats: Query features [B, C].
            ref_points: Coordinates [N, 4] or [B, N, 4].
            temperature: Softmax temperature, used only for metric='possibility'.
            metric: 'dist' returns distances; 'possibility' returns softmax
                probabilities. Legacy 'euclidean'/'l2' aliases mean 'possibility'.
            coord_space: 'raw' (nr, nc, rot, scale) or 'linear'.
            chunk_size: Number of candidates per chunk.
            feat_type: 'ingp' uses INGP feature distances; 'projector' uses
                projected feature distances.
        Returns:
            Distances or probabilities [B, N]."""
        metric = metric.lower()
        if metric in ('euclidean', 'l2'):
            metric = 'possibility'
        if metric not in ('dist', 'possibility'):
            raise ValueError(f"metric must be 'dist' or 'possibility', got {metric}")

        feat_type = feat_type.lower()
        if feat_type not in ('ingp', 'projector'):
            raise ValueError(f"feat_type must be 'ingp' or 'projector', got {feat_type}")

        if coord_space not in ('raw', 'linear'):
            raise ValueError(f"coord_space must be 'raw' or 'linear', got {coord_space}")

        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points must be 2D or 3D, got {ref_points.dim()}D")

        # Treat [N, 1, 4] as candidates for a single batch.
        if ref_points.dim() == 3 and ref_points.shape[0] != query_feats.shape[0]:
            ref_points = ref_points.reshape(-1, ref_points.shape[-1])
        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points shape not supported after reshape: {ref_points.shape}")

        # Normalize coordinates.
        if coord_space == 'linear':
            ref_points_raw = self.coord_normer.linear_to_raw(ref_points.reshape(-1, 4))
        else:
            ref_points_raw = ref_points.reshape(-1, 4)

        # Chunk candidates to limit grid_mlp activation memory.
        if chunk_size is None or chunk_size <= 0:
            chunk_size = ref_points_raw.shape[0] if ref_points.dim() == 2 else ref_points.shape[1]

        # Preprocess query features.
        if feat_type == 'ingp':
            query_emb = TF.normalize(query_feats, dim=-1)
        else:
            with torch.no_grad():
                query_emb = self.projector(query_feats)

        dist_chunks = []
        if ref_points.dim() == 2:
            for start in range(0, ref_points_raw.shape[0], chunk_size):
                end = min(start + chunk_size, ref_points_raw.shape[0])
                coords_chunk = ref_points_raw[start:end]
                if feat_type == 'ingp':
                    feats_ref = self._get_feats_fm_INGP(coords_chunk, coord_mode='raw')
                    feats_ref = TF.normalize(feats_ref, dim=-1)
                else:
                    coords_6d = self.coord_normer.raw_to_net(coords_chunk, append_linear_rot=True)
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feats_grid_raw = self._get_feats_fm_grid(grid_input)
                    coords_encoded_stage2 = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_ingp = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                    feats_ref = self.projector(TF.normalize(feats_ingp, dim=-1))
                if feat_type == 'ingp':
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                else:
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                dist_chunks.append(dist_chunk)
        else:
            B = ref_points.shape[0]
            for start in range(0, ref_points.shape[1], chunk_size):
                end = min(start + chunk_size, ref_points.shape[1])
                coords_chunk = ref_points[:, start:end, :]  # [B, chunk, 4]
                coords_chunk_flat = coords_chunk.reshape(-1, 4)
                if feat_type == 'ingp':
                    feats_ref = self._get_feats_fm_INGP(coords_chunk_flat, coord_mode='raw')
                    feats_ref = TF.normalize(feats_ref, dim=-1).view(B, -1, feats_ref.shape[-1])
                else:
                    coords_6d = self.coord_normer.raw_to_net(coords_chunk_flat, append_linear_rot=True)
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feats_grid_raw = self._get_feats_fm_grid(grid_input)
                    coords_encoded_stage2 = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_ingp = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                    feats_ref = self.projector(TF.normalize(feats_ingp, dim=-1))
                    feats_ref = feats_ref.view(B, -1, feats_ref.shape[-1])
                if feat_type == 'ingp':
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                else:
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                dist_chunks.append(dist_chunk)

        dist = torch.cat(dist_chunks, dim=1)

        if metric == 'dist':
            return dist

        logit = -temperature * dist
        prob = F.softmax(logit, dim=-1)
        return prob

def _compute_metric_from_ingp(
        self,
        query_feats,
        ref_points,
        coord_space='raw',
        chunk_size=4096,
        metric='sim',
    ):
        """Compute similarity/distance between INGP and visual features in memory-bounded chunks.

        Returns:
            Similarities or distances [B, N]."""
        if coord_space not in ('raw', 'linear'):
            raise ValueError(f"coord_space must be 'raw' or 'linear', got {coord_space}")

        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points must be 2D or 3D, got {ref_points.dim()}D")

        # Flatten layouts such as [N, 1, 4] that do not match the batch dimension.
        if ref_points.dim() == 3 and ref_points.shape[0] != query_feats.shape[0]:
            ref_points = ref_points.reshape(-1, ref_points.shape[-1])

        q_norm = TF.normalize(query_feats, dim=-1)
        metric = metric.lower()
        if metric not in ('sim', 'dist'):
            raise ValueError(f"metric must be 'sim' or 'dist', got {metric}")

        if chunk_size is None or chunk_size <= 0:
            chunk_size = ref_points.shape[0] if ref_points.dim() == 2 else ref_points.shape[1]

        metric_chunks = []
        if ref_points.dim() == 2:
            # Shared candidates [N, 4]
            for start in range(0, ref_points.shape[0], chunk_size):
                end = min(start + chunk_size, ref_points.shape[0])
                coords_chunk = ref_points[start:end]
                coord_mode = 'raw' if coord_space == 'raw' else 'linear'
                feats_chunk = self._get_feats_fm_INGP(coords_chunk, coord_mode=coord_mode)
                feats_chunk = TF.normalize(feats_chunk, dim=-1)
                if metric == 'sim':
                    # [B, C] x [C, chunk] -> [B, chunk]
                    metric_chunk = torch.matmul(q_norm, feats_chunk.t())
                else:
                    metric_chunk = torch.norm(q_norm.unsqueeze(1) - feats_chunk, p=2, dim=-1)
                metric_chunks.append(metric_chunk)
        elif ref_points.dim() == 3:
            # Per-query candidates [B, N, 4]
            B = ref_points.shape[0]
            for start in range(0, ref_points.shape[1], chunk_size):
                end = min(start + chunk_size, ref_points.shape[1])
                coords_chunk = ref_points[:, start:end, :]  # [B, chunk, 4]
                coord_mode = 'raw' if coord_space == 'raw' else 'linear'
                feats_chunk = self._get_feats_fm_INGP(coords_chunk.reshape(-1, 4), coord_mode=coord_mode)
                feats_chunk = feats_chunk.view(B, -1, feats_chunk.shape[-1])
                feats_chunk = TF.normalize(feats_chunk, dim=-1)
                if metric == 'sim':
                    metric_chunk = torch.sum(q_norm.unsqueeze(1) * feats_chunk, dim=-1)  # [B, chunk]
                else:
                    metric_chunk = torch.norm(q_norm.unsqueeze(1) - feats_chunk, p=2, dim=-1)
                metric_chunks.append(metric_chunk)
        else:
            raise ValueError(f"ref_points shape not supported: {ref_points.shape}")

        metric_out = torch.cat(metric_chunks, dim=1)
        return metric_out

def _get_feats_fm_INGP(self, coords, coord_mode='raw'):
        """Convert supported coordinate formats into normalized features.

        Matches the implementation in stage3_project_integrateRot_classify.py.

        Args:
            coords: Coordinate tensor [N, 4], [N, 5], or [N, 6].
            coord_mode: Input representation:
                'raw': [r, c, theta, s], physical dataset coordinates.
                'linear': [r_n, c_n, t_lin, s_n], linear sampler coordinates.
                'net_5d': [r_n, c_n, cos, sin, s_n], passed through directly.
                'net_6d': Network coordinates plus the linear angle, an
                    intermediate processor representation.

        Returns:
            L2-normalized features [N, C]."""

        # =========================================================
        # 1. Convert to 6D, matching stage3_project_integrateRot_classify.
        # =========================================================
        if coord_mode == 'raw':
            # Use raw_to_net with the linear rotation appended.
            coords_6d = self.coord_normer.raw_to_net(coords, append_linear_rot=True)
        elif coord_mode == 'linear':
            coords_net = self.coord_normer.linear_to_net(coords)
            theta_lin = coords[..., 2:3]  # Theta in linear space is already normalized.
            coords_6d = torch.cat([coords_net, theta_lin], dim=-1)
        elif coord_mode == 'net_5d':
            theta_lin = torch.atan2(coords[..., 3:4], coords[..., 2:3]) / torch.pi
            coords_6d = torch.cat([coords, theta_lin], dim=-1)
        elif coord_mode == 'net_6d':
            coords_6d = coords
        else:
            raise ValueError(f"Unknown coord_mode: {coord_mode}")

        # =========================================================
        # 2. Construct submodule inputs, matching stage3_project_integrateRot_classify.
        # =========================================================
        grid_input = torch.cat([coords_6d[..., :2], coords_6d[..., -1:]], dim=-1)  # [nr, nc, theta_lin]
        mlp_input = coords_6d[..., :5]  # [nr, nc, cos, sin, log_s]

        # =========================================================
        # 3. Run the forward pass.
        # =========================================================

        # (A) Query HashGrid (Backbone)
        feat_raw = self._get_feats_fm_grid(grid_input)

        # (B) Positional encoding for MLP conditioning
        pos_enc = self.pos_encoder_grid(mlp_input)

        # (C) Tiny MLP (Decoder)
        feat_out = self.grid_mlp(feat_raw, pos_enc)

        # 4. Apply L2 normalization for metric learning.
        return torch.nn.functional.normalize(feat_out, dim=-1, p=2)
