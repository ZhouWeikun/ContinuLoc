import torch
import numpy as np


class CoordsNormProcessor:
    """Coordinate-space conversion hub.

    Path: Raw(4D) <-> Linear(4D) -> Net(5D)."""

    def __init__(self, sat_dataset_instance):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === 1. Normalized row/column position bounds ===
        self.nrc_min = torch.tensor([sat_dataset_instance.nr2sample_min, sat_dataset_instance.nc2sample_min],
                                    device=self.device, dtype=torch.float32)
        self.nrc_max = torch.tensor([sat_dataset_instance.nr2sample_max, sat_dataset_instance.nc2sample_max],
                                    device=self.device, dtype=torch.float32)
        self.nrc_diff = self.nrc_max - self.nrc_min

        # === 2. Scale bounds: precompute logarithmic limits ===
        s_bnd = sat_dataset_instance.satimgsize_scale_to_ref_m_boundary
        self.scale_log_min = torch.tensor([np.log(max(s_bnd[0], 1e-6))], device=self.device, dtype=torch.float32)
        self.scale_log_max = torch.tensor([np.log(s_bnd[1])], device=self.device, dtype=torch.float32)
        self.scale_log_diff = self.scale_log_max - self.scale_log_min

    # ============================================================
    # Core coordinate conversions
    # ============================================================

    def raw_to_linear(self, coords_raw):
        """Convert physical coordinates to computational coordinates.

        Input: [..., 4] (nr, nc, theta_rad, scale_raw).
        Output: [..., 4] (nr_n, nc_n, theta_lin, scale_lin), all in [-1, 1]."""
        if coords_raw.device != self.device:
            coords_raw = coords_raw.to(self.device)

        # 1. NRC: Linear map
        nrc = coords_raw[..., 0:2]
        nrc_n = 2.0 * (nrc - self.nrc_min) / self.nrc_diff - 1.0

        # 2. Theta: Rad [-pi, pi] -> Linear [-1, 1]
        theta = coords_raw[..., 2:3]
        theta_n = theta / torch.pi  # Assume theta is already in [-pi, pi].

        # 3. Scale: Log map -> Linear [-1, 1]
        scale = coords_raw[..., 3:4]
        # Clamp protect
        scale = torch.clamp(scale, min=torch.exp(self.scale_log_min), max=torch.exp(self.scale_log_max))
        scale_log = torch.log(scale)
        scale_n = 2.0 * (scale_log - self.scale_log_min) / self.scale_log_diff - 1.0

        return torch.cat([nrc_n, theta_n, scale_n], dim=-1)

    def linear_to_net(self, coords_linear):
        """Convert computational coordinates to network coordinates.

        Input: [..., 4] (nr_n, nc_n, theta_lin, scale_lin).
        Output: [..., 5] (nr_n, nc_n, cos, sin, scale_lin)."""
        # Keep the first two dimensions (NRC) and fourth dimension (scale) unchanged.
        nrc_n = coords_linear[..., 0:2]
        scale_n = coords_linear[..., 3:4]

        # Convert the third dimension (theta) to cosine/sine.
        theta_lin = coords_linear[..., 2:3]
        theta_rad = theta_lin * torch.pi
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)

        return torch.cat([nrc_n, cos_t, sin_t, scale_n], dim=-1)

    def linear_to_raw(self, coords_linear):
        """Convert computational coordinates to physical coordinates, typically for visualization."""
        # 1. NRC
        nrc_n = coords_linear[..., 0:2]
        nrc_raw = (nrc_n + 1.0) / 2.0 * self.nrc_diff + self.nrc_min

        # 2. Theta
        theta_n = coords_linear[..., 2:3]
        theta_raw = theta_n * torch.pi

        # 3. Scale
        scale_n = coords_linear[..., 3:4]
        scale_log = (scale_n + 1.0) / 2.0 * self.scale_log_diff + self.scale_log_min
        scale_raw = torch.exp(scale_log)

        return torch.cat([nrc_raw, theta_raw, scale_raw], dim=-1)

    def raw_to_net(self, coords_raw_4d, append_linear_rot=False):
        """Convert raw coordinates to network inputs.

        Input: [..., 4] (nr, nc, theta_rad, scale_ratio).
        Args:
            append_linear_rot: Append normalized theta/pi in [-1, 1].
                False returns 5D coordinates; True returns 6D coordinates
                for mapping the HashGrid Z axis.
        Returns:
            Coordinates of shape [..., 5] or [..., 6]."""
        if coords_raw_4d.device != self.device:
            coords_raw_4d = coords_raw_4d.to(self.device)

        # 1. Normalize positions linearly to [-1, 1].
        nrc_raw = coords_raw_4d[..., 0:2]
        nrc_norm = 2.0 * (nrc_raw - self.nrc_min) / self.nrc_diff - 1.0

        # 2. Embed rotation as cosine/sine in [-1, 1].
        theta = coords_raw_4d[..., 2:3]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # 3. Normalize log scale linearly to [-1, 1].
        scale_raw = coords_raw_4d[..., 3:4]
        scale_raw_clamped = torch.clamp(scale_raw,
                                        min=torch.exp(self.scale_log_min),
                                        max=torch.exp(self.scale_log_max))
        scale_log = torch.log(scale_raw_clamped)
        scale_norm = 2.0 * (scale_log - self.scale_log_min) / self.scale_log_diff - 1.0

        # Base 5D output: [nr, nc, cos, sin, log_s].
        norm_output = torch.cat([nrc_norm, cos_t, sin_t, scale_norm], dim=-1)

        # 4. Optionally append linear rotation in [-1, 1].
        if append_linear_rot:
            # Normalize theta from [-pi, pi] to [-1, 1].
            theta_linear = theta / torch.pi
            # Concatenate to [..., 6].
            norm_output = torch.cat([norm_output, theta_linear], dim=-1)

        return norm_output

    def raw_to_norm(self, coords_raw_4d, append_linear_rot=False):
        """
        Backward-compatible alias for the old util_coords_4d_to_euc5d API.
        Output is the same 5D/6D network coordinate representation as raw_to_net().
        """
        return self.raw_to_net(coords_raw_4d, append_linear_rot=append_linear_rot)


    def net_to_raw(self, coords_norm):
        """
        Input:  [..., 5] or [..., 6]
        Output: [..., 4] -> (nr, nc, theta, scale)
        """
        if coords_norm.device != self.device:
            coords_norm = coords_norm.to(self.device)

        # Use only the first five dimensions for either 5D or 6D inputs.
        coords_basic = coords_norm[..., :5]

        # 1. Recover position.
        nrc_norm = coords_basic[..., 0:2]
        nrc_raw = (nrc_norm + 1.0) / 2.0 * self.nrc_diff + self.nrc_min

        # 2. Recover rotation with atan2.
        cos_t = coords_basic[..., 2]
        sin_t = coords_basic[..., 3]
        theta_raw = torch.atan2(sin_t, cos_t).unsqueeze(-1)

        # 3. Recover scale.
        scale_norm = coords_basic[..., 4:5]
        scale_log = (scale_norm + 1.0) / 2.0 * self.scale_log_diff + self.scale_log_min
        scale_raw = torch.exp(scale_log)

        return torch.cat([nrc_raw, theta_raw, scale_raw], dim=-1)

    def norm_to_raw(self, coords_norm):
        """Backward-compatible alias for net_to_raw()."""
        return self.net_to_raw(coords_norm)

    # ============================================================
    # Helpers: sigma conversion and weight computation in linear space
    # ============================================================

    def get_linear_sigmas(self, gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale):
        """Convert physical sigmas to linear-space sigmas for the sampler."""
        # NRC
        avg_nrc_span = (self.nrc_diff[0] + self.nrc_diff[1]) / 2.0
        sigma_lin_nrc = gs_sigma_nrc * (2.0 / (avg_nrc_span + 1e-6))

        # Theta (Rad -> Linear [-1, 1])
        # Range = 2*pi -> 2.0
        sigma_lin_theta = gs_sigma_radrot * (2.0 / (2 * torch.pi))

        # Scale (Log -> Linear [-1, 1])
        sigma_lin_scale = gs_sigma_scale * (2.0 / (self.scale_log_diff + 1e-6))

        # Return in [r, c, theta, s] order.
        sigma_lin_nrc_val = float(sigma_lin_nrc)
        sigma_lin_theta_val = float(sigma_lin_theta)
        sigma_lin_scale_val = float(sigma_lin_scale)
        return [sigma_lin_nrc_val, sigma_lin_nrc_val, sigma_lin_theta_val, sigma_lin_scale_val]

    def get_normalized_sigmas(self, gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale=None):
        """
        Backward-compatible sigma helper for the old 5D/6D normalized coordinate API.
        """
        avg_nrc_span = (self.nrc_diff[0] + self.nrc_diff[1]) / 2.0
        sigma_n_nrc = gs_sigma_nrc * (2.0 / (avg_nrc_span + 1e-6))

        sigma_n_scale = None
        if gs_sigma_scale is not None:
            sigma_n_scale = gs_sigma_scale * (2.0 / (self.scale_log_diff + 1e-6))

        sigma_n_rot = gs_sigma_radrot * (2.0 / (2 * torch.pi))

        return {
            'nrc': sigma_n_nrc,
            'scale': sigma_n_scale,
            'rot_linear': sigma_n_rot,
            'rot_rad': gs_sigma_radrot,
        }

    def compute_weight_matrix_from_norm(
        self,
        q_norm,
        ref_norm,
        gs_sigma_nrc,
        gs_sigma_radrot,
        gs_sigma_scale=None,
        use_exact_rot_dist=True,
    ):
        """
        Backward-compatible weight computation for 5D/6D normalized network coords.
        """
        sigmas = self.get_normalized_sigmas(gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale)

        sigma_n_nrc = sigmas['nrc']
        sigma_n_scale = sigmas['scale']
        sigma_rot_phys = sigmas['rot_rad']

        is_batched = q_norm.dim() == 3
        if not is_batched:
            q_norm = q_norm.unsqueeze(0)
            ref_norm = ref_norm.unsqueeze(0)

        q = q_norm.unsqueeze(2)
        ref = ref_norm.unsqueeze(1)
        input_dim = q_norm.shape[-1]

        diff_nrc = q[..., 0:2] - ref[..., 0:2]
        energy_nrc = torch.sum(diff_nrc ** 2, dim=-1) / (2 * sigma_n_nrc ** 2)

        energy_scale = 0.0
        if sigma_n_scale is not None:
            diff_scale = q[..., 4:5] - ref[..., 4:5]
            energy_scale = torch.sum(diff_scale ** 2, dim=-1) / (2 * sigma_n_scale ** 2)

        if input_dim == 6:
            theta_q = q[..., 5] * torch.pi
            theta_ref = ref[..., 5] * torch.pi
            diff = torch.abs(theta_q - theta_ref)
            diff = torch.min(diff, 2 * torch.pi - diff)
            energy_rot = (diff ** 2) / (2 * sigma_rot_phys ** 2)
        else:
            cos_sin_q = q[..., 2:4]
            cos_sin_ref = ref[..., 2:4]
            if use_exact_rot_dist:
                dot = torch.sum(cos_sin_q * cos_sin_ref, dim=-1)
                dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
                dist_rot = torch.acos(dot)
                energy_rot = (dist_rot ** 2) / (2 * sigma_rot_phys ** 2)
            else:
                diff_rot = cos_sin_q - cos_sin_ref
                dist_sq_rot = torch.sum(diff_rot ** 2, dim=-1)
                energy_rot = dist_sq_rot / (2 * sigma_rot_phys ** 2)

        weights = torch.exp(-(energy_nrc + energy_rot + energy_scale))
        if not is_batched:
            weights = weights.squeeze(0)
        return weights

    def compute_weight_matrix_linear(self, q_lin, ref_lin, norm_sigmas,ignore_dim=None):
        """Compute weights efficiently in 4D linear space.

        Args:
            q_lin: Query coordinates [B, M, 4].
            ref_lin: Reference coordinates [B, N, 4].
            norm_sigmas: Normalized [sigma_r, sigma_c, sigma_theta, sigma_s].
            ignore_dim: Index, list of indices, or None. Use 2 to ignore
                rotation, or [2, 3] to ignore rotation and scale."""
        # 1. Handle dimensions and broadcasting.
        if q_lin.dim() == 2: q_lin = q_lin.unsqueeze(0)
        if ref_lin.dim() == 2: ref_lin = ref_lin.unsqueeze(0)

        q = q_lin.unsqueeze(2)  # [B, M, 1, 4]
        ref = ref_lin.unsqueeze(1)  # [B, 1, N, 4]

        # 2. Prepare sigmas.
        sigmas = torch.as_tensor(norm_sigmas, device=q.device, dtype=q.dtype).view(1, 1, 1, 4)

        # 3. Compute linear differences.
        delta = q - ref  # [B, M, N, 4]

        # 4. Compute squared distances.
        # Square every dimension first; theta still needs a periodic correction.
        dist_sq = delta ** 2

        # 5. Correct theta at index 2.
        # d_cyclic = min(|d|, 2 - |d|)
        d_theta_raw = torch.abs(delta[..., 2])
        d_theta_cyclic = torch.min(d_theta_raw, 2.0 - d_theta_raw)
        # Overwrite the value at index 2.
        dist_sq[..., 2] = d_theta_cyclic ** 2

        # =========================================================
        # Dimension masking
        # =========================================================
        if ignore_dim is not None:
            # 1. Normalize the selection to a list.
            if isinstance(ignore_dim, int):
                dims_to_ignore = [ignore_dim]
            else:
                dims_to_ignore = ignore_dim

            # 2. Create a mask, e.g. [1, 1, 1, 1] -> [1, 1, 0, 1].
            # Start with all ones (retain every dimension).
            mask = torch.ones(4, device=q.device, dtype=q.dtype)

            # Set ignored dimensions to zero.
            mask[dims_to_ignore] = 0.0

            # 3. Broadcast and apply the mask.
            # [4] -> [1, 1, 1, 4]
            mask = mask.view(1, 1, 1, 4)

            # Force distances in ignored dimensions to zero.
            # Since 0 / sigma^2 = 0, ignored dimensions do not contribute to the sum.
            dist_sq = dist_sq * mask

        # 6. Compute the weighted sum.
        # E = sum( d^2 / (2 * sigma^2) )
        energy = torch.sum(dist_sq / (2 * sigmas ** 2), dim=-1)  # [B, M, N]

        return torch.exp(-energy)
