import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoder


class Coords5DEncoder(nn.Module):
    """Encode 5D coordinates.

    RC and scale: multiply inputs in [-1, 1] by pi, then apply positional encoding.
    Rotation: recover theta in [-pi, pi] from cosine/sine using atan2,
    then apply harmonic positional encoding."""

    def __init__(self, multires_rc=8, multires_rot=6, multires_scale=4,
                 include_input_rc_scale=True, # Include raw RC and scale inputs in their encodings.
                 log_sampling=True):
        super(Coords5DEncoder, self).__init__()

        # 1. Row/column encoder (2D)
        self.rc_encoder = PositionalEncoder(
            input_dims=2,
            multires=multires_rc,
            include_input=include_input_rc_scale,
            log_sampling=log_sampling
        )

        # 2. Rotation encoder (1D theta)
        # Use input_dims=1 for the scalar angle.
        # Exclude the raw angle to avoid a discontinuity at -pi/pi.
        self.rot_encoder = PositionalEncoder(
            input_dims=1,
            multires=multires_rot,
            include_input=False,
            log_sampling=log_sampling
        )

        # 3. Scale encoder (1D)
        self.scale_encoder = PositionalEncoder(
            input_dims=1,
            multires=multires_scale,
            include_input=include_input_rc_scale,
            log_sampling=log_sampling
        )

        # Compute the total output dimension.
        self.out_dim = (self.rc_encoder.out_dim +
                        self.rot_encoder.out_dim +
                        self.scale_encoder.out_dim)

    def forward(self, coords_5d):
        """Args:
            coords_5d: [..., 5] ordered as [nr, nc, cos, sin, log_s], all in [-1, 1]."""
        original_shape = coords_5d.shape
        coords_flat = coords_5d.reshape(-1, 5)

        # === A. Encode row/column coordinates linearly ===
        # Take the first two dimensions in [-1, 1].
        rc_norm = coords_flat[..., :2]
        # Multiply by pi so the lowest positional-encoding frequency spans a full wavelength.
        rc_encoded = self.rc_encoder(rc_norm * torch.pi)

        # === B. Recover and encode rotation ===
        # Extract cosine and sine.
        cos_t = coords_flat[..., 2]
        sin_t = coords_flat[..., 3]
        # Recover theta in radians within [-pi, pi].
        # atan2 converts the geometric vector in the 5D representation back to an angle.
        theta_rad = torch.atan2(sin_t, cos_t).unsqueeze(-1) # [N, 1]
        # Positional encoding computes harmonics such as sin(theta), sin(2*theta), etc.
        rot_encoded = self.rot_encoder(theta_rad)

        # === C. Encode scale linearly ===
        scale_norm = coords_flat[..., 4:5]
        # Also multiply by pi.
        scale_encoded = self.scale_encoder(scale_norm * torch.pi)

        # === D. Concatenate ===
        coords_encoded = torch.cat([rc_encoded, rot_encoded, scale_encoded], dim=-1)

        # Restore the original leading dimensions.
        output_shape = list(original_shape[:-1]) + [self.out_dim]
        return coords_encoded.reshape(output_shape)
