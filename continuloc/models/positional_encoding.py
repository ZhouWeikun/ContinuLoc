import torch
import torch.nn as nn

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoder(nn.Module):
    def __init__(self, input_dims, multires, include_input=True, log_sampling=True):
        super().__init__()
        self.input_dims = input_dims
        self.multires = multires
        self.include_input = include_input

        # Compute the output dimension.
        # include_input=True: input_dims + 2 * input_dims * multires.
        # include_input=False: 2 * input_dims * multires.
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dims
        self.out_dim += 2 * input_dims * multires

        # Precompute frequencies.
        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., multires - 1, steps=multires)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** multires, steps=multires)

        # Register as a buffer so .to(device) moves frequencies automatically.
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        """
        x: [..., input_dims]
        """
        # x is in radians; no extra pi normalization is needed because sin has period 2*pi.
        # freq_bands: [L]

        # Expand for broadcasting: [..., 1] * [L] -> [..., L].
        # x.unsqueeze(-1) shape: [..., input_dims, 1]
        embed = x.unsqueeze(-1) * self.freq_bands

        # [..., input_dims, L] -> [..., input_dims * L]
        embed = embed.flatten(start_dim=-2)

        # Compute sine and cosine.
        # Output: [..., input_dims * L * 2].
        result = torch.cat([torch.sin(embed), torch.cos(embed)], dim=-1)

        if self.include_input:
            result = torch.cat([x, result], dim=-1)

        return result


def encode_4d_coords(coords, rc_encoder, rot_endcoder, scale_encoder):
    """Legacy 4D coordinate encoding for compatibility.

    Input: [nr, nc, rotation_rad, scale_ratio].
    Returns encoded coordinates."""
    if len(coords.shape) == 3:
        b, n, _ = coords.shape
        coords = coords.reshape(-1, 4)
    else:
        b = 0
    rcs_encoded = rc_encoder(coords[:, :2])
    rots = torch.stack([torch.sin(coords[:, 2]), torch.cos(coords[:, 2])]).T
    rots_encoded = rot_endcoder(rots)
    scales_encoded = scale_encoder(coords[:, -1:])
    coords_encoded = torch.concatenate(
        [rcs_encoded, rots_encoded, scales_encoded], dim=-1)

    if b > 0:
        coords_encoded = coords_encoded.reshape(b, n, -1)

    return coords_encoded

