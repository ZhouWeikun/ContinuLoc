# trainer_depends/datasets/util_SubspaceSampler.py

import numpy as np
import torch
from typing import Tuple, Optional, Union, List


class SubspaceSampler:
    """Partition and sample a 4D coordinate space.

    Coarse cells define subspaces/classes. Fine grids within each coarse cell
    provide uniform sampling. Coordinates are [nr, nc, rot, scale]: normalized
    positions, periodic rotation in radians within [-pi, pi], and scale ratio."""

    def __init__(
            self,
            sat_dataset,
            n_coarse: Tuple[int, int, int, int] = (8, 8, 12, 3),
            n_fine_per_coarse: Tuple[int, int, int, int] = (1, 1, 1, 1),
    ):
        """Args:
            sat_dataset: SatDataset providing coordinate bounds.
            n_coarse: Coarse counts (n_nr, n_nc, n_rot, n_scale).
            n_fine_per_coarse: Fine subdivision counts within each coarse subspace."""
        self.sat_dataset = sat_dataset
        self.n_coarse = np.array(n_coarse, dtype=np.int32)
        self.n_fine_per_coarse = np.array(n_fine_per_coarse, dtype=np.int32)

        # Total fine subdivision counts
        self.n_fine_total = self.n_coarse * self.n_fine_per_coarse

        # Number of subspaces/classification categories
        self.n_subspaces = int(np.prod(self.n_coarse))

        # Number of fine cells per coarse subspace
        self.n_fine_cells_per_subspace = int(np.prod(self.n_fine_per_coarse))

        # Get each dimension's bounds.
        self._init_coord_ranges()

        # Compute each dimension's step size.
        self._compute_bin_sizes()

        # Lazily initialized GPU cache
        self._gpu_cache = {}

    def _get_gpu_cache(self, device: torch.device) -> dict:
        """Get or lazily initialize the GPU cache.

        Precompute all_coarse_starts [n_subspaces, 4], fine_offsets
        [n_fine_cells_per_subspace, 4], and fine_bin_sizes [4]."""
        device_key = str(device)
        if device_key not in self._gpu_cache:
            # Base constants
            coord_mins = torch.from_numpy(self.coord_ranges[:, 0]).to(device=device, dtype=torch.float32)
            coarse_bin_sizes = torch.from_numpy(self.coarse_bin_sizes).to(device=device, dtype=torch.float32)
            fine_bin_sizes = torch.from_numpy(self.fine_bin_sizes_in_coarse).to(device=device, dtype=torch.float32)

            # Precompute the starting coordinates of coarse subspaces.
            all_coarse_starts = self._precompute_all_coarse_starts(coord_mins, coarse_bin_sizes, device)

            # Precompute fine-cell offsets relative to each coarse subspace origin.
            fine_offsets = self._precompute_fine_offsets(fine_bin_sizes, device)

            self._gpu_cache[device_key] = {
                'coord_ranges': torch.from_numpy(self.coord_ranges).to(device=device, dtype=torch.float32),
                'coarse_bin_sizes': coarse_bin_sizes,
                'coord_mins': coord_mins,
                'all_coarse_starts': all_coarse_starts,  # [n_subspaces, 4]
                'fine_offsets': fine_offsets,            # [n_fine_cells_per_subspace, 4]
                'fine_bin_sizes': fine_bin_sizes,        # [4]
            }
        return self._gpu_cache[device_key]

    def _precompute_all_coarse_starts(self, coord_mins: torch.Tensor,
                                       coarse_bin_sizes: torch.Tensor,
                                       device: torch.device) -> torch.Tensor:
        """Precompute all coarse subspace origins.

        Returns:
            Starting coordinates [n_subspaces, 4]."""
        indices = [torch.arange(n, device=device) for n in self.n_coarse]
        grid = torch.meshgrid(*indices, indexing='ij')
        multi_indices = torch.stack([g.flatten() for g in grid], dim=1).float()
        all_starts = coord_mins + multi_indices * coarse_bin_sizes
        return all_starts

    def _precompute_fine_offsets(self, fine_bin_sizes: torch.Tensor,
                                  device: torch.device) -> torch.Tensor:
        """Precompute fine-cell offsets within a coarse subspace.

        Returns:
            Relative offsets [n_fine_cells_per_subspace, 4]."""
        indices = [torch.arange(n, device=device) for n in self.n_fine_per_coarse]
        grid = torch.meshgrid(*indices, indexing='ij')
        multi_indices = torch.stack([g.flatten() for g in grid], dim=1).float()
        fine_offsets = multi_indices * fine_bin_sizes
        return fine_offsets

    def _init_coord_ranges(self):
        """Initialize coordinate bounds for every dimension."""
        sd = self.sat_dataset

        # Spatial coordinate bounds
        self.nr_range = np.array([sd.nr2sample_min, sd.nr2sample_max], dtype=np.float32)
        self.nc_range = np.array([sd.nc2sample_min, sd.nc2sample_max], dtype=np.float32)

        # Periodic rotation bounds [-pi, pi]
        self.rot_range = np.array([-np.pi, np.pi], dtype=np.float32)

        # Scale bounds
        self.scale_range = sd.satimgsize_scale_to_refm_boundary.astype(np.float32)

        # Combine bounds into a [4, 2] array.
        self.coord_ranges = np.stack([
            self.nr_range,
            self.nc_range,
            self.rot_range,
            self.scale_range
        ], axis=0)  # [4, 2]

    def _compute_bin_sizes(self):
        """Compute coarse and fine subdivision step sizes."""
        ranges = self.coord_ranges[:, 1] - self.coord_ranges[:, 0]  # [4]

        # Coarse step sizes, explicitly cast to float32
        self.coarse_bin_sizes = (ranges / self.n_coarse).astype(np.float32)  # [4]

        # Fine step sizes relative to the complete range
        self.fine_bin_sizes = (ranges / self.n_fine_total).astype(np.float32)  # [4]

        # Fine step sizes within one coarse cell
        self.fine_bin_sizes_in_coarse = (self.coarse_bin_sizes / self.n_fine_per_coarse).astype(np.float32)  # [4]

    # ==================== Periodicity ====================

    def _wrap_rotation(self, rot: np.ndarray) -> np.ndarray:
        """Wrap rotation angles to [-pi, pi]."""
        return ((rot + np.pi) % (2 * np.pi)) - np.pi

    def _wrap_rotation_torch(self, rot: torch.Tensor) -> torch.Tensor:
        """Wrap rotation angles to [-pi, pi] using PyTorch."""
        return ((rot + np.pi) % (2 * np.pi)) - np.pi

    def _build_normalized_grid_centers_torch(
            self,
            grid_dims: Tuple[int, int, int, int],
            device: torch.device
    ) -> torch.Tensor:
        """Generate regular grid centers in [0, 1].

        Share grid-center construction across local refinement samplers
        instead of repeating meshgrid and flatten operations."""
        grids = []
        for dim_size in grid_dims:
            if dim_size <= 0:
                raise ValueError(f"Each grid dim must be > 0, got {grid_dims}")
            grid = (torch.arange(dim_size, device=device, dtype=torch.float32) + 0.5) / dim_size
            grids.append(grid)

        mesh = torch.meshgrid(*grids, indexing='ij')
        return torch.stack([m.flatten() for m in mesh], dim=1)

    def _ensure_size_tensor_1d(
            self,
            size_value: Union[np.ndarray, torch.Tensor, Tuple, List],
            device: torch.device
    ) -> torch.Tensor:
        """Normalize size inputs to a device-local float tensor of shape [4].

        This adapts input representations without changing numeric semantics."""
        if isinstance(size_value, (tuple, list)):
            size_tensor = torch.tensor(size_value, dtype=torch.float32, device=device)
        elif isinstance(size_value, np.ndarray):
            size_tensor = torch.from_numpy(size_value).float().to(device)
        elif torch.is_tensor(size_value):
            size_tensor = size_value.to(device=device, dtype=torch.float32)
        else:
            size_tensor = torch.tensor(size_value, dtype=torch.float32, device=device)

        if size_tensor.ndim > 1:
            size_tensor = size_tensor.flatten()[:4]

        if size_tensor.shape[0] != 4:
            raise ValueError(f"Expected size tensor shape [4], got {tuple(size_tensor.shape)}")

        return size_tensor

    # ==================== Coordinates to indices ====================

    def coords_to_coarse_indices(
            self,
            coords_4d: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert 4D coordinates to scalar coarse-subspace indices.

        Args:
            coords_4d: Coordinates [..., 4] with arbitrary leading batch dimensions.
        Returns:
            Indices [...] in [0, n_subspaces - 1]."""
        is_tensor = torch.is_tensor(coords_4d)
        if is_tensor:
            device = coords_4d.device
            coords = coords_4d.cpu().numpy()
        else:
            coords = np.asarray(coords_4d)

        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 4)  # [N, 4]

        # Compute bin indices for each dimension.
        bin_indices = self._coords_to_bin_indices(coords_flat, self.n_coarse)  # [N, 4]

        # Flatten indices in row-major order.
        indices = self._multi_index_to_flat(bin_indices, self.n_coarse)  # [N]

        indices = indices.reshape(original_shape)

        if is_tensor:
            return torch.from_numpy(indices).to(device).long()
        return indices

    def coords_to_fine_indices(
            self,
            coords_4d: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Convert 4D coordinates to coarse/fine index pairs.

        Returns:
            coarse_indices: Coarse-subspace indices [...].
            fine_indices: Fine-cell indices within each coarse subspace [...]."""
        is_tensor = torch.is_tensor(coords_4d)
        if is_tensor:
            device = coords_4d.device
            coords = coords_4d.cpu().numpy()
        else:
            coords = np.asarray(coords_4d)

        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 4)

        # Compute global fine-bin indices.
        fine_global_indices = self._coords_to_bin_indices(coords_flat, self.n_fine_total)  # [N, 4]

        # Decompose into coarse and local fine indices.
        coarse_bin_indices = fine_global_indices // self.n_fine_per_coarse  # [N, 4]
        fine_bin_indices = fine_global_indices % self.n_fine_per_coarse  # [N, 4]

        # Convert to scalar flat indices.
        coarse_indices = self._multi_index_to_flat(coarse_bin_indices, self.n_coarse)
        fine_indices = self._multi_index_to_flat(fine_bin_indices, self.n_fine_per_coarse)

        coarse_indices = coarse_indices.reshape(original_shape)
        fine_indices = fine_indices.reshape(original_shape)

        if is_tensor:
            return (torch.from_numpy(coarse_indices).to(device).long(),
                    torch.from_numpy(fine_indices).to(device).long())
        return coarse_indices, fine_indices

    def _coords_to_bin_indices(
            self,
            coords: np.ndarray,
            n_bins: np.ndarray
    ) -> np.ndarray:
        """Convert coordinates to per-dimension bin indices.

        Args:
            coords: Coordinates [N, 4].
            n_bins: Bin counts [4].
        Returns:
            Bin indices [N, 4]."""
        # Handle rotation periodicity.
        coords = coords.copy()
        coords[:, 2] = self._wrap_rotation(coords[:, 2])

        # Normalize to [0, 1].
        normalized = (coords - self.coord_ranges[:, 0]) / (self.coord_ranges[:, 1] - self.coord_ranges[:, 0])

        # Convert to bin indices and clip.
        bin_indices = (normalized * n_bins).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        return bin_indices

    def _multi_index_to_flat(
            self,
            multi_indices: np.ndarray,
            shape: np.ndarray
    ) -> np.ndarray:
        """Flatten multidimensional indices in row-major order.

        Args:
            multi_indices: Per-dimension indices [N, 4].
            shape: Dimension sizes [4].
        Returns:
            Flat indices [N]."""
        strides = np.array([
            shape[1] * shape[2] * shape[3],
            shape[2] * shape[3],
            shape[3],
            1
        ], dtype=np.int64)

        return (multi_indices * strides).sum(axis=-1)

    def _flat_to_multi_index(
            self,
            flat_indices: np.ndarray,
            shape: np.ndarray
    ) -> np.ndarray:
        """Expand flat indices into multidimensional indices.

        Args:
            flat_indices: Flat indices [N].
            shape: Dimension sizes [4].
        Returns:
            Per-dimension indices [N, 4]."""
        flat_indices = np.asarray(flat_indices).flatten()
        multi_indices = np.zeros((len(flat_indices), 4), dtype=np.int32)
        remainder = flat_indices.copy()

        for i in range(4):
            divisor = int(np.prod(shape[i + 1:])) if i < 3 else 1
            multi_indices[:, i] = remainder // divisor
            remainder = remainder % divisor

        return multi_indices

    def flat_to_multi_index_flexible(
            self,
            flat_indices: Union[np.ndarray, torch.Tensor],
            shape: Optional[Union[np.ndarray, torch.Tensor, Tuple[int, int, int, int]]] = None,
            use_fine: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """Expand flat indices, supporting tensors and custom shapes.

        Args:
            flat_indices: Flat indices [...].
            shape: Optional (nr, nc, rot, scale) tuple/list/array. None uses
                n_coarse or n_fine_total according to use_fine.
            use_fine: Select fine subdivisions when shape is None; default False.
        Returns:
            Indices [..., 4] ordered as [nr_idx, nc_idx, rot_idx, scale_idx].

        Examples:
            >>> # Default coarse subdivision
            >>> labels_multi = sampler.flat_to_multi_index_flexible(labels_flat)
            >>> # Fine subdivision
            >>> labels_multi = sampler.flat_to_multi_index_flexible(labels_flat, use_fine=True)
            >>> # Custom shape
            >>> labels_multi = sampler.flat_to_multi_index_flexible(labels_flat, shape=(10, 10, 6, 2))"""
        is_tensor = torch.is_tensor(flat_indices)

        # Determine the shape.
        if shape is None:
            # Select automatically based on use_fine and available fine subdivisions.
            if use_fine and self.n_fine_cells_per_subspace > 1:
                shape_to_use = self.n_fine_total
            else:
                shape_to_use = self.n_coarse
        else:
            # Use the custom shape.
            if torch.is_tensor(shape):
                shape_to_use = shape.cpu().numpy()
            elif isinstance(shape, (list, tuple)):
                shape_to_use = np.array(shape, dtype=np.int32)
            else:
                shape_to_use = np.asarray(shape, dtype=np.int32)

        # Convert to NumPy for computation.
        if is_tensor:
            device = flat_indices.device
            indices = flat_indices.cpu().numpy()
        else:
            indices = np.asarray(flat_indices)

        original_shape = indices.shape

        # Call the underlying NumPy implementation.
        multi_indices = self._flat_to_multi_index(indices, shape_to_use)
        multi_indices = multi_indices.reshape(*original_shape, 4)

        # Convert back to a tensor if the input was a tensor.
        if is_tensor:
            return torch.from_numpy(multi_indices).to(device).long()
        return multi_indices

    def coarse_indices_to_multi(
            self,
            flat_indices: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Expand flat coarse-subspace indices.

        Args:
            flat_indices: Flat coarse indices [...].
        Returns:
            Indices [..., 4] ordered as [nr_idx, nc_idx, rot_idx, scale_idx]."""
        is_tensor = torch.is_tensor(flat_indices)
        if is_tensor:
            device = flat_indices.device
            indices = flat_indices.cpu().numpy()
        else:
            indices = np.asarray(flat_indices)

        original_shape = indices.shape
        multi_indices = self._flat_to_multi_index(indices, self.n_coarse)
        multi_indices = multi_indices.reshape(*original_shape, 4)

        if is_tensor:
            return torch.from_numpy(multi_indices).to(device).long()
        return multi_indices

    # ==================== Indices to coordinates ====================

    def coarse_index_to_center(
            self,
            coarse_indices: Union[int, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Get coarse-subspace center coordinates.

        Args:
            coarse_indices: Scalar or array of coarse-subspace indices.
        Returns:
            Centers [N, 4] or [4]."""
        is_scalar = np.isscalar(coarse_indices)
        if is_scalar:
            indices = np.array([coarse_indices])
        elif torch.is_tensor(coarse_indices):
            indices = coarse_indices.cpu().numpy().flatten()
        else:
            indices = np.asarray(coarse_indices).flatten()

        # Expand to multidimensional indices.
        multi_indices = self._flat_to_multi_index(indices, self.n_coarse)  # [N, 4]

        # Compute center coordinates.
        centers = (self.coord_ranges[:, 0] +
                   (multi_indices + 0.5) * self.coarse_bin_sizes)  # [N, 4]

        if is_scalar:
            return centers[0]
        return centers.astype(np.float32)

    # ==================== Adjacent subspaces ====================

    def get_adjacent_subspaces(
            self,
            coarse_idx: int,
            include_diagonal: bool = False
    ) -> np.ndarray:
        """Get neighboring subspace indices, accounting for rotation periodicity.

        Args:
            coarse_idx: Coarse-subspace index.
            include_diagonal: Include diagonally adjacent cells.
        Returns:
            Array of adjacent subspace indices."""
        # Expand to multidimensional indices.
        multi_idx = self._flat_to_multi_index(np.array([coarse_idx]), self.n_coarse)[0]

        adjacent = []

        # Define offsets.
        if include_diagonal:
            offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * 4)).T.reshape(-1, 4)
            offsets = offsets[~np.all(offsets == 0, axis=1)]  # Remove the zero offset.
        else:
            offsets = np.array([
                [-1, 0, 0, 0], [1, 0, 0, 0],
                [0, -1, 0, 0], [0, 1, 0, 0],
                [0, 0, -1, 0], [0, 0, 1, 0],
                [0, 0, 0, -1], [0, 0, 0, 1]
            ])

        for offset in offsets:
            neighbor = multi_idx + offset

            # Wrap rotation boundaries and clip the other dimensions.
            neighbor[0] = np.clip(neighbor[0], 0, self.n_coarse[0] - 1)
            neighbor[1] = np.clip(neighbor[1], 0, self.n_coarse[1] - 1)
            neighbor[2] = neighbor[2] % self.n_coarse[2]  # Periodic dimension
            neighbor[3] = np.clip(neighbor[3], 0, self.n_coarse[3] - 1)

            # Convert to flat indices.
            flat_idx = self._multi_index_to_flat(neighbor.reshape(1, 4), self.n_coarse)[0]

            if flat_idx != coarse_idx:
                adjacent.append(flat_idx)

        return np.unique(adjacent)

    # ==================== Sampling methods ====================

    def sample_uniform_grid_by_bins(
            self,
            n_bins: Union[Tuple[int, int, int, int], List[int], np.ndarray, torch.Tensor],
            device: Optional[torch.device] = None,
            scale_mode: str = "linear",
    ) -> torch.Tensor:
        """Sample global bin centers uniformly at the requested resolution.

        Args:
            n_bins: Counts (n_nr, n_nc, n_rot, n_scale).
            device: Target device.
            scale_mode: Scale discretization: linear or log.
        Returns:
            Coordinates [N_total, 4]."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.is_tensor(n_bins):
            n_bins_np = n_bins.detach().cpu().numpy()
        else:
            n_bins_np = np.asarray(n_bins, dtype=np.int32)

        if n_bins_np.shape[0] != 4:
            raise ValueError(f"n_bins must be length-4, got shape {n_bins_np.shape}")

        scale_mode = str(scale_mode).strip().lower()
        if scale_mode not in ("linear", "log"):
            raise ValueError(f"scale_mode must be 'linear' or 'log', got {scale_mode}")

        coord_ranges = torch.from_numpy(self.coord_ranges).to(device=device, dtype=torch.float32)
        grids = []
        for i in range(4):
            n_i = int(n_bins_np[i])
            if n_i <= 0:
                raise ValueError(f"n_bins[{i}] must be > 0, got {n_i}")
            if i == 3:
                if not hasattr(self.sat_dataset, "mk_sacle_levels"):
                    raise AttributeError("sat_dataset must implement mk_sacle_levels for scale sampling.")
                scale_values, _ = self.sat_dataset.mk_sacle_levels(n_level=n_i, scale_mode=scale_mode)
                centers = scale_values.to(device=device, dtype=torch.float32)
            else:
                coord_min = coord_ranges[i, 0]
                coord_max = coord_ranges[i, 1]
                bin_size = (coord_max - coord_min) / float(n_i)
                centers = coord_min + (torch.arange(n_i, device=device, dtype=torch.float32) + 0.5) * bin_size
            grids.append(centers)

        mesh = torch.meshgrid(*grids, indexing='ij')
        coords = torch.stack([m.reshape(-1) for m in mesh], dim=1)
        coords[:, 2] = self._wrap_rotation_torch(coords[:, 2])
        return coords

    def sample_all_subspaces_gpu(
            self,
            n_points_per_subspace: int = 4,
            use_fine: bool = None,
            device: torch.device = None,
            rand_offset=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample every subspace once, without duplicate subspace visits.

        When coarse and fine partitions coincide, sample directly within each
        subspace. Otherwise, sample fine cells for more uniform coverage.
        Useful for full-space coverage and global feature galleries/indices.

        Args:
            n_points_per_subspace: Samples per subspace (P).
            device: GPU device.
            use_fine: Enable fine-cell sampling; None enables it when fine > 1.
            rand_offset: Random offsets within cells; False samples centers.
        Returns:
            Coordinates [n_subspaces, P, 4] and labels [n_subspaces, P]."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # [n_subspaces, 4]

        P = n_points_per_subspace
        N = self.n_subspaces

        # Determine automatically whether to use fine subdivisions.
        if use_fine is None:
            use_fine = self.n_fine_cells_per_subspace > 1

        if use_fine and self.n_fine_cells_per_subspace > 1:
            # Different coarse/fine partitions: sample fine cells.
            fine_offsets = cache['fine_offsets']      # [n_fine_cells, 4]
            fine_bin_sizes = cache['fine_bin_sizes']  # [4]
            n_fine_cells = fine_offsets.shape[0]

            # Randomly select a fine cell for each point in each subspace.
            selected_fine = torch.randint(0, n_fine_cells, (N, P), device=device)
            fine_off = fine_offsets[selected_fine]  # [N, P, 4]

            if rand_offset:
                # Random offset within the fine cell
                rand_off = torch.rand(N, P, 4, device=device) * fine_bin_sizes
            else:
                # Fine-cell center
                rand_off = fine_bin_sizes.view(1, 1, 4) * 0.5

            # coords = coarse_start + fine_offset + rand_offset
            coords = all_coarse_starts.unsqueeze(1) + fine_off + rand_off  # [N, P, 4]
        else:
            # Matching coarse/fine partitions: sample directly inside the subspace.
            coarse_bin_sizes = cache['coarse_bin_sizes']  # [4]
            if rand_offset:
                rand_offsets = torch.rand(N, P, 4, device=device) * coarse_bin_sizes
            else:
                rand_offsets = coarse_bin_sizes.view(1, 1, 4) * 0.5
            coords = all_coarse_starts.unsqueeze(1) + rand_offsets  # [N, P, 4]

        # Wrap rotation to [-π, π]
        coords[:, :, 2] = ((coords[:, :, 2] + torch.pi) % (2 * torch.pi)) - torch.pi

        # Labels are the corresponding subspace indices.
        labels = torch.arange(N, device=device).unsqueeze(1).expand(-1, P)

        return coords, labels


    def sample_batch_gpu_coarse(
            self,
            anchor_labels: torch.Tensor,
            n_subspaces_to_sample: int = 128,
            n_points_per_subspace: int = 4,
            include_anchor_subspace: bool = True,
            rand_offset: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample on the GPU using cached origins and random offsets.

        Precompute all subspace origins once, then index the lookup table
        directly instead of converting flat indices to multidimensional indices.
        Each sample requires one lookup and one random offset.

        Args:
            anchor_labels: Anchor subspace labels [B] on the GPU.
            n_subspaces_to_sample: Number of selected subspaces.
            n_points_per_subspace: Samples per subspace.
            include_anchor_subspace: Force inclusion of the anchor subspace.
            rand_offset: Use random offsets; False selects cell centers.
        Returns:
            Coordinates [B, N_total, 4] and candidate labels [B, N_total]."""
        device = anchor_labels.device
        B = anchor_labels.shape[0]

        # Get the GPU cache, including all precomputed subspace origins.
        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # Precomputed lookup table [n_subspaces, 4]
        coarse_bin_sizes = cache['coarse_bin_sizes']    # [4]

        n_total = n_subspaces_to_sample * n_points_per_subspace

        # ========== 1. Select random subspaces ==========
        selected_subspaces = torch.randint(
            0, self.n_subspaces, (B, n_subspaces_to_sample), device=device
        )

        if include_anchor_subspace:
            selected_subspaces[:, 0] = anchor_labels

        # ========== 2. Look up their starting coordinates directly ==========
        # all_coarse_starts: [n_subspaces, 4]
        # selected_subspaces: [B, K]
        # Result: [B, K, 4].
        coarse_starts = all_coarse_starts[selected_subspaces]

        # ========== 3. Sample within subspaces ==========
        # Uniform random offsets [B, K, P, 4] within [0, bin_size).
        if rand_offset:
            rand_offsets = torch.rand(
                B, n_subspaces_to_sample, n_points_per_subspace, 4, device=device
            ) * coarse_bin_sizes
        else:
            rand_offsets = coarse_bin_sizes.view(1, 1, 1, 4) * 0.5

        # Coordinates = origin + offset.
        coords = coarse_starts.unsqueeze(2) + rand_offsets  # [B, K, P, 4]
        coords = coords.view(B, n_total, 4)

        # Wrap rotation to [-π, π]
        coords[:, :, 2] = ((coords[:, :, 2] + torch.pi) % (2 * torch.pi)) - torch.pi

        # ========== 4. Generate labels ==========
        candidate_labels = selected_subspaces.unsqueeze(2).expand(
            -1, -1, n_points_per_subspace
        ).reshape(B, n_total)

        return coords, candidate_labels


    def sample_batch_gpu_fine(
            self,
            anchor_labels: torch.Tensor,
            n_subspaces_to_sample: int = 128,
            n_points_per_subspace: int = 4,
            include_anchor_subspace: bool = True,
            rand_offset: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample fine cells on the GPU, including distinct coarse/fine partitions.

        Cache all_coarse_starts [n_subspaces, 4], fine_offsets [n_fine_cells, 4],
        and fine_bin_sizes [4]. Compute coordinates as selected coarse origins
        plus selected fine-cell offsets plus rand() * fine_bin_sizes.

        Args:
            anchor_labels: Anchor subspace labels [B] on the GPU.
            n_subspaces_to_sample: Number of selected coarse subspaces (K).
            n_points_per_subspace: Samples per subspace (P).
            include_anchor_subspace: Force inclusion of the anchor subspace.
            rand_offset: Random fine-cell offsets; False selects cell centers.
        Returns:
            Coordinates [B, K*P, 4] and candidate labels [B, K*P]."""
        device = anchor_labels.device
        B = anchor_labels.shape[0]
        K = n_subspaces_to_sample
        P = n_points_per_subspace

        # Get the GPU cache.
        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # [n_subspaces, 4]
        fine_offsets = cache['fine_offsets']            # [n_fine_cells, 4]
        fine_bin_sizes = cache['fine_bin_sizes']        # [4]

        n_fine_cells = fine_offsets.shape[0]
        n_total = K * P

        # ========== 1. Select random coarse subspaces ==========
        selected_subspaces = torch.randint(0, self.n_subspaces, (B, K), device=device)

        if include_anchor_subspace:
            selected_subspaces[:, 0] = anchor_labels

        # Look up coarse origins: [B, K, 4].
        coarse_starts = all_coarse_starts[selected_subspaces]

        # ========== 2. Select a random fine cell for each point ==========
        # selected_fine [B, K, P] identifies each point's fine cell.
        selected_fine = torch.randint(0, n_fine_cells, (B, K, P), device=device)

        # Look up fine-cell offsets: [B, K, P, 4].
        fine_off = fine_offsets[selected_fine]

        # ========== 3. Sample random offsets within fine cells ==========
        if rand_offset:
            rand_off = torch.rand(B, K, P, 4, device=device) * fine_bin_sizes
        else:
            rand_off = fine_bin_sizes.view(1, 1, 1, 4) * 0.5

        # ========== 4. Assemble final coordinates ==========
        # coords = coarse_start + fine_offset + rand_offset
        coords = coarse_starts.unsqueeze(2) + fine_off + rand_off  # [B, K, P, 4]
        coords = coords.view(B, n_total, 4)

        # Wrap rotation to [-π, π]
        coords[:, :, 2] = ((coords[:, :, 2] + torch.pi) % (2 * torch.pi)) - torch.pi

        # ========== 5. Generate labels ==========
        candidate_labels = selected_subspaces.unsqueeze(2).expand(-1, -1, P).reshape(B, n_total)

        return coords, candidate_labels

    def sample_grid_in_subspaces_gpu(
            self,
            subspace_indices: torch.Tensor,
            grid_dims: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Generate regular grids inside selected subspaces for top-K refinement.

        Args:
            subspace_indices: Flat subspace indices [B, K].
            grid_dims: Per-subspace counts (nr, nc, rot, scale); e.g.
                (4, 4, 3, 1) generates 4*4*3*1 points in each coarse cell.
        Returns:
            Physical coordinates [B, K, Total_Points, 4], where
            Total_Points = nr * nc * rot * scale."""
        device = subspace_indices.device
        B, K = subspace_indices.shape

        # 1. Get base sampling information.
        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # [n_subspaces, 4]
        coarse_bin_sizes = cache['coarse_bin_sizes']  # [4]

        # 2. Get the origins of selected subspaces.
        # [B, K, 4]
        start_coords = all_coarse_starts[subspace_indices]

        # 3. Generate a local normalized grid in [0, 1].
        # grid_dims: [D_r, D_c, D_rot, D_s]
        # Generate bin centers, e.g. four bins give [0.125, 0.375, 0.625, 0.875].
        local_grid_normalized = self._build_normalized_grid_centers_torch(grid_dims, device)

        # 4. Map to physical coordinates.
        # Coords = Start + Normalized_Grid * Bin_Size
        # Start: [B, K, 1, 4]
        # Grid:  [1, 1, Total_Points, 4]
        # Bin:   [1, 1, 1, 4]

        # [B, K, Total_Points, 4]
        coords = start_coords.unsqueeze(2) + local_grid_normalized.view(1, 1, -1, 4) * coarse_bin_sizes.view(1, 1, 1, 4)

        # 5. Handle rotation periodicity.
        coords[..., 2] = self._wrap_rotation_torch(coords[..., 2])

        return coords

    def sample_grid_at_2d_indices_gpu(
            self,
            indices_2d: Union[List, np.ndarray, torch.Tensor],
            grid_dims: Tuple[int, int, int, int],
            device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample all rotations and scales at selected 2D spatial indices.

        Visit every coarse rotation/scale cell at each selected position and
        sample a uniform fine grid within each resulting 4D coarse cell.

        Args:
            indices_2d: List of (nr_idx, nc_idx) tuples, NumPy array [N, 2],
                or torch tensor [N, 2].
            grid_dims: Per-cell (nr, nc, rot, scale) counts; (4, 4, 3, 1) gives 48 points.
            device: Target device; None selects automatically.
        Returns:
            Coordinates [N, K_rot*K_scale, Total_Points, 4] and flat labels
            [N, K_rot*K_scale], where K_rot=n_coarse[2], K_scale=n_coarse[3],
            and Total_Points=nr*nc*rot*scale.

        Examples:
            >>> indices_2d = [(0, 0), (1, 1)]
            >>> coords, labels = sampler.sample_grid_at_2d_indices_gpu(
            ...     indices_2d, grid_dims=(4, 4, 3, 1)
            ... )
            >>> # coords.shape: [2, n_rot*n_scale, 48, 4]
            >>> # labels.shape: [2, n_rot*n_scale]"""
        # 1. Convert input indices to a torch tensor.
        if isinstance(indices_2d, list):
            indices_2d = np.array(indices_2d, dtype=np.int32)
        if isinstance(indices_2d, np.ndarray):
            indices_2d = torch.from_numpy(indices_2d).long()

        if device is None:
            device = indices_2d.device if isinstance(indices_2d, torch.Tensor) else torch.device('cpu')
        indices_2d = indices_2d.to(device)

        # Ensure shape [N, 2].
        if indices_2d.ndim == 1:
            indices_2d = indices_2d.unsqueeze(0)
        assert indices_2d.shape[-1] == 2, f"Expected shape [N, 2], got {indices_2d.shape}"

        N = indices_2d.shape[0]
        K_rot = int(self.n_coarse[2])   # Coarse rotation-bin count
        K_scale = int(self.n_coarse[3])  # Coarse scale-bin count
        K_total = K_rot * K_scale

        # 2. Generate all rotation/scale index combinations.
        # rot_indices: [K_rot], scale_indices: [K_scale]
        rot_indices = torch.arange(K_rot, device=device)
        scale_indices = torch.arange(K_scale, device=device)

        # Cartesian product: [K_rot, K_scale] -> [K_total, 2].
        rot_mesh, scale_mesh = torch.meshgrid(rot_indices, scale_indices, indexing='ij')
        rot_scale_pairs = torch.stack([rot_mesh.flatten(), scale_mesh.flatten()], dim=1)  # [K_total, 2]

        # 3. Build complete 4D subspace indices for each 2D position.
        # indices_2d: [N, 2] -> [N, 1, 2]
        # rot_scale_pairs: [K_total, 2] -> [1, K_total, 2]
        # Concatenate to [N, K_total, 4].
        spatial_2d_expanded = indices_2d.unsqueeze(1).expand(N, K_total, 2)  # [N, K_total, 2]
        rot_scale_expanded = rot_scale_pairs.unsqueeze(0).expand(N, K_total, 2)  # [N, K_total, 2]

        multi_indices_4d = torch.cat([spatial_2d_expanded, rot_scale_expanded], dim=-1)  # [N, K_total, 4]

        # 4. Convert to flat subspace indices.
        # multi_indices_4d: [N, K_total, 4] -> [N*K_total, 4]
        multi_indices_flat = multi_indices_4d.reshape(-1, 4).cpu().numpy()
        subspace_indices_flat = self._multi_index_to_flat(multi_indices_flat, self.n_coarse)  # [N*K_total]
        subspace_indices = torch.from_numpy(subspace_indices_flat).to(device).long().reshape(N, K_total)

        # 5. Call the existing sampling function.
        coords = self.sample_grid_in_subspaces_gpu(subspace_indices, grid_dims)  # [N, K_total, Total_Points, 4]

        return coords, subspace_indices

    def sample_grid_around_coords_gpu(
            self,
            center_coords: Union[np.ndarray, torch.Tensor],
            grid_dims: Tuple[int, int, int, int],
            sample_space_size: Optional[Union[np.ndarray, torch.Tensor, Tuple]] = None,
            device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample uniform grids around supplied center coordinates.

        Generate the requested number of samples per dimension inside a box
        centered at each input. The default box size is coarse_bin_sizes;
        grid construction follows sample_grid_in_subspaces_gpu.

        Args:
            center_coords: NumPy array or torch tensor [N, 4] or [4].
            grid_dims: Counts (nr, nc, rot, scale); (4, 4, 3, 2) yields 96 points.
            sample_space_size: Extents [4], as a tuple/list, NumPy array, or tensor.
                None uses coarse_bin_sizes.
            device: Target device; None selects automatically.
        Returns:
            Physical coordinates [N, Total_Points, 4], or [Total_Points, 4]
            for a single [4] input. Total_Points=nr*nc*rot*scale.

        Examples:
            >>> center = np.array([0.5, 0.5, 0.0, 1.0])
            >>> coords = sampler.sample_grid_around_coords_gpu(center, grid_dims=(4, 4, 3, 2))
            >>> # coords.shape: [96, 4]
            >>> centers = np.array([[0.5, 0.5, 0.0, 1.0], [0.3, 0.7, 1.5, 1.2]])
            >>> coords = sampler.sample_grid_around_coords_gpu(centers, grid_dims=(4, 4, 3, 2))
            >>> # coords.shape: [2, 96, 4]
            >>> coords = sampler.sample_grid_around_coords_gpu(
            ...     center, grid_dims=(4, 4, 3, 2), sample_space_size=(0.1, 0.1, 0.5, 0.2)
            ... )"""
        # 1. Convert center coordinates to a torch tensor.
        if isinstance(center_coords, np.ndarray):
            center_coords = torch.from_numpy(center_coords).float()
        elif not torch.is_tensor(center_coords):
            center_coords = torch.tensor(center_coords, dtype=torch.float32)

        # Determine the device.
        if device is None:
            device = center_coords.device if center_coords.is_cuda else torch.device('cpu')
        center_coords = center_coords.to(device)

        # Expand a single coordinate from [4] to [1, 4].
        is_single_coord = False
        if center_coords.ndim == 1:
            center_coords = center_coords.unsqueeze(0)
            is_single_coord = True

        assert center_coords.shape[-1] == 4, f"Expected last dim=4, got {center_coords.shape}"
        N = center_coords.shape[0]

        # 2. Determine the sampling extents.
        if sample_space_size is None:
            # Use coarse_bin_sizes as the default sampling extents.
            cache = self._get_gpu_cache(device)
            space_size = cache['coarse_bin_sizes']  # Expected shape: [4].
        else:
            # Use custom sampling extents.
            space_size = self._ensure_size_tensor_1d(sample_space_size, device)

        # 3. Generate a local normalized grid in [0, 1].
        # Follow the grid construction in sample_grid_in_subspaces_gpu.
        local_grid_normalized = self._build_normalized_grid_centers_torch(grid_dims, device)

        # 4. Shift the normalized grid from [0, 1] to [-0.5, 0.5], centered at zero.
        # Multiply by the sampling extents to obtain offsets from each center.
        # offset = (normalized - 0.5) * space_size
        local_offsets = (local_grid_normalized - 0.5) * space_size.view(1, 4)  # [Total_Points, 4]

        # 5. Compute final coordinates as center + offset.
        # center_coords: [N, 4] -> [N, 1, 4]
        # local_offsets: [Total_Points, 4] -> [1, Total_Points, 4]
        coords = center_coords.unsqueeze(1) + local_offsets.unsqueeze(0)  # [N, Total_Points, 4]

        # 6. Handle rotation periodicity.
        coords[..., 2] = self._wrap_rotation_torch(coords[..., 2])

        # 7. Return [Total_Points, 4] for a single input coordinate.
        if is_single_coord:
            coords = coords.squeeze(0)

        return coords


    # ==================== Helpers ====================
    def get_subspace_info(self) -> dict:
        """Return subspace partition information."""
        return {
            'n_coarse': self.n_coarse.tolist(),
            'n_fine_per_coarse': self.n_fine_per_coarse.tolist(),
            'n_subspaces': self.n_subspaces,
            'n_fine_cells_per_subspace': self.n_fine_cells_per_subspace,
            'coord_ranges': self.coord_ranges.tolist(),
            'coarse_bin_sizes': self.coarse_bin_sizes.tolist(),
            'fine_bin_sizes': self.fine_bin_sizes.tolist(),
        }

    def __repr__(self) -> str:
        return (f"SubspaceSampler(\n"
                f"  n_coarse={self.n_coarse.tolist()},\n"
                f"  n_fine_per_coarse={self.n_fine_per_coarse.tolist()},\n"
                f"  n_subspaces={self.n_subspaces},\n"
                f"  n_fine_cells_per_subspace={self.n_fine_cells_per_subspace}\n"
                f")")
