"""Transforms with parameter tracking.

Record rotation and scale augmentation parameters and apply them to 4D coordinates."""
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from typing import Tuple, Optional


def _resolve_fill_value(img, fill):
    if fill != "mean":
        return fill
    if torch.is_tensor(img):
        if img.ndim == 3:
            return img.mean(dim=(1, 2)).tolist()
        if img.ndim == 4:
            return img.mean(dim=(2, 3))
    img_np = np.asarray(img)
    if img_np.ndim == 0:
        return float(img_np)
    if img_np.ndim == 2:
        return float(img_np.mean())
    mean_vals = img_np.mean(axis=(0, 1))
    return tuple(int(round(x)) for x in mean_vals.tolist())


class RandomRotationWithParams(torch.nn.Module):
    """Random rotation transform exposing the applied angle.

    This project uses positive counterclockwise rotations, consistent with
    batch_rotate_images_per_sample. Its image-coordinate convention differs
    from torchvision.transforms.functional.rotate, so negate the angle passed
    to F.rotate."""
    def __init__(self, degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill="mean"):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.interpolation = interpolation
        self.fill = fill
        self.last_angle = None  # Store the last angle using the project's positive-counterclockwise convention.

    def forward(self, img):
        """Args:
            img: PIL image or tensor.
        Returns:
            The rotated image."""
        angle = random.uniform(self.degrees[0], self.degrees[1])
        self.last_angle = angle  # Record the angle in the project's convention.

        # Negate the angle to account for the differing F.rotate convention.
        fill_val = _resolve_fill_value(img, self.fill)
        return F.rotate(img, -angle, self.interpolation, fill=fill_val)

    def get_last_angle(self):
        """Return the last rotation angle in degrees, positive counterclockwise."""
        return self.last_angle


class RandomScaleWithParams(torch.nn.Module):
    """Random scaling to simulate changes in UAV altitude.

    scale > 1: Lower altitude, smaller ground coverage, larger objects.
        Crop a smaller central region and resize it to base_size.
    scale < 1: Higher altitude, larger ground coverage, smaller objects.
        Shrink the image and pad it to base_size.
    scale == 1: Leave the image unchanged.

    Apply before rotation to preserve more valid image content."""
    def __init__(self, scale_range=(0.8, 1.2), base_size=224, interpolation=transforms.InterpolationMode.BILINEAR, pad_mode=None):
        super().__init__()
        self.scale_range = scale_range
        self.base_size = base_size
        self.interpolation = interpolation
        self.pad_mode = str(pad_mode).lower() if pad_mode is not None else None
        self.last_scale = None  # Store the last scale factor.

    def forward(self, img):
        """Args:
            img: PIL image or tensor (C, H, W).
        Returns:
            Scaled image with spatial size (base_size, base_size)."""
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        self.last_scale = scale

        if scale > 1.0:
            # Lower UAV altitude gives smaller ground coverage.
            # Crop a central region of size base_size / scale and resize to base_size.
            crop_size = int(self.base_size / scale)
            img_cropped = F.center_crop(img, crop_size)
            img_scaled = F.resize(img_cropped, self.base_size, self.interpolation)

        elif scale < 1.0:
            # Higher UAV altitude gives larger ground coverage.
            # Resize to base_size * scale, then pad to base_size.
            resize_size = int(self.base_size * scale)
            img_resized = F.resize(img, resize_size, self.interpolation)

            # Compute padding.
            pad_total = self.base_size - resize_size
            pad_left = pad_total // 2
            pad_top = pad_total // 2
            pad_right = pad_total - pad_left
            pad_bottom = pad_total - pad_top

            if self.pad_mode in {"zero", "zeros", "0"}:
                fill_val = 0
            else:
                fill_val = _resolve_fill_value(img_resized, "mean")
            if torch.is_tensor(img_resized):
                if img_resized.ndim == 2:
                    if isinstance(fill_val, (list, tuple, np.ndarray)):
                        fill_scalar = float(np.mean(fill_val))
                    else:
                        fill_scalar = float(fill_val)
                    img_scaled = img_resized.new_full(
                        (self.base_size, self.base_size), fill_scalar
                    )
                    img_scaled[
                        pad_top:pad_top + resize_size,
                        pad_left:pad_left + resize_size
                    ] = img_resized
                else:
                    if isinstance(fill_val, (list, tuple, np.ndarray)):
                        fill_channels = torch.tensor(
                            fill_val, dtype=img_resized.dtype, device=img_resized.device
                        )
                    elif torch.is_tensor(fill_val):
                        fill_channels = fill_val.to(img_resized.device, img_resized.dtype)
                    else:
                        fill_channels = torch.full(
                            (img_resized.shape[0],),
                            float(fill_val),
                            dtype=img_resized.dtype,
                            device=img_resized.device,
                        )
                    img_scaled = img_resized.new_empty(
                        (img_resized.shape[0], self.base_size, self.base_size)
                    )
                    img_scaled[:] = fill_channels.view(-1, 1, 1)
                    img_scaled[
                        :,
                        pad_top:pad_top + resize_size,
                        pad_left:pad_left + resize_size
                    ] = img_resized
            else:
                img_scaled = F.pad(
                    img_resized,
                    [pad_left, pad_top, pad_right, pad_bottom],
                    fill=fill_val
                )

        else:
            # For scale == 1.0, preserve content while ensuring the requested size.
            img_scaled = F.resize(img, self.base_size, self.interpolation)

        return img_scaled

    def get_last_scale(self):
        """Return the last scale factor."""
        return self.last_scale


class GridSamplingQueryGeometry(torch.nn.Module):
    """One affine resampling after optional antialiased source reduction.

    Coordinates use Pillow edge coordinates (center=size/2). Positive rotation
    matches RandomRotationWithParams: F.rotate(image, -rotation_deg).
    A centered square of the original image defines the unchanged footprint.
    No intermediate rotated/scaled 224px images are produced. Shear remains an
    unlabelled augmentation, exactly as in the pipelined pipeline.
    """
    def __init__(self, size=224, rotation_range=(-180, 180), scale_range=(.95, 1.05),
                 shear_range=(0, 0), pad_mode=None, prefilter_resolution_factor=2.0):
        super().__init__()
        if size <= 0 or prefilter_resolution_factor < 0:
            raise ValueError('size must be positive and prefilter_resolution_factor nonnegative')
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError('Invalid scale_range')
        self.size = int(size)
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.pad_mode = pad_mode
        self.prefilter_resolution_factor = float(prefilter_resolution_factor)
        self.last_params = None

    def get_params(self):
        if self.last_params is None:
            raise RuntimeError('Call the transform before reading its parameters')
        return dict(self.last_params)

    def inverse_matrix(self, source_size, rotation_deg, scale, shear_deg=0):
        """Output -> source; inverse order is shear, rotation, scale, crop."""
        angle = np.deg2rad(rotation_deg)
        rotation_inverse = np.array([[np.cos(angle), np.sin(angle)],
                                     [-np.sin(angle), np.cos(angle)]])
        shear_inverse = np.array([[1., np.tan(np.deg2rad(shear_deg))], [0., 1.]])
        linear = (source_size / self.size / scale) * (rotation_inverse @ shear_inverse)
        translation = np.full(2, source_size / 2) - linear @ np.full(2, self.size / 2)
        return np.column_stack([linear, translation])

    def render(self, img, rotation_deg, scale, shear_deg=0):
        from PIL import Image, ImageStat
        if not isinstance(img, Image.Image):
            raise TypeError('GridSamplingQueryGeometry expects an original PIL image')
        if not np.isfinite([rotation_deg, scale, shear_deg]).all() or scale <= 0 or abs(shear_deg) >= 89:
            raise ValueError('Invalid geometry parameters')
        side = min(img.size)
        # Crop without interpolation; preserve the original image center to half-pixel precision.
        # Odd aspect differences are handled by the affine translation below.
        offset = np.array([(img.width-side)/2, (img.height-side)/2])
        left, top = int(np.floor(offset[0])), int(np.floor(offset[1]))
        source = img.crop((left, top, left+side, top+side))
        residual = offset - [left, top]
        # Pillow computes channel sums without materializing a full NumPy RGB copy.
        fill = 0 if self.pad_mode in {'zero', 'zeros', '0'} else tuple(round(v) for v in ImageStat.Stat(source).mean)
        if self.prefilter_resolution_factor:
            shear_inverse = np.array([[1., np.tan(np.deg2rad(shear_deg))], [0., 1.]])
            target = max(self.size, int(np.ceil(self.prefilter_resolution_factor*self.size*max(1., scale)*np.linalg.norm(shear_inverse, 2))))
            if side > target:
                source = source.resize((target, target), Image.Resampling.LANCZOS)
        matrix = self.inverse_matrix(source.width, rotation_deg, scale, shear_deg)
        matrix[:, 2] += residual * source.width / side
        result = source.transform((self.size, self.size), Image.Transform.AFFINE,
                                  matrix.reshape(-1).tolist(), Image.Resampling.BICUBIC, fillcolor=fill)
        self.last_params = dict(rotation_deg=float(rotation_deg), scale=float(scale), shear_deg=float(shear_deg))
        return result

    def forward(self, img):
        # Keep rotation/scale RNG order identical to the pipelined tracked transforms.
        scale = random.uniform(*self.scale_range)
        angle = random.uniform(*self.rotation_range)
        shear = float(torch.empty(1).uniform_(*self.shear_range).item())
        return self.render(img, angle, scale, shear)


class TransformWithParams:
    """Wrap a transform pipeline to expose rotation and scale parameters."""
    def __init__(self, transform_list):
        """
        Args:
            transform_list: list of transforms
        """
        self.transforms = transform_list
        self.rotation_transform = None
        self.scale_transform = None
        self.grid_sampling_transform = None

        # Locate the rotation and scale transforms.
        for t in transform_list:
            if isinstance(t, GridSamplingQueryGeometry):
                self.grid_sampling_transform = t
            elif isinstance(t, RandomRotationWithParams):
                self.rotation_transform = t
            elif isinstance(t, RandomScaleWithParams):
                self.scale_transform = t

    def __call__(self, img):
        """Apply all transforms."""
        for t in self.transforms:
            img = t(img)
        return img

    def get_params(self):
        """Get the parameters of the last transformation.

        Returns:
            dict: {'rotation_deg': float, 'scale': float}."""
        if self.grid_sampling_transform is not None:
            return self.grid_sampling_transform.get_params()
        params = {}
        if self.rotation_transform is not None:
            params['rotation_deg'] = self.rotation_transform.get_last_angle()
        else:
            params['rotation_deg'] = 0.0

        if self.scale_transform is not None:
            params['scale'] = self.scale_transform.get_last_scale()
        else:
            params['scale'] = 1.0

        return params


def mk_pil_transform_with_params(
        mean,
        std,
        imgsize2net=224,
        rand_rot=False,
        rotation_range_deg=180,
        rand_scale=False,
        scale_range=(0.8, 1.2),
        rand_affine=False,
        affine_para=None,
        rand_erase=False,
        color_jitter=False,
        center_crop=False,
        rand_crop=False,
        pad_mode=None,
        geometry_method='pipelined',
        prefilter_resolution_factor=2.0,
):
    """Create a transform pipeline with parameter tracking.

    geometry_method defaults to pipelined. grid_sampling combines geometric
    operations; prefilter_resolution_factor controls the antialiasing
    intermediate resolution.

    Transform order:
        1. Resize/crop to the initial dimensions.
        2. Scale to simulate altitude changes before rotation loses content.
        3. Rotate to simulate heading changes.
        4. Apply other augmentations, such as affine transforms and color jitter.
        5. Convert to a tensor and normalize.

    Args:
        rand_rot: Enable rotation augmentation.
        rotation_range_deg: Rotation range in degrees.
        rand_scale: Enable scale augmentation.
        scale_range: Minimum and maximum scale.
        Other arguments follow mk_pil_transform.

    Returns:
        TransformWithParams exposing the applied transformation parameters."""
    if geometry_method not in {'pipelined', 'grid_sampling'}:
        raise ValueError(f'Unknown query geometry pipeline: {geometry_method}')
    if geometry_method == 'grid_sampling':
        if not center_crop or rand_crop:
            raise ValueError('Grid sampling geometry requires center_crop=True and rand_crop=False')
        affine = dict(affine_para or {}) if rand_affine else {}
        for key, neutral in [('degrees', 0), ('translate', (0., 0.)), ('scale', (1., 1.))]:
            value = affine.get(key, neutral)
            if np.any(np.asarray(value) != np.asarray(neutral)):
                raise ValueError(f'Grid sampling geometry does not support extra affine {key}')
        shear = affine.get('shear', 5 if rand_affine else 0)
        if not isinstance(shear, (int, float)):
            raise ValueError('Grid sampling geometry currently accepts scalar x-shear only')
        rotation = (-rotation_range_deg, rotation_range_deg) if isinstance(rotation_range_deg, (int, float)) else rotation_range_deg
        transform_list = [GridSamplingQueryGeometry(
            imgsize2net, rotation if rand_rot else (0, 0), scale_range if rand_scale else (1, 1),
            (-shear, shear), pad_mode, prefilter_resolution_factor)]
    else:
        transform_list = [transforms.Resize(imgsize2net)]

        if center_crop:
            transform_list.append(transforms.CenterCrop(imgsize2net))
        if rand_crop:
            transform_list.append(transforms.RandomCrop(imgsize2net))

        pad_mode_norm = str(pad_mode).lower() if pad_mode is not None else None
        use_zero = pad_mode_norm in {"zero", "zeros", "0"}

        # ===== Apply scaling before rotation =====
        # 1. Add tracked scaling first to reduce content loss from rotation.
        if rand_scale:
            transform_list.append(RandomScaleWithParams(
                scale_range=scale_range,
                base_size=imgsize2net,
                pad_mode=pad_mode_norm,
            ))

        # 2. Add tracked rotation.
        if rand_rot:
            transform_list.append(RandomRotationWithParams(
                degrees=rotation_range_deg,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0 if use_zero else "mean"
            ))

        # Other standard transforms
        if rand_affine:
            default_affine_params = {
                'degrees': 0,  # Rotation is handled separately.
                'translate': (0, 0),
                'scale': (1.0, 1.0),  # Scaling is handled separately.
                'shear': 5,
            }
            if affine_para:
                if not isinstance(affine_para, dict):
                    raise TypeError("affine_para必须是一个字典。")
                default_affine_params.update(affine_para)

            def _mean_fill_from_stats(mean_vals):
                if mean_vals is None:
                    return 0
                mean_arr = np.asarray(mean_vals, dtype=float)
                if mean_arr.ndim == 0:
                    val = float(mean_arr)
                    return int(round(val * 255.0)) if val <= 1.0 else int(round(val))
                if mean_arr.max() <= 1.0:
                    mean_arr = mean_arr * 255.0
                return tuple(int(round(x)) for x in mean_arr.tolist())

            fill_val = 0 if use_zero else _mean_fill_from_stats(mean)
            transform_list.append(
                transforms.RandomAffine(
                    **default_affine_params,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=fill_val
                )
            )

    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
        )

    transform_list.append(transforms.ToTensor())

    if rand_erase:
        transform_list.append(
            transforms.RandomErasing(p=0.1, scale=(0.05, 0.15), ratio=(0.3, 3.3), value=0.)
        )

    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return TransformWithParams(transform_list)


def apply_augment_to_coords(coords_4d, rotation_deg, scale_factor):
    """Apply augmentation parameters to 4D coordinates.

    Args:
        coords_4d: Tensor [N, 4] or [4], ordered as [nr, nc, rot_rad, scale_ratio].
        rotation_deg: Rotation angle in degrees, positive clockwise.
        scale_factor: Scale factor.

    Returns:
        Augmented coordinate tensor.

    Position (nr, nc) stays fixed because the UAV location does not change.
    Rotation is added: rot_new = rot_old + rot_augment. Zooming in reduces
    ground coverage: scale_new = scale_old / scale_augment."""
    coords_augmented = coords_4d.clone()

    # Convert augmentation rotation to radians and add it to the original rotation.
    # Image rotation is positive counterclockwise; heading is typically positive clockwise.
    # Adjust the sign to match the coordinate-system convention.
    rot_augment_rad = torch.deg2rad(torch.tensor(rotation_deg, device=coords_4d.device, dtype=coords_4d.dtype))

    if coords_4d.ndim == 1:
        coords_augmented[2] = coords_augmented[2] + rot_augment_rad
        coords_augmented[2] = torch.atan2(torch.sin(coords_augmented[2]), torch.cos(coords_augmented[2]))
    else:  # ndim == 2
        coords_augmented[:, 2] = coords_augmented[:, 2] + rot_augment_rad
        coords_augmented[:, 2] = torch.atan2(torch.sin(coords_augmented[:, 2]), torch.cos(coords_augmented[:, 2]))
    # Apply the scale augmentation factor.
    # scale_factor > 1 zooms in: lower altitude and smaller field of view.
    # scale_factor < 1 zooms out: higher altitude and larger field of view.
    if coords_4d.ndim == 1:
        coords_augmented[3] = coords_augmented[3] / scale_factor
    else:
        coords_augmented[:, 3] = coords_augmented[:, 3] / scale_factor

    return coords_augmented


# Test code
if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    # Create a test image.
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))

    # Create a transform with parameter tracking.
    transform = mk_pil_transform_with_params(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        imgsize2net=224,
        rand_rot=True,
        rotation_range_deg=180,
        rand_scale=True,
        scale_range=(0.8, 1.2),
        center_crop=True,
    )

    # Apply the transform.
    img_transformed = transform(img)
    print(f"Transformed image shape: {img_transformed.shape}")

    # Retrieve parameters.
    params = transform.get_params()
    print(f"Transform parameters: {params}")

    # Test the coordinate transformation.
    coords_original = torch.tensor([0.5, 0.5, 0.0, 1.0])  # [nr, nc, rot, scale]
    coords_augmented = apply_augment_to_coords(
        coords_original,
        rotation_deg=params['rotation_deg'],
        scale_factor=params['scale']
    )
    print(f"Original coords: {coords_original}")
    print(f"Augmented coords: {coords_augmented}")
