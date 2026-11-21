import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from typing import List, Union


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


def _get_image_size(img):
    if torch.is_tensor(img):
        return [img.shape[-1], img.shape[-2]]
    return F.get_image_size(img)

# Custom transform with batch support
class RandomRotationWithAngle(torch.nn.Module):
    """Randomly rotate an image or image batch.

    For a single image, self.angle stores one angle. For a batch, each image
    can receive a different random angle and self.angle stores the list.

    Args:
        degrees: A scalar d gives [-d, d]; a (min, max) pair gives [min, max].
        same_on_batch: False (default) samples angles independently. True
            shares one angle across the batch, as in torchvision."""

    def __init__(self, degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill="mean", same_on_batch = False):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.interpolation = interpolation
        self.fill = fill
        self.same_on_batch = same_on_batch  # Control whether the batch shares one angle.

        # Store either a single angle or a list of angles.
        self.angle = None

    def forward(self, img: torch.Tensor, angle: Union[float, List[float], None] = None) -> torch.Tensor:
        """Args:
            img: A 3D image tensor or 4D image batch.
            angle: Optional scalar or list of angles. None samples randomly.
                For batches, a scalar applies one angle to all images; a list
                supplies an individual angle per image. Defaults to None.

        Returns:
            The rotated image or batch tensor."""
        # --- 1. Handle a batch ---
        if img.ndim == 4:
            batch_size = img.shape[0]

            # Step A: determine the angles.
            angles_to_apply = None
            if angle is not None:
                # Prefer explicitly provided deterministic angles.
                angles_to_apply = angle
            else:
                # Otherwise, sample random angles.
                if self.same_on_batch:
                    angles_to_apply = random.uniform(self.degrees[0], self.degrees[1])
                else:
                    angles_to_apply = [random.uniform(self.degrees[0], self.degrees[1]) for _ in range(batch_size)]

            self.angle = angles_to_apply  # Record the angles actually used.

            # Step B: rotate according to the angle type.
            if isinstance(angles_to_apply, (int, float)):
                angles_to_apply = [angles_to_apply] * batch_size
            elif isinstance(angles_to_apply, (list, tuple)):
                if len(angles_to_apply) != batch_size:
                    raise ValueError(f"提供的角度列表长度 ({len(angles_to_apply)}) 与批次大小 ({batch_size}) 不匹配。")
            else:
                raise TypeError(f"要应用的旋转角度必须是浮点数或浮点数列表，但得到的是 {type(angles_to_apply)}。")

            rotated_images = []
            for i, ang in enumerate(angles_to_apply):
                fill_val = _resolve_fill_value(img[i], self.fill)
                if torch.is_tensor(fill_val):
                    fill_val = fill_val[i].tolist()
                rotated_images.append(
                    F.rotate(img[i], ang, self.interpolation, fill=fill_val)
                )
            return torch.stack(rotated_images)

        # --- 2. Handle a single image ---
        elif img.ndim == 3:
            angle_to_apply = None
            if angle is not None:
                # Prefer explicitly supplied deterministic angles.
                angle_to_apply = angle
            else:
                # Otherwise, sample random angles.
                angle_to_apply = random.uniform(self.degrees[0], self.degrees[1])

            self.angle = angle_to_apply  # Record the angles actually used.
            fill_val = _resolve_fill_value(img, self.fill)
            return F.rotate(img, angle_to_apply, self.interpolation, fill=fill_val)

        # --- 3. Handle invalid inputs ---
        else:
            raise TypeError(f"输入必须是3D或4D的Tensor，但收到了维度为{img.ndim}的输入。")


class RandomAffineWithMean(torch.nn.Module):
    """
    RandomAffine with per-image mean fill support for tensor/PIL inputs.
    """
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill="mean",
        center=None,
        same_on_batch=False,
    ):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill
        self.center = center
        self.same_on_batch = same_on_batch

    def _sample_params(self, img):
        img_size = _get_image_size(img)
        return transforms.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, img_size
        )

    def forward(self, img):
        if img.ndim == 4:
            batch_size = img.shape[0]
            rotated_images = []
            if self.same_on_batch:
                angle, translations, scale, shear = self._sample_params(img[0])
            for i in range(batch_size):
                if not self.same_on_batch:
                    angle, translations, scale, shear = self._sample_params(img[i])
                fill_val = _resolve_fill_value(img[i], self.fill)
                if torch.is_tensor(fill_val):
                    fill_val = fill_val[i].tolist()
                rotated_images.append(
                    F.affine(
                        img[i],
                        angle=angle,
                        translate=translations,
                        scale=scale,
                        shear=shear,
                        interpolation=self.interpolation,
                        fill=fill_val,
                        center=self.center,
                    )
                )
            return torch.stack(rotated_images)
        if img.ndim == 3:
            angle, translations, scale, shear = self._sample_params(img)
            fill_val = _resolve_fill_value(img, self.fill)
            return F.affine(
                img,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=self.interpolation,
                fill=fill_val,
                center=self.center,
            )
        raise TypeError(f"输入必须是3D或4D的Tensor，但收到了维度为{img.ndim}的输入。")

def mk_tensor_transform(
        imgsize2net=224,
        rand_rot = False,
        rand_affine = False,
        affine_para = None,
        rand_erase = False,
        ):
    transform_list = [transforms.Resize(imgsize2net,antialias=True)]

    rotator = None
    if rand_rot:
        # Instantiate the custom transform.
        rotator = RandomRotationWithAngle(degrees=180,same_on_batch=True)
        # Add the instance to the transformation list.
        transform_list.append(rotator)

    if rand_affine:
        # 1. Define default RandomAffine parameters.
        default_affine_params = {
            'degrees': 0,
            'translate': (0, 0),
            'scale': (1.0, 1.0),
            'shear': 5,
        }
        # 2. Update defaults with affine_para when provided.
        if affine_para:
            if not isinstance(affine_para, dict):
                raise TypeError("affine_para必须是一个字典。")
            default_affine_params.update(affine_para)
            # 3. Unpack the parameter dictionary with **.
        transform_list.append(
            RandomAffineWithMean(
                **default_affine_params,  # Pass the merged affine parameters.
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill="mean"
            )
        )

    if rand_erase:
        transform_list.append(transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=0))

    transform2ret = transforms.Compose(transform_list)
    return transform2ret,rotator


def mk_pil_transform(
        mean,
        std,
        imgsize2net=224,
        rand_affine = False,
        affine_para = None,
        rand_rot = False,
        rand_erase = False,
        color_jitter = False,
        center_crop = False,
        rand_crop = False,
        ):
    """Create a transform that does not accept inputs already converted to tensors."""
    transform_list = [transforms.Resize(imgsize2net)]
    if center_crop:
        transform_list += [transforms.CenterCrop(imgsize2net)]
    if rand_crop:
        transform_list += [transforms.RandomCrop(imgsize2net)]
    def _mean_fill_from_stats(mean_vals):
        if mean_vals is None:
            return 0
        mean_arr = np.asarray(mean_vals, dtype=float)
        if mean_arr.max() <= 1.0:
            mean_arr = mean_arr * 255.0
        return tuple(int(round(x)) for x in mean_arr.tolist())

    fill_val = _mean_fill_from_stats(mean)
    if rand_rot:
        transform_list.append(transforms.RandomRotation(180, interpolation=3, fill=fill_val))
    if rand_affine:
        # 1. Define default RandomAffine parameters.
        default_affine_params = {
            'degrees': 180,
            'translate': (0, 0),
            'scale': (1.0, 1.0),
            'shear': 5,
        }
        # 2. Update defaults with affine_para when provided.
        if affine_para:
            if not isinstance(affine_para, dict):
                raise TypeError("affine_para必须是一个字典。")
            default_affine_params.update(affine_para)
            # 3. Unpack the parameter dictionary with **.
        transform_list.append(
            transforms.RandomAffine(
                **default_affine_params,  # Pass the merged affine parameters.
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=fill_val
            )
        )
    if color_jitter:
        transform_list.append(
            transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            ])
        )
    transform_list += [transforms.ToTensor()]
    if rand_erase:
        transform_list.append(transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=1))
    transform_list +=[transforms.Normalize(mean=mean, std=std)]

    transform2ret = transforms.Compose(transform_list)
    return transform2ret
