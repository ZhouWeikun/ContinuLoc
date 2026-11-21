"""Batch rotation utilities for efficiently applying multiple angles to images."""
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import numpy as np


def _mean_fill_from_tensor(img):
    if img.ndim == 3:
        return img.mean(dim=(1, 2)).tolist()
    if img.ndim == 4:
        return img.mean(dim=(2, 3))
    raise TypeError(f"输入必须是3D或4D的Tensor，但收到了维度为{img.ndim}的输入。")

def batch_rotate_images_per_sample(images, angles_deg):
    """Rotate each image by its corresponding angle.

    Args:
        images: Input images [N, C, H, W].
        angles_deg: Tensor [N] or list of N rotation angles in degrees.

    Returns:
        Rotated images [N, C, H, W]."""
    N, C, H, W = images.shape
    device = images.device

    # Convert to a tensor.
    if isinstance(angles_deg, list):
        angles_deg = torch.tensor(angles_deg, dtype=torch.float32, device=device)
    elif isinstance(angles_deg, np.ndarray):
        angles_deg = torch.from_numpy(angles_deg).float().to(device)
    else:
        angles_deg = angles_deg.to(device)

    assert angles_deg.shape[0] == N, f"角度数量({angles_deg.shape[0]})必须等于图像数量({N})"

    # Convert to radians.
    angles_rad = angles_deg * (torch.pi / 180.0)

    # Construct one rotation matrix per image.
    cos_vals = torch.cos(angles_rad)  # [N]
    sin_vals = torch.sin(angles_rad)  # [N]

    # Affine transformation matrices [N, 2, 3]
    theta = torch.zeros(N, 2, 3, device=device)
    theta[:, 0, 0] = cos_vals
    theta[:, 0, 1] = sin_vals
    theta[:, 1, 0] = -sin_vals
    theta[:, 1, 1] = cos_vals

    # Generate the sampling grid.
    grid = F.affine_grid(theta, images.size(), align_corners=False)

    # Apply rotation.
    rotated_images = F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return rotated_images


def batch_rotate_images(images, angles_deg):
    """Efficiently rotate a batch of images to multiple angles.

    Args:
        images: Input images [N, C, H, W].
        angles_deg: List or array of rotation angles in degrees.

    Returns:
        Rotated images [N, R, C, H, W], where R is the number of angles."""
    N, C, H, W = images.shape
    n_angles = len(angles_deg)

    # Method 1: torch.nn.functional.affine_grid (recommended for speed).
    # Precompute affine matrices for all rotations.
    device = images.device
    angles_rad = torch.tensor(angles_deg, dtype=torch.float32, device=device) * (torch.pi / 180.0)

    # Construct rotation matrices.
    cos_vals = torch.cos(angles_rad)  # [R]
    sin_vals = torch.sin(angles_rad)  # [R]

    # Affine transformation matrices [R, 2, 3]
    theta = torch.zeros(n_angles, 2, 3, device=device)
    theta[:, 0, 0] = cos_vals   # cos
    theta[:, 0, 1] = sin_vals   # sin
    theta[:, 1, 0] = -sin_vals  # -sin
    theta[:, 1, 1] = cos_vals   # cos

    # Expand across all images: [N*R, 2, 3].
    theta_expanded = theta.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N * n_angles, 2, 3)

    # Expand images: [N*R, C, H, W].
    images_expanded = images.unsqueeze(1).repeat(1, n_angles, 1, 1, 1).reshape(N * n_angles, C, H, W)

    # Generate the sampling grid.
    grid = F.affine_grid(theta_expanded, images_expanded.size(), align_corners=False)

    # Apply rotation.
    rotated = F.grid_sample(images_expanded, grid, mode='bilinear', padding_mode='border', align_corners=False)

    # Reshape to [N, R, C, H, W].
    rotated_images = rotated.reshape(N, n_angles, C, H, W)

    return rotated_images


def batch_rotate_images_sequential(images, angles_deg, batch_size=8):
    """Rotate images in angle batches to reduce memory use.

    Args:
        images: Input images [N, C, H, W].
        angles_deg: List of rotation angles in degrees.
        batch_size: Number of angles processed at once.

    Returns:
        Rotated images [N, R, C, H, W]."""
    N, C, H, W = images.shape
    n_angles = len(angles_deg)
    device = images.device

    result_list = []

    # Process angles in batches.
    for i in range(0, n_angles, batch_size):
        angles_batch = angles_deg[i:i + batch_size]
        rotated_batch = batch_rotate_images(images, angles_batch)
        result_list.append(rotated_batch)

    # Concatenate all batches.
    rotated_images = torch.cat(result_list, dim=1)  # [N, R, C, H, W]

    return rotated_images


def rotate_with_torchvision(images, angles_deg):
    """Rotate with torchvision.rotate (slower, but broadly compatible).

    Args:
        images: Input images [N, C, H, W].
        angles_deg: List of rotation angles in degrees.

    Returns:
        Rotated images [N, R, C, H, W]."""
    N, C, H, W = images.shape
    n_angles = len(angles_deg)

    rotated_list = []

    fill_vals = _mean_fill_from_tensor(images)
    if torch.is_tensor(fill_vals):
        fill_vals = fill_vals.detach().cpu().tolist()
    for angle in angles_deg:
        # Apply the current angle to all images.
        per_angle = []
        for i in range(N):
            fill_val = fill_vals[i] if isinstance(fill_vals, list) else fill_vals
            per_angle.append(
                TF.rotate(images[i], angle, interpolation=TF.InterpolationMode.BILINEAR, fill=fill_val)
            )
        rotated_list.append(torch.stack(per_angle))

    # Stack: [R, N, C, H, W] -> [N, R, C, H, W]
    rotated_images = torch.stack(rotated_list, dim=0).permute(1, 0, 2, 3, 4)

    return rotated_images


# ==================== Performance benchmark ====================

if __name__ == '__main__':
    import time

    # Create test data.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 100  # Number of images
    C, H, W = 3, 224, 224
    n_angles = 36  # Number of rotation angles

    images = torch.randn(N, C, H, W, device=device)
    angles_deg = [-180 + 10 * i for i in range(n_angles)]

    print(f"测试配置:")
    print(f"  设备: {device}")
    print(f"  图像数量: {N}")
    print(f"  图像尺寸: {C}x{H}x{W}")
    print(f"  旋转角度数: {n_angles}")
    print()

    # ========== Method 1: vectorized rotation (recommended) ==========
    print("方法1: batch_rotate_images (全批量)")
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated1 = batch_rotate_images(images, angles_deg)

    torch.cuda.synchronize() if device == 'cuda' else None
    time1 = time.time() - start
    print(f"  时间: {time1:.3f}s")
    print(f"  输出shape: {rotated1.shape}")
    print()

    # ========== Method 2: chunked rotation (lower memory use) ==========
    print("方法2: batch_rotate_images_sequential (分批)")
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated2 = batch_rotate_images_sequential(images, angles_deg, batch_size=8)

    torch.cuda.synchronize() if device == 'cuda' else None
    time2 = time.time() - start
    print(f"  时间: {time2:.3f}s")
    print(f"  输出shape: {rotated2.shape}")
    print()

    # ========== Method 3: individual torchvision rotations (slow) ==========
    print("方法3: rotate_with_torchvision (逐个角度)")
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated3 = rotate_with_torchvision(images, angles_deg)

    torch.cuda.synchronize() if device == 'cuda' else None
    time3 = time.time() - start
    print(f"  时间: {time3:.3f}s")
    print(f"  输出shape: {rotated3.shape}")
    print()

    # ========== Method 4: original implementation (slowest baseline) ==========
    print("方法4: 原始循环方法 (baseline)")
    from util_mk_data_transform import RandomRotationWithAngle
    rotater = RandomRotationWithAngle(degrees=180, same_on_batch=True)

    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated4_list = []
    for angle in angles_deg:
        rotated = rotater(images, angle)
        rotated4_list.append(rotated)
    rotated4 = torch.stack(rotated4_list, dim=1)

    torch.cuda.synchronize() if device == 'cuda' else None
    time4 = time.time() - start
    print(f"  时间: {time4:.3f}s")
    print(f"  输出shape: {rotated4.shape}")
    print()

    # ========== Performance comparison ==========
    print("=" * 50)
    print("性能对比:")
    print(f"  方法1 (全批量):    {time1:.3f}s  (1.00x)")
    print(f"  方法2 (分批):      {time2:.3f}s  ({time2/time1:.2f}x)")
    print(f"  方法3 (torchvision): {time3:.3f}s  ({time3/time1:.2f}x)")
    print(f"  方法4 (原始循环):  {time4:.3f}s  ({time4/time1:.2f}x)")
    print()
    print(f"加速比 (相对于原始方法):")
    print(f"  方法1: {time4/time1:.1f}x")
    print(f"  方法2: {time4/time2:.1f}x")
    print(f"  方法3: {time4/time3:.1f}x")
