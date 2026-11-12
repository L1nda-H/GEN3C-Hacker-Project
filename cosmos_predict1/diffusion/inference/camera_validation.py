"""Shared OpenCV camera conventions for source preparation and rendering."""

import torch


def validate_opencv_cameras(w2cs, intrinsics):
    """Validate paired [...,4,4] rigid poses and [...,3,3] pinhole matrices."""
    if (not isinstance(w2cs, torch.Tensor) or w2cs.ndim < 2 or w2cs.shape[-2:] != (4, 4)
            or not isinstance(intrinsics, torch.Tensor)
            or intrinsics.shape != (*w2cs.shape[:-2], 3, 3)
            or intrinsics.device != w2cs.device):
        raise ValueError("OpenCV cameras require paired [...,4,4]/[...,3,3] matrices on one device")
    for matrix in (w2cs, intrinsics):
        if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
            raise ValueError("OpenCV cameras must contain finite floating matrices")
    # Validate in float64 regardless of input precision or caller autocast.
    pose, k = w2cs.double(), intrinsics.double()
    if (not torch.allclose(k[..., 2, :], k.new_tensor([0, 0, 1]).expand_as(k[..., 2, :]), atol=1e-5, rtol=0)
            or (k[..., 0, 0] <= 0).any() or (k[..., 1, 1] <= 0).any()
            or (k[..., 1, 0].abs() > 1e-5).any()):
        raise ValueError("Intrinsics must use the OpenCV upper-triangular pinhole convention")
    rotation = pose[..., :3, :3]
    if (not torch.allclose(pose[..., 3, :], pose.new_tensor([0, 0, 0, 1]).expand_as(pose[..., 3, :]),
                           atol=1e-5, rtol=0)
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation,
                                  torch.eye(3, device=pose.device, dtype=pose.dtype).expand_as(rotation),
                                  atol=1e-3, rtol=0)
            or not torch.allclose(torch.linalg.det(rotation), pose.new_ones(pose.shape[:-2]), atol=1e-3, rtol=0)):
        raise ValueError("w2cs must be right-handed rigid OpenCV world-to-camera matrices")
