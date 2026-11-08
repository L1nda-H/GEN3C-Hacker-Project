"""Align source observations for segmentation and existing 3D caches."""

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


ObservationKey = tuple[int, int]


@dataclass(frozen=True)
class SourceObservation:
    """Floating RGB [3,H,W], depth/mask/labels [H,W], K [3,3], w2c [4,4].

    Tensors share a device. RGB retains its range. Integer labels reserve -1
    for unknown and 0 for environment; positive values identify instances.
    """

    rgb: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    w2c: torch.Tensor
    geometry_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedSource:
    """Camera-Z observations with pre-sanitization geometry validity."""

    observations: Mapping[ObservationKey, SourceObservation]

    def cache_kwargs(self) -> dict:
        """Pack sorted frames/views as F,N,C,H,W for Cache3D_Base.

        Requires a complete frame/view grid and common image size. The cache
        adds B/V dimensions; labels stay in observations pending Step 6.
        """
        frames = sorted({key[0] for key in self.observations})
        views = sorted({key[1] for key in self.observations})
        keys = self.cache_keys
        if len(keys) != len(self.observations):
            raise ValueError("Cache construction requires the same views in every frame")
        observations = [self.observations[key] for key in keys]
        if len({tuple(obs.rgb.shape) for obs in observations}) != 1:
            raise ValueError("Cache construction requires a common image size; pass output_size to prepare_source")
        geometry_dtype = torch.float32
        for obs in observations:
            for tensor in (obs.depth, obs.intrinsics, obs.w2c):
                geometry_dtype = torch.promote_types(geometry_dtype, tensor.dtype)

        def pack(name, channel=False):
            tensors = [getattr(obs, name) for obs in observations]
            if name in ("depth", "intrinsics", "w2c"):
                tensors = [tensor.to(geometry_dtype) for tensor in tensors]
            if channel:
                tensors = [tensor.unsqueeze(0) for tensor in tensors]
            stacked = torch.stack(tensors)
            return stacked.reshape(len(frames), len(views), *stacked.shape[1:])

        return dict(
            input_image=pack("rgb"),
            input_depth=pack("depth", channel=True),
            input_mask=pack("geometry_mask", channel=True),
            input_intrinsics=pack("intrinsics"),
            input_w2c=pack("w2c"),
            input_format=["F", "N", "C", "H", "W"],
            is_depth=True,
        )

    @property
    def cache_keys(self) -> tuple[ObservationKey, ...]:
        """Frame-major order of the rectangular cache layout."""
        return tuple(
            (frame, view)
            for frame in sorted({key[0] for key in self.observations})
            for view in sorted({key[1] for key in self.observations})
        )


def _validate_observation(obs: SourceObservation) -> None:
    if obs.rgb.ndim != 3 or obs.rgb.shape[0] != 3 or not obs.rgb.is_floating_point():
        raise ValueError("RGB must be a floating tensor [3, H, W]")
    h, w = obs.rgb.shape[-2:]
    if min(h, w) <= 0 or obs.depth.shape != (h, w) or not obs.depth.is_floating_point():
        raise ValueError("Depth must be floating [H, W] and pixel-aligned with RGB")
    tensors = [obs.rgb, obs.depth, obs.intrinsics, obs.w2c]
    for name in ("geometry_mask", "labels"):
        tensor = getattr(obs, name)
        if tensor is not None:
            if tensor.shape != (h, w):
                raise ValueError(f"{name} must be [H, W] and pixel-aligned with RGB")
            tensors.append(tensor)
    if obs.geometry_mask is not None:
        if not torch.all((obs.geometry_mask == 0) | (obs.geometry_mask == 1)):
            raise ValueError("geometry_mask must contain only boolean or binary validity")
    if obs.labels is not None:
        if obs.labels.is_floating_point() or obs.labels.is_complex() or obs.labels.dtype == torch.bool:
            raise ValueError("labels must have an integer dtype")
        if torch.any(obs.labels < -1):
            raise ValueError("labels must be -1 (unknown), 0 (environment), or positive IDs")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("All observation tensors must share a device")
    for name, shape in (("intrinsics", (3, 3)), ("w2c", (4, 4))):
        matrix = getattr(obs, name)
        if matrix.shape != shape or not matrix.is_floating_point() or not torch.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite floating matrix {shape}")
    # Check in float64 so input precision does not weaken convention checks.
    k, pose = obs.intrinsics.double(), obs.w2c.double()
    if not torch.allclose(k[2], k.new_tensor([0, 0, 1]), atol=1e-5, rtol=0):
        raise ValueError("Intrinsics must use OpenCV pinhole convention with last row [0, 0, 1]")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[1, 0]) > 1e-5:
        raise ValueError("Intrinsics require positive focal lengths and an upper triangular matrix")
    if not torch.allclose(pose[3], pose.new_tensor([0, 0, 0, 1]), atol=1e-5, rtol=0):
        raise ValueError("w2c must have homogeneous last row [0, 0, 0, 1]")
    rotation = pose[:3, :3]
    if not torch.allclose(rotation.T @ rotation, torch.eye(3, device=pose.device, dtype=pose.dtype), atol=1e-3):
        raise ValueError("w2c must be rigid; source scale belongs in depth and translation units")
    if not torch.isclose(torch.linalg.det(rotation), pose.new_tensor(1), atol=1e-3):
        raise ValueError("w2c rotation must be right-handed")


def _nearest_resize(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    # Half-pixel nearest: floor((i + 0.5) * input/output), ties upward.
    # Integer arithmetic and direct indexing preserve int64 IDs exactly.
    h, w = tensor.shape
    y = ((2 * torch.arange(size[0], device=tensor.device) + 1) * h) // (2 * size[0])
    x = ((2 * torch.arange(size[1], device=tensor.device) + 1) * w) // (2 * size[1])
    return tensor[y[:, None], x[None, :]]


def prepare_source(
    observations: Mapping[ObservationKey, SourceObservation],
    *,
    is_depth: bool = True,
    camera_convention: str = "opencv",
    crop: tuple[int, int, int, int] | None = None,
    output_size: tuple[int, int] | None = None,
) -> PreparedSource:
    """Align observations keyed by nonnegative (frame_idx, view_id) integers.

    Inputs must share pixels within each observation and use OpenCV w2c
    (x right, y down, z forward, integer pixel centers). The caller ensures
    synchronization and shared world coordinates/units across observations.

    Apply crop (top,left,height,width), then output_size (H,W), to every image.
    Resize RGB bilinearly; sample depth, validity, and labels with nearest
    neighbors. Update intrinsics with crop offsets and half-pixel scaling.

    With is_depth=False, convert ray distance to camera-Z before transforms.
    Preserve invalid depths and capture validity before cache sanitization.
    """
    if camera_convention != "opencv":
        raise ValueError("Only OpenCV world-to-camera inputs are supported")
    if not observations:
        raise ValueError("A source must contain at least one observation")
    if output_size is not None:
        if len(output_size) != 2 or any(type(value) is not int or value <= 0 for value in output_size):
            raise ValueError("output_size must contain positive integer (height, width)")
    if crop is not None:
        if len(crop) != 4 or any(type(value) is not int for value in crop):
            raise ValueError("crop must contain integer (top, left, height, width)")
        if min(crop[:2]) < 0 or min(crop[2:]) <= 0:
            raise ValueError("crop must have nonnegative origin and positive size")

    prepared = {}
    source_device = None
    for key, obs in observations.items():
        if not isinstance(key, tuple) or len(key) != 2 or any(type(i) is not int or i < 0 for i in key):
            raise ValueError("Observation keys must be nonnegative integer (frame_idx, view_id) pairs")
        _validate_observation(obs)
        if source_device is not None and source_device != obs.rgb.device:
            raise ValueError("All source observations must share a device")
        source_device = obs.rgb.device
        valid = torch.isfinite(obs.depth) & (obs.depth > 0)
        if obs.geometry_mask is not None:
            valid = valid & obs.geometry_mask.bool()
        depth = obs.depth
        if not is_depth:
            # Lazy import: aligned camera-Z preprocessing does not require Warp.
            from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import unproject_points

            dtype = torch.float64 if depth.dtype == torch.float64 else torch.float32
            identity = torch.eye(4, device=depth.device, dtype=dtype)[None]
            camera_points = unproject_points(
                depth.to(dtype)[None, None], identity, obs.intrinsics.to(dtype)[None],
                is_depth=False, mask=valid[None, None],
            )
            depth = torch.where(valid, camera_points[0, ..., 2].to(depth.dtype), depth)

        rgb, labels = obs.rgb, obs.labels
        k = obs.intrinsics.clone()
        h, w = depth.shape
        if crop is not None:
            top, left, height, width = crop
            if top + height > h or left + width > w:
                raise ValueError(f"crop is outside observation {key}")
            rgb = rgb[:, top:top + height, left:left + width]
            depth = depth[top:top + height, left:left + width]
            valid = valid[top:top + height, left:left + width]
            labels = labels[top:top + height, left:left + width] if labels is not None else None
            k[0, 2] -= left
            k[1, 2] -= top
        h, w = depth.shape
        if output_size is not None and output_size != (h, w):
            rgb = F.interpolate(rgb[None], size=output_size, mode="bilinear", align_corners=False)[0]
            depth = _nearest_resize(depth, output_size)
            valid = _nearest_resize(valid, output_size)
            labels = _nearest_resize(labels, output_size) if labels is not None else None
            sy, sx = output_size[0] / h, output_size[1] / w
            k[0] *= sx
            k[1] *= sy
            k[0, 2] += (sx - 1) / 2
            k[1, 2] += (sy - 1) / 2
        prepared[key] = SourceObservation(rgb, depth, k, obs.w2c, valid, labels)
    return PreparedSource(prepared)
