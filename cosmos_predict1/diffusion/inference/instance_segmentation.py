"""Align and segment source observations without assigning cross-image identities."""

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

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


@dataclass(frozen=True)
class SegmentPrompts:
    """Optional XYXY boxes [N,4] and XY points [M,2] on integer pixel centers.

    point_labels [M] are 1 for foreground and 0 for background. Backends decide
    how prompts are grouped into objects; no prompts requests automatic proposals.
    """

    boxes: torch.Tensor | None = None
    points: torch.Tensor | None = None
    point_labels: torch.Tensor | None = None


@dataclass(frozen=True)
class InstanceProposal:
    """Backend mask (bool or probabilities [0,1]) and quality in [0,1]."""

    mask: torch.Tensor
    confidence: float
    class_name: str | None = None


@dataclass(frozen=True)
class SegmentPrediction:
    """Masks cover the entire backend input grid, without padding/cropping.

    Backend wrappers must undo their own letterboxing and convert logits to
    probabilities. environment_mask contains only explicitly confirmed regions.
    """

    proposals: Sequence[InstanceProposal] = ()
    environment_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class InstanceObservation:
    frame_idx: int
    view_id: int
    local_id: int
    mask: torch.Tensor
    confidence: float
    appearance: torch.Tensor
    track_id: int | None = None
    class_name: str | None = None


@dataclass(frozen=True)
class SegmentedObservation:
    """Prepared-grid masks independent of depth validity; no canonical labels.

    Instance masks retain their full support, including ambiguous pixels.
    Consumers must exclude ambiguous_mask before rasterizing accepted labels.
    unassigned_mask excludes candidates, ambiguity, and confirmed environment.
    """

    instances: tuple[InstanceObservation, ...]
    ambiguous_mask: torch.Tensor
    unassigned_mask: torch.Tensor
    environment_mask: torch.Tensor


@dataclass(frozen=True)
class SegmentedSource:
    prepared_source: PreparedSource
    observations: Mapping[ObservationKey, SegmentedObservation]


def _rgb_for_backend(rgb, source_range, target_range, layout, size=None):
    if rgb.ndim != 3 or rgb.shape[0] != 3 or not rgb.is_floating_point() or min(rgb.shape[-2:]) <= 0:
        raise ValueError("RGB must be a nonempty floating tensor [3,H,W]")
    if len(source_range) != 2 or not all(math.isfinite(v) for v in source_range):
        raise ValueError("rgb_range must contain two finite bounds")
    low, high = source_range
    if low >= high or not torch.isfinite(rgb).all() or (rgb < low).any() or (rgb > high).any():
        raise ValueError("RGB values must lie within the explicitly supplied rgb_range")
    if layout not in ("CHW", "HWC") or target_range not in ("0_1", "-1_1", "uint8"):
        raise ValueError("Backend expects CHW/HWC and 0_1, -1_1, or uint8")
    result = (rgb.float() - low) / (high - low)
    if size is not None:
        result = F.interpolate(result[None], size=size, mode="bilinear", align_corners=False)[0]
    if target_range == "-1_1":
        result = result * 2 - 1
    elif target_range == "uint8":
        result = (result * 255).round().to(torch.uint8)
    if layout == "HWC":
        result = result.permute(1, 2, 0)
    return result.contiguous()


def _binary_mask(mask, size, device, threshold):
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(size):
        raise ValueError(f"Backend masks must cover its input grid {tuple(size)}")
    if mask.dtype != torch.bool:
        if mask.is_complex() or not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
            raise ValueError("Masks must be boolean or probabilities in [0,1]")
        mask = mask >= threshold
    return mask.to(device=device, dtype=torch.bool)


class SegmenterAdapter:
    """Wrap a callable backend(image, prompts) -> SegmentPrediction.

    Input is RGB in the configured layout/range on the prepared RGB device.
    Optional input_size resizes the image and prompts; full-grid predictions are
    sampled back to prepared coordinates with categorical nearest sampling.
    Backend wrappers handle model-specific preprocessing and device transfers.
    """

    def __init__(self, backend: Callable, *, input_layout="HWC", input_range="uint8",
                 input_size: tuple[int, int] | None = None, mask_threshold=0.5):
        if input_size is not None and (len(input_size) != 2 or
                                      any(type(v) is not int or v <= 0 for v in input_size)):
            raise ValueError("input_size must contain positive integer (height, width)")
        if not 0 < mask_threshold < 1:
            raise ValueError("mask_threshold must be between 0 and 1")
        self.backend = backend
        self.input_layout = input_layout
        self.input_range = input_range
        self.input_size = input_size
        self.mask_threshold = mask_threshold

    @torch.no_grad()
    def segment(self, rgb: torch.Tensor, *, rgb_range: tuple[float, float],
                prompts: SegmentPrompts | None = None) -> SegmentPrediction:
        h, w = rgb.shape[-2:]
        size = self.input_size or (h, w)
        image = _rgb_for_backend(rgb, rgb_range, self.input_range, self.input_layout, self.input_size)
        prompts = prompts or SegmentPrompts()
        scaled = {}
        for name, width in (("boxes", 4), ("points", 2)):
            value = getattr(prompts, name)
            if value is None:
                scaled[name] = None
                continue
            if value.ndim != 2 or value.shape[1] != width or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite [N,{width}] coordinates")
            value = value.to(device=rgb.device, dtype=torch.float32).clone()
            if ((value[:, 0::2] < 0) | (value[:, 0::2] > w - 1)).any() or (
                (value[:, 1::2] < 0) | (value[:, 1::2] > h - 1)
            ).any():
                raise ValueError("Prompts must lie on the prepared image grid")
            if name == "boxes" and (value[:, 2:] < value[:, :2]).any():
                raise ValueError("Boxes must use ordered XYXY coordinates")
            scale = value.new_tensor([size[1] / w, size[0] / h] * (width // 2))
            scaled[name] = (value + 0.5) * scale - 0.5
        labels = prompts.point_labels
        if prompts.points is not None:
            if labels is None or labels.shape != (len(prompts.points),) or not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Each point needs a binary point_label")
            labels = labels.to(device=rgb.device, dtype=torch.int64)
        elif labels is not None:
            raise ValueError("point_labels require points")
        prediction = self.backend(image, SegmentPrompts(point_labels=labels, **scaled))

        def remap(mask):
            result = _binary_mask(mask, size, rgb.device, self.mask_threshold)
            return _nearest_resize(result, (h, w)) if size != (h, w) else result

        proposals = []
        for proposal in prediction.proposals:
            confidence = float(proposal.confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Segmentation confidence must be finite and in [0,1]")
            proposals.append(InstanceProposal(remap(proposal.mask), confidence, proposal.class_name))
        environment = remap(prediction.environment_mask) if prediction.environment_mask is not None else None
        return SegmentPrediction(tuple(proposals), environment)


class AppearanceEncoderAdapter:
    """Wrap one encoder(masked_crop, crop_mask) -> floating embedding [D].

    Crops tightly bound the object; background pixels are zeroed in the backend's
    input range. The encoder receives the mask as well, and must exclude those
    pixels from pooling/preprocessing. Use this same encoder across the source.
    Outputs are finite, nonzero, L2-normalized vectors on the RGB device.
    """

    def __init__(self, backend: Callable, *, input_layout="CHW", input_range="0_1"):
        self.backend = backend
        self.input_layout = input_layout
        self.input_range = input_range

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor, mask: torch.Tensor, *, rgb_range: tuple[float, float]) -> torch.Tensor:
        if mask.shape != rgb.shape[-2:] or mask.dtype != torch.bool or mask.device != rgb.device or not mask.any():
            raise ValueError("Appearance mask must be nonempty boolean [H,W] on the RGB device")
        y, x = mask.nonzero(as_tuple=True)
        top, bottom, left, right = y.min().item(), y.max().item() + 1, x.min().item(), x.max().item() + 1
        crop_mask = mask[top:bottom, left:right]
        crop = _rgb_for_backend(rgb[:, top:bottom, left:right], rgb_range,
                                self.input_range, self.input_layout)
        broadcast_mask = crop_mask[None] if self.input_layout == "CHW" else crop_mask[..., None]
        crop = crop.masked_fill(~broadcast_mask, 0)
        embedding = self.backend(crop, crop_mask)
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 1 or embedding.numel() == 0:
            raise ValueError("Appearance encoder must return an embedding [D]")
        if not embedding.is_floating_point() or not torch.isfinite(embedding).all():
            raise ValueError("Appearance embedding must be finite floating point")
        embedding = embedding.to(device=rgb.device, dtype=torch.float32)
        norm = torch.linalg.vector_norm(embedding)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Appearance embedding must have a finite nonzero norm")
        return embedding / norm


def clean_instance_proposals(
    proposals: Sequence[InstanceProposal], *, min_confidence=0.0, min_area=1,
    duplicate_iou=0.9, duplicate_area_ratio=0.95, boundary_width=0,
) -> tuple[tuple[InstanceProposal, ...], torch.Tensor]:
    """Confidence-ranked duplicate suppression and conservative uncertainty.

    Suppress only near-equal-area, high-IoU masks. Strictly contained masks with
    different support survive even at high IoU. All remaining overlaps are
    ambiguous. boundary_width optionally marks an inner boundary band uncertain.
    Empty/low-quality proposals are discarded. Masks remain unchanged.
    """
    if not 0 <= min_confidence <= 1 or not 0 < duplicate_iou <= 1 or not 0 < duplicate_area_ratio <= 1:
        raise ValueError("Quality and duplicate thresholds must lie in [0,1] (duplicate thresholds > 0)")
    if type(min_area) is not int or min_area < 1 or type(boundary_width) is not int or boundary_width < 0:
        raise ValueError("min_area must be positive and boundary_width nonnegative integers")
    if not proposals:
        raise ValueError("No mask grid available; segment_source handles empty predictions")
    template = proposals[0].mask
    if template.ndim != 2 or template.dtype != torch.bool:
        raise ValueError("Proposal masks must be boolean [H,W]")
    for proposal in proposals:
        if proposal.mask.shape != template.shape or proposal.mask.device != template.device or proposal.mask.dtype != torch.bool:
            raise ValueError("Proposal masks must share a boolean [H,W] grid and device")
        if not math.isfinite(proposal.confidence) or not 0 <= proposal.confidence <= 1:
            raise ValueError("Segmentation confidence must be finite and in [0,1]")
    kept = []
    for candidate in sorted(proposals, key=lambda p: p.confidence, reverse=True):
        area = int(candidate.mask.sum())
        if area < min_area or candidate.confidence < min_confidence:
            continue
        duplicate = False
        for previous in kept:
            other_area = int(previous.mask.sum())
            intersection = int((candidate.mask & previous.mask).sum())
            # Containment alone must never establish duplicate identity.
            strictly_contained = intersection == min(area, other_area) and area != other_area
            if (not strictly_contained and intersection / (area + other_area - intersection) >= duplicate_iou
                    and min(area, other_area) / max(area, other_area) >= duplicate_area_ratio):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    occupied = torch.zeros_like(template)
    ambiguous = torch.zeros_like(template)
    for proposal in kept:
        mask = proposal.mask
        ambiguous |= occupied & mask
        occupied |= mask
        if boundary_width:
            # Explicit padding treats pixels outside the image as background.
            padded = F.pad(mask[None, None].float(), (boundary_width,) * 4, value=0)
            eroded = -F.max_pool2d(-padded, 2 * boundary_width + 1, stride=1)[0, 0] > 0
            ambiguous |= mask & ~eroded
    return tuple(kept), ambiguous


def segment_source(
    source: PreparedSource, segmenter: SegmenterAdapter, appearance_encoder: AppearanceEncoderAdapter, *,
    rgb_range: tuple[float, float], prompts: Mapping[ObservationKey, SegmentPrompts] | None = None,
    environment_masks: Mapping[ObservationKey, torch.Tensor] | None = None,
    min_confidence=0.0, min_area=1, duplicate_iou=0.9, duplicate_area_ratio=0.95, boundary_width=0,
) -> SegmentedSource:
    """Discover candidates on prepared RGB, keeping every observation key.

    Prompts and user-confirmed environment masks use prepared coordinates. An
    explicit environment mask is combined with backend-confirmed environment;
    candidate regions and uncertainty are always excluded. Depth validity never
    erases candidates or RGB environment evidence. Step 5 must intersect valid
    geometry before assigning label 0 or positive canonical IDs.
    """
    prompts = prompts or {}
    environment_masks = environment_masks or {}
    if (set(prompts) | set(environment_masks)) - set(source.observations):
        raise ValueError("Prompts/environment masks reference unknown observation keys")
    results = {}
    embedding_size = None
    for key, observation in source.observations.items():
        prediction = segmenter.segment(observation.rgb, rgb_range=rgb_range, prompts=prompts.get(key))
        empty = torch.zeros(observation.rgb.shape[-2:], dtype=torch.bool, device=observation.rgb.device)
        # Run cleanup even for empty backend results to validate quality settings.
        cleanup_input = prediction.proposals or (InstanceProposal(empty, 0.0),)
        kept, ambiguous = clean_instance_proposals(
            cleanup_input, min_confidence=min_confidence, min_area=min_area,
            duplicate_iou=duplicate_iou, duplicate_area_ratio=duplicate_area_ratio, boundary_width=boundary_width,
        )
        instances = []
        occupied = empty.clone()
        for local_id, proposal in enumerate(kept, start=1):
            appearance = appearance_encoder.encode(observation.rgb, proposal.mask, rgb_range=rgb_range)
            if embedding_size is not None and appearance.numel() != embedding_size:
                raise ValueError("One source requires comparable appearance embeddings of a fixed dimension")
            embedding_size = appearance.numel()
            instances.append(InstanceObservation(*key, local_id, proposal.mask, proposal.confidence,
                                                 appearance, class_name=proposal.class_name))
            occupied |= proposal.mask
        environment = empty.clone()
        if prediction.environment_mask is not None:
            environment |= prediction.environment_mask
        if key in environment_masks:
            supplied = environment_masks[key]
            if supplied.dtype != torch.bool or supplied.shape != empty.shape or supplied.device != empty.device:
                raise ValueError("Confirmed environment must be boolean [H,W] on the prepared RGB device")
            environment |= supplied
        environment &= ~occupied & ~ambiguous
        unassigned = ~occupied & ~ambiguous & ~environment
        results[key] = SegmentedObservation(tuple(instances), ambiguous, unassigned, environment)
    return SegmentedSource(source, results)
