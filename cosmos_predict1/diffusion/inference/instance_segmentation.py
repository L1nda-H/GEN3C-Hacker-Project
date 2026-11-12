"""Align and segment source observations without assigning cross-image identities."""

import hashlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_predict1.diffusion.inference.camera_validation import validate_opencv_cameras


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


def _cache_layout(tensor, input_format):
    """Normalize image-grid tensors without casting integer/categorical data."""
    dims = ("B", "F", "N", "V", "C", "H", "W")
    if (tensor.ndim != len(input_format) or len(set(input_format)) != len(input_format)
            or set(input_format) - set(dims) or not {"C", "H", "W"} <= set(input_format)):
        raise ValueError("input_format must uniquely name tensor dimensions including C,H,W")
    tensor = tensor.permute(*(input_format.index(dim) for dim in dims if dim in input_format))
    for index, dim in enumerate(dims):
        if dim not in input_format:
            tensor = tensor.unsqueeze(index)
    return tensor


@dataclass(frozen=True)
class PreparedSource:
    """Camera-Z observations with pre-sanitization geometry validity."""

    observations: Mapping[ObservationKey, SourceObservation]
    preparation: Mapping = field(default_factory=dict)

    def cache_kwargs(self, *, labeled_source=None) -> dict:
        """Pack sorted frames/views as F,N,C,H,W for Cache3D_Base.

        Requires a complete frame/view grid and common image size. The cache
        adds B/V dimensions. Supply a validated LabeledSource to attach labels.
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

        kwargs = dict(
            input_image=pack("rgb"),
            input_depth=pack("depth", channel=True),
            input_mask=pack("geometry_mask", channel=True),
            input_intrinsics=pack("intrinsics"),
            input_w2c=pack("w2c"),
            input_format=["F", "N", "C", "H", "W"],
            is_depth=True,
            observation_keys=keys,
        )
        if labeled_source is not None:
            from cosmos_predict1.diffusion.inference.instance_labeling import (
                check_labeled_source, prepared_source_fingerprint,
            )

            check_labeled_source(labeled_source)
            if prepared_source_fingerprint(self) != labeled_source.source_fingerprint:
                raise ValueError("Labels do not match the prepared-source fingerprint")
            kwargs.update(
                input_labels=torch.stack([labeled_source.labels[key] for key in keys]).reshape(
                    len(frames), len(views), 1, *observations[0].depth.shape,
                ),
                object_registry=labeled_source.registry,
                source_fingerprint=labeled_source.source_fingerprint,
                segmentation_revision=labeled_source.segmented_source.revision,
            )
        return kwargs

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
    validate_opencv_cameras(obs.w2c, obs.intrinsics)


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
    return PreparedSource(prepared, dict(crop=crop, output_size=output_size,
                                         is_depth=is_depth, camera_convention=camera_convention))


# requirements.txt pins the implementation used by these APIs.
SAM21_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM21_MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def file_sha256(path: str | Path) -> str:
    """Hash a checkpoint once at loading, or a saved artifact at persistence."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelProvenance:
    sam_revision: str
    configuration: str
    checkpoint_sha256: str
    settings: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not all(isinstance(value, str) and value for value in (self.sam_revision, self.configuration)):
            raise ValueError("Model provenance requires a revision and configuration")
        if (not isinstance(self.checkpoint_sha256, str) or len(self.checkpoint_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.checkpoint_sha256)):
            raise ValueError("Checkpoint SHA-256 must be a lowercase hexadecimal digest")

    def with_settings(self, **settings):
        return replace(self, settings={**self.settings, **settings})


@dataclass(frozen=True)
class SegmentPrompts:
    """One object hypothesis, using prepared-image XY pixel coordinates.

    All points [N,2], their binary point_labels [N], and the optional XYXY box
    [4] describe ONE object. Separate objects need separate groups and positive
    local_id values. Reusing a local_id with revised prompts replaces that
    object's result when segment_source is rerun; it does not establish identity
    across images. Supply the full revised click set on each call.
    """

    local_id: int
    points: torch.Tensor | None = None
    point_labels: torch.Tensor | None = None
    box: torch.Tensor | None = None
    name: str | None = None
    multimask_output: bool = False


@dataclass(frozen=True)
class InstanceProposal:
    """A SAM binary mask on the prepared grid, plus optional quality metadata.

    confidence is SAM's predicted IoU, not calibrated identity probability.
    Propagated video masks may have no confidence. Native video IDs must be
    scoped as (source_id, view_id, session_id, obj_id); image prediction sets none.
    """

    mask: torch.Tensor
    confidence: float | None
    local_id: int
    name: str | None = None
    stability_score: float | None = None
    track_id: tuple[str, int, str, int] | None = None


@dataclass(frozen=True)
class SegmentPrediction:
    proposals: tuple[InstanceProposal, ...] = ()


@dataclass(frozen=True)
class InstanceObservation:
    frame_idx: int
    view_id: int
    local_id: int
    mask: torch.Tensor
    confidence: float | None
    name: str
    stability_score: float | None = None
    track_id: tuple[str, int, str, int] | None = None


@dataclass(frozen=True)
class SegmentedObservation:
    """Full RGB masks independent of depth validity; no canonical labels.

    Consumers must exclude ambiguous_mask before rasterizing instance labels.
    unassigned_mask excludes candidates, ambiguity, and confirmed environment.
    Confirmed environment must still intersect geometry validity in step 5.
    confirmed_environment_mask retains the original user confirmation for editing.
    """

    instances: tuple[InstanceObservation, ...]
    ambiguous_mask: torch.Tensor
    unassigned_mask: torch.Tensor
    environment_mask: torch.Tensor
    # Preserve user confirmation before candidate/ambiguity/geometry filtering.
    confirmed_environment_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class SegmentedSource:
    prepared_source: PreparedSource
    observations: Mapping[ObservationKey, SegmentedObservation]
    revision: str = field(default_factory=lambda: uuid4().hex)
    provenance: ModelProvenance | None = None


def _sam_rgb(rgb: torch.Tensor, rgb_range: tuple[float, float]) -> np.ndarray:
    """SAM owns spatial preprocessing; convert only range/layout/device here."""
    if rgb.ndim != 3 or rgb.shape[0] != 3 or not rgb.is_floating_point() or min(rgb.shape[-2:]) <= 0:
        raise ValueError("RGB must be a nonempty floating tensor [3,H,W]")
    if len(rgb_range) != 2 or not all(math.isfinite(v) for v in rgb_range):
        raise ValueError("rgb_range must contain two finite bounds")
    low, high = rgb_range
    if low >= high or not torch.isfinite(rgb).all() or (rgb < low).any() or (rgb > high).any():
        raise ValueError("RGB values must lie within the explicitly supplied rgb_range")
    image = ((rgb.detach().float() - low) / (high - low) * 255).round().to(torch.uint8)
    return image.permute(1, 2, 0).cpu().contiguous().numpy()


def _sam_mask(mask, size, device) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=device)
    if tuple(mask.shape) != tuple(size):
        raise ValueError(f"SAM masks must match the prepared image grid {tuple(size)}")
    # The public predictor returns binary masks as floating NumPy arrays.
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("SAM must return binary masks, not logits")
    return mask.bool().clone()


def _quality(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("SAM image quality scores must be finite numbers") from exc
    if not math.isfinite(value):
        raise ValueError("SAM quality scores must be finite when supplied")
    return value


def _sam_prompts(group: SegmentPrompts, size: tuple[int, int]) -> dict:
    if type(group.local_id) is not int or group.local_id <= 0:
        raise ValueError("Prompt groups require a positive integer local_id")
    if type(group.multimask_output) is not bool:
        raise ValueError("multimask_output must be boolean")
    if group.name is not None and not isinstance(group.name, str):
        raise ValueError("Object name must be a string")
    h, w = size

    def coords(value, shape):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.is_complex():
            raise ValueError(f"Prompt coordinates must be tensors of shape {shape}")
        value = value.detach().float().cpu()
        if not torch.isfinite(value).all():
            raise ValueError("Prompt coordinates must be finite")
        if ((value[..., 0::2] < 0) | (value[..., 0::2] > w - 1)).any() or (
            (value[..., 1::2] < 0) | (value[..., 1::2] > h - 1)
        ).any():
            raise ValueError("Prompts must lie on the prepared image grid")
        return value.numpy().copy()

    points = labels = box = None
    if group.points is not None:
        if group.points.ndim != 2 or group.points.shape[1] != 2 or len(group.points) == 0:
            raise ValueError("points must be nonempty [N,2]")
        points = coords(group.points, (len(group.points), 2))
        labels = group.point_labels
        if labels is None or labels.shape != (len(points),) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Each point requires a binary point_label")
        labels = labels.detach().to(device="cpu", dtype=torch.int32).numpy().copy()
    elif group.point_labels is not None:
        raise ValueError("point_labels require points")
    if group.box is not None:
        box = coords(group.box, (4,))
        if (box[2:] <= box[:2]).any():
            raise ValueError("Box must have positive area and ordered XYXY coordinates")
    if points is None and box is None:
        raise ValueError("Each object prompt group requires clicks or a box")
    return dict(point_coords=points, point_labels=labels, box=box,
                multimask_output=group.multimask_output, return_logits=False, normalize_coords=True)


class SAM21Segmenter:
    """Thin wrapper for the official SAM 2.1 image APIs; no custom segmentation.

    Use from_checkpoint for production. Direct predictor/generator injection is
    supported for contract tests and callers already managing a shared SAM model.
    Instances have mutable predictor state: serialize calls to a shared wrapper.
    """

    def __init__(self, predictor, automatic_mask_generator=None, *, provenance=None):
        self.predictor = predictor
        self.automatic_mask_generator = automatic_mask_generator
        self.provenance = provenance

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, model_size="large", device="cuda",
                        automatic_mask_options: Mapping | None = None) -> "SAM21Segmenter":
        """Build a matching SAM 2.1 model/config pair without downloading weights.

        Quality, stability, sampling and box-NMS options go directly to SAM's
        automatic generator. requirements.txt includes SAM and GEN3C's dependency pins.
        """
        if model_size not in SAM21_MODELS:
            raise ValueError(f"model_size must be one of {tuple(SAM21_MODELS)}")
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"SAM 2.1 checkpoint does not exist: {checkpoint_path}")
        options = dict(automatic_mask_options or {})
        if options.get("output_mode", "binary_mask") != "binary_mask":
            raise ValueError("SAM21Segmenter requires output_mode='binary_mask'")
        options["output_mode"] = "binary_mask"
        from cosmos_predict1.utils.model_config import model_config_session

        # Reuse the application's session, including a prior normal import sam2.
        # For standalone calls, the helper scopes initialization before imports.
        with model_config_session(sam_config=SAM21_MODELS[model_size][0]):
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            except ModuleNotFoundError as exc:
                if exc.name == "sam2" or (exc.name or "").startswith("sam2."):
                    raise ImportError("Install SAM 2.1 with pip install -r requirements.txt") from exc
                raise
            model = build_sam2(SAM21_MODELS[model_size][0], str(checkpoint_path), device=device, mode="eval")
            return cls(SAM2ImagePredictor(model), SAM2AutomaticMaskGenerator(model, **options),
                       provenance=ModelProvenance(SAM21_REVISION, SAM21_MODELS[model_size][0],
                                                  file_sha256(checkpoint_path),
                                                  dict(automatic_mask_options=options)))

    @torch.inference_mode()
    def segment(self, rgb: torch.Tensor, *, rgb_range: tuple[float, float],
                prompts: Sequence[SegmentPrompts] | None = None) -> SegmentPrediction:
        """None requests automatic candidates; a sequence requests only its objects.

        Each group yields at most one mask. For multimask predictions select the
        highest predicted-IoU nonempty alternative. No NMS is applied across user
        groups: genuine contained objects remain distinct, with ambiguous overlap.
        Automatic discovery delegates all proposal filtering and NMS to SAM.
        """
        image = _sam_rgb(rgb, rgb_range)
        size, device = image.shape[:2], rgb.device
        proposals = []
        if prompts is None:
            if self.automatic_mask_generator is None:
                raise ValueError("Automatic discovery requires a SAM2AutomaticMaskGenerator")
            for local_id, record in enumerate(self.automatic_mask_generator.generate(image), start=1):
                mask = _sam_mask(record["segmentation"], size, device)
                if mask.any():
                    proposals.append(InstanceProposal(mask, _quality(record["predicted_iou"]), local_id,
                                                       stability_score=_quality(record["stability_score"])))
        else:
            prompts = tuple(prompts)
            arguments = [_sam_prompts(group, size) for group in prompts]
            if len({group.local_id for group in prompts}) != len(prompts):
                raise ValueError("Each object prompt group requires a unique local_id within its observation")
            if prompts:
                self.predictor.set_image(image)
            for group, kwargs in zip(prompts, arguments):
                masks, qualities, _ = self.predictor.predict(**kwargs)
                masks = np.asarray(masks)
                qualities = np.asarray(qualities)
                if masks.ndim != 3 or masks.shape[1:] != size or qualities.shape != (len(masks),):
                    raise ValueError("SAM must return masks [Q,H,W] and quality scores [Q]")
                alternatives = [(_sam_mask(mask, size, device), _quality(score))
                                for mask, score in zip(masks, qualities)]
                alternatives = [(mask, score) for mask, score in alternatives if mask.any()]
                if alternatives:
                    mask, score = max(alternatives, key=lambda pair: pair[1])
                    proposals.append(InstanceProposal(mask, score, group.local_id, group.name))
        return SegmentPrediction(tuple(proposals))


def segment_source(
    source: PreparedSource, segmenter: SAM21Segmenter, *, rgb_range: tuple[float, float],
    prompts: Mapping[ObservationKey, Sequence[SegmentPrompts]] | None = None,
    environment_masks: Mapping[ObservationKey, torch.Tensor] | None = None,
    boundary_width: int = 0,
) -> SegmentedSource:
    """Keep SAM masks keyed by observation, independent of geometry validity.

    Unspecified observation keys use automatic discovery; an explicit empty list
    requests no objects. Environment masks must be explicitly confirmed by the
    caller. Recompute after corrections with unchanged prompt local_ids; cached
    association/canonical labels must then be invalidated by their future owner.
    No generated canonical IDs, GUI, video propagation, or association is implied.
    """
    if type(boundary_width) is not int or boundary_width < 0:
        raise ValueError("boundary_width must be a nonnegative integer")
    prompts = prompts or {}
    environment_masks = environment_masks or {}
    if (set(prompts) | set(environment_masks)) - set(source.observations):
        raise ValueError("Prompts/environment masks reference unknown observation keys")
    results = {}
    for key, observation in source.observations.items():
        prediction = segmenter.segment(observation.rgb, rgb_range=rgb_range, prompts=prompts.get(key))
        results[key] = _segmented_observation(
            key, observation, prediction.proposals, environment_masks.get(key), boundary_width,
        )
    provenance = segmenter.provenance
    if provenance is not None:
        provenance = provenance.with_settings(mode="image", rgb_range=rgb_range, boundary_width=boundary_width)
    return SegmentedSource(source, results, provenance=provenance)


def _segmented_observation(key, observation, proposals, environment=None, boundary_width=0):
    """Shared image/video uncertainty and confirmed-environment policy."""
    empty = torch.zeros(observation.rgb.shape[-2:], dtype=torch.bool, device=observation.rgb.device)
    instances = []
    occupied, ambiguous = empty.clone(), empty.clone()
    for proposal in proposals:
        mask = proposal.mask
        instances.append(InstanceObservation(
            *key, proposal.local_id, mask, proposal.confidence,
            proposal.name if proposal.name is not None else f"Object {proposal.local_id}",
            proposal.stability_score, proposal.track_id,
        ))
        ambiguous |= occupied & mask
        occupied |= mask
        if boundary_width:
            padded = F.pad(mask[None, None].float(), (boundary_width,) * 4, value=0)
            eroded = -F.max_pool2d(-padded, 2 * boundary_width + 1, stride=1)[0, 0] > 0
            ambiguous |= mask & ~eroded
    environment = empty if environment is None else environment
    if environment.dtype != torch.bool or environment.shape != empty.shape or environment.device != empty.device:
        raise ValueError("Confirmed environment must be boolean [H,W] on the prepared RGB device")
    confirmed_environment = environment.clone()
    environment = environment & ~occupied & ~ambiguous
    return SegmentedObservation(tuple(instances), ambiguous,
                                ~occupied & ~ambiguous & ~environment, environment, confirmed_environment)
