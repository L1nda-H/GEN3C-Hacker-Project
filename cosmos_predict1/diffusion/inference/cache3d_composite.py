"""Selection, placement, and rendering for static labeled image caches.

The environment defines destination world coordinates. Cameras consuming these
samples use OpenCV world-to-camera conventions. Conditioning and joint renders
share global nearest-surface visibility, routing winners into separate source
buffers or one scene buffer. No child cache tensors are changed by this wrapper.
"""

from dataclasses import dataclass
import math

import torch

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base
from cosmos_predict1.diffusion.inference.camera_validation import validate_opencv_cameras
from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    adaptive_surface_winners, pixel_cell_winner_depth,
)

_CANDIDATE_BATCH_SIZE = 262144
_SUPPORT_DEPTH_TOLERANCE = .05


def combine_visible_buffers(rgb, coverage):
    """Recombine disjoint visible source buffers, retaining -1 RGB in holes.

    Inputs use [B,T,N,C,H,W]; outputs retain a singleton source dimension.
    """
    visible = coverage.bool()
    combined_coverage = visible.any(dim=2, keepdim=True)
    combined_rgb = torch.where(visible, rgb, 0.).sum(dim=2, keepdim=True)
    return combined_rgb.masked_fill(~combined_coverage, -1.), combined_coverage.to(coverage.dtype)


@dataclass(frozen=True)
class CompositeSamples:
    """One source/object's valid samples, in original row-major pixel order.

    points and rgb are [K,3]; pixel_indices is [K] on the source image grid.
    Tangents are [K,2,3] full source-pixel steps in destination world.
    RGB is omitted for depth-only rendering. Returned tensors own their storage
    so consumers cannot edit child tensors.
    """

    source_id: str
    instance_id: int
    points: torch.Tensor
    rgb: torch.Tensor | None
    pixel_indices: torch.Tensor
    tangents: torch.Tensor


@dataclass(frozen=True)
class _SampleTable:
    """Private CPU storage; never returned to callers as mutable samples."""

    points: torch.Tensor
    colors: torch.Tensor | None
    metadata: torch.Tensor | None
    tangents: torch.Tensor


class Cache3D_Composite:
    """Compose B=F=N=V=1 caches with one shared RGB range (GEN3C: [-1,1]).

    Placement uses column vectors: p_destination = T_source @ T_object @ p_source.
    Object transforms act in source coordinates. To rotate/scale around a chosen
    source-space pivot, supply Translate(pivot) @ RotateScale @ Translate(-pivot).
    Static samples can be reused at every target time; there is no timeline API.
    Rendering requires image_size=(H,W). Depth ties within depth_tolerance
    destination units use source ID, canonical object ID, then source pixel.
    Frame/point chunks bound projection memory; outputs use the camera device.
    Private CPU sample tables persist across renders. Change selections and
    placement through setters; call refresh_source after direct child edits.
    Rendering needs original source cameras supplied to add_source(). The
    candidate budget bounds expanded pixel batches, never footprint extent.
    """

    def __init__(self, *, rgb_range=(-1., 1.), image_size=None, depth_tolerance=1e-5,
                 frame_chunk_size=2, point_chunk_size=65536):
        if (len(rgb_range) != 2 or not all(math.isfinite(x) for x in rgb_range)
                or rgb_range[0] >= rgb_range[1]):
            raise ValueError("RGB range must have two finite increasing bounds")
        self.rgb_range = tuple(rgb_range)
        if image_size is not None and (
            len(image_size) != 2 or any(type(n) is not int or n <= 0 for n in image_size)
        ):
            raise ValueError("Target image_size must be positive integer (H,W)")
        if isinstance(depth_tolerance, bool) or not math.isfinite(depth_tolerance) or depth_tolerance < 0:
            raise ValueError("Depth tolerance must be finite and nonnegative")
        if any(type(n) is not int or n <= 0 for n in (frame_chunk_size, point_chunk_size)):
            raise ValueError("Frame and point chunk sizes must be positive integers")
        self.image_size = None if image_size is None else tuple(image_size)
        self.depth_tolerance = depth_tolerance
        self.frame_chunk_size = frame_chunk_size
        self.point_chunk_size = point_chunk_size
        self.sources = {}
        self.environment_source_id = None
        self.selections = {}
        self.source_transforms = {}
        self.object_transforms = {}
        self._attachments = {}
        self._label_states = {}
        self._placement_conflicts = set()
        self._sample_tables = {}
        self._joint_table = None
        self._table_rgb_range = self.rgb_range
        self._source_cameras = {}

    def _invalidate_samples(self, source_id=None):
        if source_id is None:
            self._sample_tables.clear()
        else:
            self._sample_tables.pop(source_id, None)
        self._joint_table = None

    @property
    def render_settings(self):
        """Runtime renderer settings for reproducible review manifests."""
        return dict(candidate_batch_size=_CANDIDATE_BATCH_SIZE,
                    support_depth_tolerance=_SUPPORT_DEPTH_TOLERANCE,
                    depth_tolerance=self.depth_tolerance,
                    frame_chunk_size=self.frame_chunk_size, point_chunk_size=self.point_chunk_size)

    @staticmethod
    def _identity():
        return torch.eye(4, dtype=torch.float64)

    @staticmethod
    def _similarity(matrix):
        try:
            matrix = torch.as_tensor(matrix).detach()
            if matrix.is_complex() or matrix.dtype == torch.bool:
                raise ValueError("Transform must be real")
            matrix = matrix.to(device="cpu", dtype=torch.float64).clone()
        except (TypeError, RuntimeError) as error:
            raise ValueError("Transform must be a real 4x4 matrix") from error
        if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
            raise ValueError("Transform must be a finite 4x4 matrix")
        if not torch.allclose(matrix[3], torch.tensor([0., 0., 0., 1.], dtype=matrix.dtype), atol=1e-7, rtol=0):
            raise ValueError("Transform must have affine bottom row [0,0,0,1]")
        linear = matrix[:3, :3]
        scale = torch.linalg.vector_norm(linear) / math.sqrt(3)
        if scale <= 0 or not torch.isfinite(scale):
            raise ValueError("Transform must have positive uniform scale")
        rotation = linear / scale
        if (not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=matrix.dtype), atol=1e-6, rtol=1e-6)
                or torch.linalg.det(rotation) <= 0):
            raise ValueError("Transform must be an orientation-preserving similarity (no shear/reflection)")
        return matrix

    def _validate_source(self, source_id, cache):
        if not isinstance(cache, Cache3D_Base):
            raise ValueError("Source must be a labeled Cache3D_Base child")
        if cache.input_image.ndim != 7 or tuple(cache.input_image.shape[:4]) != (1, 1, 1, 1):
            raise ValueError("Static composite requires B=F=N=V=1")
        if cache.object_registry is None or cache.object_registry.source_id != source_id:
            raise ValueError("Source ID must match registry ownership")
        cache.check_label_consistency()
        image = cache.input_image
        if (not torch.isfinite(image).all() or (image < self.rgb_range[0]).any()
                or (image > self.rgb_range[1]).any()):
            raise ValueError(f"Source RGB must be finite and within the shared range {self.rgb_range}")

    @staticmethod
    def _attachment(cache):
        return (id(cache), cache.geometry_revision, cache.source_fingerprint, cache.observation_keys)

    @staticmethod
    def _label_state(cache):
        registry = cache.object_registry
        return (id(cache.input_labels), id(registry), cache.segmentation_revision,
                cache._label_binding,
                None if registry is None else (registry.source_id, tuple(sorted(registry.aliases.items())),
                                                tuple(sorted(registry.names))))

    def refresh_source(self, source_id):
        """Validate corrections and canonicalize IDs once, retaining placements.

        replace_labels() is detected automatically. Call this explicitly after
        editing child tensors in place; arbitrary direct edits are not tracked.
        Geometry changes through update_cache() always require rebuilding.
        """
        if source_id not in self.sources:
            raise ValueError(f"Unknown source ID {source_id!r}")
        cache = self.sources[source_id]
        if self._attachments[source_id] != self._attachment(cache):
            raise ValueError(f"Source attachment for {source_id!r} changed; rebuild the composite")
        self._validate_source(source_id, cache)
        selected = frozenset(self._resolve(source_id, i, cache) for i in self.selections[source_id])
        placements = {}
        conflicts = set()
        for (owner, instance_id), matrix in self.object_transforms.items():
            if owner != source_id:
                continue
            resolved = self._resolve(source_id, instance_id, cache)
            key = (source_id, resolved)
            if (source_id, instance_id) in self._placement_conflicts:
                conflicts.add(key)
            if resolved in placements and not torch.equal(placements[resolved], matrix):
                conflicts.add(key)
            placements[resolved] = matrix
        self.selections[source_id] = selected
        self.object_transforms = {key: matrix for key, matrix in self.object_transforms.items()
                                  if key[0] != source_id}
        self.object_transforms.update({(source_id, i): matrix for i, matrix in placements.items()})
        self._placement_conflicts = {key for key in self._placement_conflicts if key[0] != source_id} | conflicts
        self._label_states[source_id] = self._label_state(cache)
        self._invalidate_samples(source_id)

    def _source(self, source_id):
        if source_id not in self.sources:
            raise ValueError(f"Unknown source ID {source_id!r}")
        cache = self.sources[source_id]
        if self._attachments[source_id] != self._attachment(cache):
            raise ValueError(f"Source attachment for {source_id!r} changed; rebuild the composite")
        if self._label_states[source_id] != self._label_state(cache):
            self.refresh_source(source_id)
        return cache

    def add_source(self, source_id, cache, is_environment=False, *, source_w2c=None, source_intrinsics=None):
        """Register unchanged child geometry; positive objects start unselected."""
        if not isinstance(source_id, str) or not source_id or source_id in self.sources:
            raise ValueError("Source IDs must be unique nonempty strings")
        if is_environment and self.environment_source_id is not None:
            raise ValueError("Exactly one source may supply the environment")
        self._validate_source(source_id, cache)
        if source_w2c is not None or source_intrinsics is not None:
            if (source_w2c is None or source_intrinsics is None
                    or source_w2c.shape != (4, 4) or source_intrinsics.shape != (3, 3)):
                raise ValueError("Source camera requires w2c [4,4] and intrinsics [3,3]")
            pose = source_w2c.detach().cpu().float().clone()
            intrinsic = source_intrinsics.detach().cpu().float().clone()
            validate_opencv_cameras(pose, intrinsic)
            self._source_cameras[source_id] = (pose, intrinsic)
        self.sources[source_id] = cache
        self.selections[source_id] = frozenset({0} if is_environment else ())
        self.source_transforms[source_id] = self._identity()
        self._attachments[source_id] = self._attachment(cache)
        self._label_states[source_id] = self._label_state(cache)
        if is_environment:
            self.environment_source_id = source_id
        # Adding a source can change diagnostic source indices.
        self._invalidate_samples()

    def _resolve(self, source_id, instance_id, cache):
        if type(instance_id) is not int or instance_id < 0:
            raise ValueError("Select registered positive IDs or environment ID 0; -1 is unresolved")
        if instance_id == 0:
            if source_id != self.environment_source_id:
                raise ValueError("Only the designated source may supply environment ID 0")
            return 0
        return cache.object_registry.resolve(instance_id)

    def set_selection(self, source_id, instance_ids):
        """Replace selection; environment 0 is always included from its source."""
        cache = self._source(source_id)
        selected = {self._resolve(source_id, i, cache) for i in instance_ids}
        if source_id == self.environment_source_id:
            selected.add(0)
        selected = frozenset(selected)
        if self.selections[source_id] != selected:
            self.selections[source_id] = selected
            self._invalidate_samples(source_id)

    def set_transform(self, source_id, matrix, instance_id=None):
        """Set source placement or one positive object's source-space placement."""
        cache = self._source(source_id)
        matrix = self._similarity(matrix)
        if instance_id is None:
            if source_id == self.environment_source_id and not torch.equal(matrix, self._identity()):
                raise ValueError("Environment source transform must stay at identity")
            if not torch.equal(self.source_transforms[source_id], matrix):
                self.source_transforms[source_id] = matrix
                self._invalidate_samples(source_id)
        else:
            instance_id = self._resolve(source_id, instance_id, cache)
            if instance_id == 0:
                raise ValueError("Cannot transform environment ID 0")
            key = (source_id, instance_id)
            if (key in self._placement_conflicts or
                    not torch.equal(self.object_transforms.get(key, self._identity()), matrix)):
                self._invalidate_samples(source_id)
            self.object_transforms[key] = matrix
            self._placement_conflicts.discard(key)

    def _source_pixel_tangents(self, source_id, cache):
        """Full source-pixel steps in world space, without crossing label/depth edges.

        Average compatible one-sided differences; missing axes use a conservative
        frontoparallel cell at source camera-Z. Source cameras are runtime-only.
        """
        if source_id not in self._source_cameras:
            raise ValueError(f"Adaptive source {source_id!r} requires source_w2c and source_intrinsics")
        pose, intrinsic = self._source_cameras[source_id]
        height, width = cache.input_image.shape[-2:]
        points = cache.input_points.reshape(height, width, 3).float()
        depth = (points * pose[2, :3]).sum(-1) + pose[2, 3]
        labels = cache.input_labels.reshape(height, width)
        valid = (cache.input_mask.reshape(height, width).bool() & torch.isfinite(points).all(-1)
                 & torch.isfinite(depth) & (depth > 0))
        directions = torch.linalg.inv(intrinsic)[:, :2].T @ pose[:3, :3]
        fallback = torch.where(valid, depth, 0.)[..., None, None] * directions
        tangents = fallback.clone()
        for tangent_axis, grid_axis in ((0, 1), (1, 0)):
            left, right = [slice(None)] * 2, [slice(None)] * 2
            left[grid_axis], right[grid_axis] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            compatible = (valid[left] & valid[right] & (labels[left] == labels[right])
                          & ((depth[left] - depth[right]).abs()
                             <= _SUPPORT_DEPTH_TOLERANCE * torch.minimum(depth[left], depth[right])))
            difference = torch.where(compatible[..., None], points[right] - points[left], 0.)
            total = torch.zeros_like(points)
            count = torch.zeros_like(depth)
            total[left] += difference
            total[right] += difference
            count[left] += compatible
            count[right] += compatible
            tangents[..., tangent_axis, :] = torch.where(
                (count > 0)[..., None], total / count.clamp_min(1)[..., None], fallback[..., tangent_axis, :],
            )
        return tangents.reshape(-1, 2, 3)

    def _iter_render_inputs(self, *, include_rgb=True, source_ids=None):
        """Yield selected valid samples in deterministic source/object order.

        Check attachments and refresh changed label snapshots before yielding. A
        label correction may merge selected IDs; differing placements on merged
        objects require an explicit set_transform() to resolve the conflict.
        Empty objects yield no samples. No target time is needed for static data.
        """
        if self.environment_source_id is None:
            raise ValueError("A designated environment source is required before rendering")
        inputs = []
        for source_id in sorted(self.sources) if source_ids is None else source_ids:
            cache = self._source(source_id)
            selected = self.selections[source_id]
            if any((source_id, i) in self._placement_conflicts for i in selected):
                raise ValueError("Merged objects have conflicting transforms; call set_transform again")
            inputs.append((source_id, cache, selected))
        for source_id, cache, selected in inputs:
            support = self._source_pixel_tangents(source_id, cache) if selected else None
            points = cache.input_points.reshape(-1, 3)
            rgb = cache.input_image[0, 0, 0, 0].permute(1, 2, 0).reshape(-1, 3) if include_rgb else None
            labels = cache.input_labels.reshape(-1)
            valid = cache.input_mask.reshape(-1).bool() & torch.isfinite(points).all(-1)
            for instance_id in sorted(selected):
                indices = torch.nonzero(valid & (labels == instance_id), as_tuple=True)[0]
                if not indices.numel():
                    continue
                transform = (self.source_transforms[source_id] @ self.object_transforms.get(
                    (source_id, instance_id), self._identity())).to(torch.float32)
                transformed = points[indices].to(torch.float32) @ transform[:3, :3].T + transform[:3, 3]
                tangents = support[indices] @ transform[:3, :3].T
                yield CompositeSamples(source_id, instance_id, transformed, rgb[indices] if include_rgb else None,
                                       indices, tangents)

    def _resolve_visibility(self, points, poses, intrinsics, tangents):
        """Replaceable visibility policy; ranks follow source/object/pixel order."""
        return adaptive_surface_winners(
            points, tangents, poses, intrinsics, self.image_size,
            depth_tolerance=self.depth_tolerance, point_chunk_size=self.point_chunk_size,
            candidate_batch_size=_CANDIDATE_BATCH_SIZE,
        )

    def _target_cameras(self, target_w2cs, target_intrinsics, start_frame_idx):
        if self.image_size is None:
            raise ValueError("Configure target image_size=(H,W) before rendering")
        if type(start_frame_idx) is not int or start_frame_idx < 0:
            raise ValueError("Render offset must be a nonnegative integer")
        if (not isinstance(target_w2cs, torch.Tensor) or target_w2cs.ndim != 4
                or target_w2cs.shape[0] != 1 or target_w2cs.shape[1] <= 0
                or target_w2cs.shape[2:] != (4, 4)):
            raise ValueError("Target w2cs must have shape [1,T,4,4] with T > 0")
        if (not isinstance(target_intrinsics, torch.Tensor)
                or target_intrinsics.shape != (*target_w2cs.shape[:2], 3, 3)
                or target_intrinsics.device != target_w2cs.device):
            raise ValueError("Target intrinsics must be [1,T,3,3] on the camera device")
        validate_opencv_cameras(target_w2cs, target_intrinsics)
        return target_w2cs[0].float(), target_intrinsics[0].float()

    @torch.no_grad()
    def _render(self, target_w2cs, target_intrinsics, start_frame_idx, *, rgb, depth, diagnostics=False):
        poses, intrinsics = self._target_cameras(target_w2cs, target_intrinsics, start_frame_idx)
        # Diffusion may call from an autocast context; visibility needs float32.
        with torch.autocast(device_type=poses.device.type, enabled=False):
            return self._render_frames(poses, intrinsics, rgb=rgb, depth=depth, diagnostics=diagnostics)

    def _supports(self, table, rgb, diagnostics):
        return (table is not None and (not rgb or table.colors is not None)
                and (not diagnostics or table.metadata is not None))

    def _render_tables(self, *, rgb, diagnostics):
        # Always check live attachment/label state, even on cache hits. Refreshes
        # invalidate tables and preserve label-merge placement conflict errors.
        source_ids = self.conditioning_source_ids
        for source_id in source_ids:
            if any((source_id, i) in self._placement_conflicts for i in self.selections[source_id]):
                raise ValueError("Merged objects have conflicting transforms; call set_transform again")
        if self._table_rgb_range != self.rgb_range:
            self._invalidate_samples()
            self._table_rgb_range = self.rgb_range
        source_index = {name: i for i, name in enumerate(sorted(self.sources))}
        for source_id in source_ids:
            previous = self._sample_tables.get(source_id)
            if self._supports(previous, rgb, diagnostics):
                continue
            # Retain already requested attributes when upgrading a depth table.
            include_rgb = rgb or (previous is not None and previous.colors is not None)
            include_metadata = diagnostics or (previous is not None and previous.metadata is not None)
            samples = list(self._iter_render_inputs(include_rgb=include_rgb, source_ids=(source_id,)))
            points = torch.cat([s.points for s in samples]) if samples else torch.empty(0, 3, dtype=torch.float32)
            colors = None
            if include_rgb:
                colors = torch.cat([s.rgb.float() for s in samples]) if samples else torch.empty(0, 3, dtype=torch.float32)
                if self.rgb_range != (-1., 1.):
                    colors = (colors - self.rgb_range[0]) * (2 / (self.rgb_range[1] - self.rgb_range[0])) - 1
            metadata = None
            if include_metadata:
                metadata = torch.cat([
                    torch.stack([torch.full_like(s.pixel_indices, source_index[source_id]),
                                 torch.full_like(s.pixel_indices, s.instance_id), s.pixel_indices], dim=-1)
                    for s in samples
                ]) if samples else torch.empty(0, 3, dtype=torch.int64)
            tangents = (torch.cat([s.tangents for s in samples]) if samples
                        else torch.empty(0, 2, 3, dtype=torch.float32))
            self._sample_tables[source_id] = _SampleTable(points, colors, metadata, tangents)
            self._joint_table = None
        return source_ids

    def _joint_render_table(self, *, rgb, diagnostics):
        source_ids = self._render_tables(rgb=rgb, diagnostics=diagnostics)
        if not self._supports(self._joint_table, rgb, diagnostics):
            tables = [self._sample_tables[source_id] for source_id in source_ids]
            # A single source needs no additional copies.
            if len(tables) == 1:
                self._joint_table = tables[0]
            else:
                self._joint_table = _SampleTable(
                    torch.cat([t.points for t in tables]),
                    torch.cat([t.colors for t in tables]) if rgb else None,
                    torch.cat([t.metadata for t in tables]) if diagnostics else None,
                    torch.cat([t.tangents for t in tables]),
                )
        return self._joint_table

    def _render_frames(self, poses, intrinsics, *, rgb, depth, diagnostics, source_slots=None):
        # Ownership needs sample metadata, but never full diagnostic outputs.
        table = self._joint_render_table(rgb=rgb, diagnostics=diagnostics or source_slots is not None)
        slots = 1 if source_slots is None else max(source_slots) + 1
        slot_map = None if source_slots is None else torch.tensor(source_slots, dtype=torch.int64)
        points, colors, metadata = table.points, table.colors, table.metadata
        if diagnostics:
            source_ids = tuple(sorted(self.sources))

        frames, (height, width) = len(poses), self.image_size
        pixels = height * width
        device = poses.device
        # Flat storage lets bounded winner batches write directly into outputs.
        coverage = torch.zeros(frames * slots * pixels, device=device, dtype=torch.float32)
        image = torch.full((frames * slots * pixels, 3), -1., device=device, dtype=torch.float32) if rgb else None
        z_buffer = torch.zeros_like(coverage) if depth else None
        if diagnostics:
            winners = torch.full((frames * pixels, 3), -1, dtype=torch.int64, device=device)
            nearest_depth = torch.zeros_like(coverage)
        for start in range(0, frames, self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, frames)
            winner, nearest = self._resolve_visibility(points, poses[start:end], intrinsics[start:end], table.tangents)
            if diagnostics:
                nearest_depth[start * pixels:end * pixels] = nearest.reshape(-1)
            winner = winner.reshape(-1)
            # Bound indices, CPU/GPU transfers, and selected attributes as well
            # as projection work. Never materialize all covered pixel indices.
            for offset in range(0, winner.numel(), self.point_chunk_size):
                batch = winner[offset:offset + self.point_chunk_size]
                covered = torch.nonzero(batch >= 0, as_tuple=True)[0]
                if not covered.numel():
                    continue
                rank = batch[covered].cpu()
                frame = (offset + covered) // pixels + start
                slot = 0 if slot_map is None else slot_map[metadata[rank, 0]].to(device)
                destination = (frame * slots + slot) * pixels + (offset + covered) % pixels
                coverage[destination] = 1
                if rgb:
                    image[destination] = colors[rank].to(device)
                if depth:
                    chosen = points[rank].to(device)
                    pixel = (offset + covered) % pixels
                    xy = torch.stack((pixel % width, pixel // width), -1).float()
                    z_buffer[destination] = pixel_cell_winner_depth(
                        chosen, table.tangents[rank].to(device), poses[frame], intrinsics[frame], xy,
                    )
                if diagnostics:
                    winners[destination] = metadata[rank].to(device)
        result = dict(coverage=coverage.reshape(1, frames, slots, 1, height, width))
        if rgb:
            result['rgb'] = image.reshape(frames, slots, height, width, 3).permute(0, 1, 4, 2, 3)[None]
        if depth:
            result['depth'] = z_buffer.reshape(1, frames, slots, height, width)
        if diagnostics:
            result.update(source_ids=source_ids, winners=winners.reshape(1, frames, height, width, 3),
                          nearest_depth=nearest_depth.reshape(1, frames, 1, height, width))
        return result

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        """Render float32 RGB/depth and binary coverage on the camera device.

        RGB [1,T,1,3,H,W] uses [-1,1], depth [1,T,1,H,W] is camera-Z,
        coverage [1,T,1,1,H,W] is shared by both. Hole values are -1/0/0.
        Static sources repeat at any nonnegative start_frame_idx. Only the
        requested RGB or depth buffer is allocated and populated.
        """
        result = self._render(target_w2cs, target_intrinsics, start_frame_idx,
                              rgb=not render_depth, depth=render_depth)
        return result['depth' if render_depth else 'rgb'], result['coverage']

    def render_diagnostics(self, target_w2cs, target_intrinsics, start_frame_idx=0, *, include_rgb=False):
        """Return depth, coverage, nearest_depth, and categorical winner metadata.

        winners [1,T,H,W,3] stores (source index into source_ids, canonical ID,
        original pixel index), with -1 for holes. IDs always remain int64.
        Set include_rgb=True to obtain all buffers in one visibility pass.
        """
        return self._render(target_w2cs, target_intrinsics, start_frame_idx,
                            rgb=include_rgb, depth=True, diagnostics=True)

    @torch.no_grad()
    def render_conditioning(self, target_w2cs, target_intrinsics, start_frame_idx=0, *, include_depth=False):
        """Render globally visible surfaces into stable source-ID ordered slots.

        RGB is [1,T,N,3,H,W], coverage [1,T,N,1,H,W], and optional camera-Z
        depth [1,T,N,H,W]. Empty or fully hidden geometry retains its slot;
        sources with no selection are omitted. Hole values are -1/0/0.
        """
        poses, intrinsics = self._target_cameras(target_w2cs, target_intrinsics, start_frame_idx)
        source_ids = self.conditioning_source_ids
        selected_slots = {name: i for i, name in enumerate(source_ids)}
        source_slots = tuple(selected_slots.get(name, -1) for name in sorted(self.sources))
        with torch.autocast(device_type=poses.device.type, enabled=False):
            return self._render_frames(poses, intrinsics, rgb=True, depth=include_depth,
                                       diagnostics=False, source_slots=source_slots)

    @property
    def conditioning_source_ids(self):
        """Stable buffer ownership, independent of target visibility or geometry validity."""
        if self.environment_source_id is None:
            raise ValueError("A designated environment source is required before rendering")
        for source_id in sorted(self.sources):
            self._source(source_id)
        return tuple(source_id for source_id in sorted(self.sources) if self.selections[source_id])
