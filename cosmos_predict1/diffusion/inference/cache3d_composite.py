"""Selection, placement, and joint visibility for static labeled image caches.

The environment defines destination world coordinates. Cameras consuming these
samples use OpenCV world-to-camera conventions. All sources share one nearest
surface rasterizer; no child cache tensors are changed by this wrapper.
"""

from dataclasses import dataclass
import math

import torch

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base
from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import nearest_surface_winners


@dataclass(frozen=True)
class CompositeSamples:
    """One source/object's valid samples, in original row-major pixel order.

    points and rgb are [K,3]; pixel_indices is [K] on the source image grid.
    Returned tensors own their storage so consumers cannot edit child tensors.
    """

    source_id: str
    instance_id: int
    points: torch.Tensor
    rgb: torch.Tensor
    pixel_indices: torch.Tensor


class Cache3D_Composite:
    """Compose B=F=N=V=1 caches with one shared RGB range (GEN3C: [-1,1]).

    Placement uses column vectors: p_destination = T_source @ T_object @ p_source.
    Object transforms act in source coordinates. To rotate/scale around a chosen
    source-space pivot, supply Translate(pivot) @ RotateScale @ Translate(-pivot).
    Static samples can be reused at every target time; there is no timeline API.
    Rendering requires image_size=(H,W). Depth ties within depth_tolerance
    destination units use source ID, canonical object ID, then source pixel.
    Frame/point chunks bound projection memory; outputs use the camera device.
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

    def _source(self, source_id):
        if source_id not in self.sources:
            raise ValueError(f"Unknown source ID {source_id!r}")
        cache = self.sources[source_id]
        if self._attachments[source_id] != self._attachment(cache):
            raise ValueError(f"Source attachment for {source_id!r} changed; rebuild the composite")
        if self._label_states[source_id] != self._label_state(cache):
            self.refresh_source(source_id)
        return cache

    def add_source(self, source_id, cache, is_environment=False):
        """Register unchanged child geometry; positive objects start unselected."""
        if not isinstance(source_id, str) or not source_id or source_id in self.sources:
            raise ValueError("Source IDs must be unique nonempty strings")
        if is_environment and self.environment_source_id is not None:
            raise ValueError("Exactly one source may supply the environment")
        self._validate_source(source_id, cache)
        self.sources[source_id] = cache
        self.selections[source_id] = frozenset({0} if is_environment else ())
        self.source_transforms[source_id] = self._identity()
        self._attachments[source_id] = self._attachment(cache)
        self._label_states[source_id] = self._label_state(cache)
        if is_environment:
            self.environment_source_id = source_id

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
        self.selections[source_id] = frozenset(selected)

    def set_transform(self, source_id, matrix, instance_id=None):
        """Set source placement or one positive object's source-space placement."""
        cache = self._source(source_id)
        matrix = self._similarity(matrix)
        if instance_id is None:
            if source_id == self.environment_source_id and not torch.equal(matrix, self._identity()):
                raise ValueError("Environment source transform must stay at identity")
            self.source_transforms[source_id] = matrix
        else:
            instance_id = self._resolve(source_id, instance_id, cache)
            if instance_id == 0:
                raise ValueError("Cannot transform environment ID 0")
            self.object_transforms[source_id, instance_id] = matrix
            self._placement_conflicts.discard((source_id, instance_id))

    def _iter_render_inputs(self):
        """Yield selected valid samples in deterministic source/object order.

        Check attachments and refresh changed label snapshots before yielding. A
        label correction may merge selected IDs; differing placements on merged
        objects require an explicit set_transform() to resolve the conflict.
        Empty objects yield no samples. No target time is needed for static data.
        """
        if self.environment_source_id is None:
            raise ValueError("A designated environment source is required before rendering")
        inputs = []
        for source_id in sorted(self.sources):
            cache = self._source(source_id)
            selected = self.selections[source_id]
            if any((source_id, i) in self._placement_conflicts for i in selected):
                raise ValueError("Merged objects have conflicting transforms; call set_transform again")
            inputs.append((source_id, cache, selected))
        for source_id, cache, selected in inputs:
            points = cache.input_points.reshape(-1, 3)
            rgb = cache.input_image[0, 0, 0, 0].permute(1, 2, 0).reshape(-1, 3)
            labels = cache.input_labels.reshape(-1)
            valid = cache.input_mask.reshape(-1).bool() & torch.isfinite(points).all(-1)
            for instance_id in sorted(selected):
                indices = torch.nonzero(valid & (labels == instance_id), as_tuple=True)[0]
                if not indices.numel():
                    continue
                transform = (self.source_transforms[source_id] @ self.object_transforms.get(
                    (source_id, instance_id), self._identity())).to(torch.float32)
                transformed = points[indices].to(torch.float32) @ transform[:3, :3].T + transform[:3, 3]
                yield CompositeSamples(source_id, instance_id, transformed, rgb[indices], indices)

    def _resolve_visibility(self, points, poses, intrinsics):
        """Replaceable visibility policy; ranks follow source/object/pixel order."""
        return nearest_surface_winners(
            points, poses, intrinsics, self.image_size,
            depth_tolerance=self.depth_tolerance, point_chunk_size=self.point_chunk_size,
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
        for matrix in (target_w2cs, target_intrinsics):
            if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
                raise ValueError("Target cameras must contain finite floating matrices")
        pose, k = target_w2cs[0].double(), target_intrinsics[0].double()
        if (not torch.allclose(pose[:, 3], pose.new_tensor([0, 0, 0, 1]).expand(len(pose), -1), atol=1e-5, rtol=0)
                or not torch.allclose(pose[:, :3, :3].transpose(-1, -2) @ pose[:, :3, :3],
                                      torch.eye(3, device=pose.device, dtype=pose.dtype).expand(len(pose), -1, -1),
                                      atol=1e-3, rtol=0)
                or not torch.allclose(torch.linalg.det(pose[:, :3, :3]), pose.new_ones(len(pose)), atol=1e-3, rtol=0)):
            raise ValueError("Target w2cs must be right-handed rigid OpenCV world-to-camera matrices")
        if (not torch.allclose(k[:, 2], k.new_tensor([0, 0, 1]).expand(len(k), -1), atol=1e-5, rtol=0)
                or (k[:, 0, 0] <= 0).any() or (k[:, 1, 1] <= 0).any() or (k[:, 1, 0].abs() > 1e-5).any()):
            raise ValueError("Target intrinsics must use the OpenCV pinhole convention")
        return target_w2cs[0].float(), target_intrinsics[0].float()

    @torch.no_grad()
    def _render(self, target_w2cs, target_intrinsics, start_frame_idx, diagnostics=False):
        poses, intrinsics = self._target_cameras(target_w2cs, target_intrinsics, start_frame_idx)
        # Diffusion may call from an autocast context; visibility needs float32.
        with torch.autocast(device_type=poses.device.type, enabled=False):
            return self._render_frames(poses, intrinsics, diagnostics)

    def _render_frames(self, poses, intrinsics, diagnostics):
        # Iterator order is exactly the categorical tie priority. Keep the static
        # sample table on CPU and transfer only projection chunks or winning RGB.
        samples = list(self._iter_render_inputs())
        points = torch.cat([s.points for s in samples]) if samples else torch.empty(0, 3, dtype=torch.float32)
        colors = torch.cat([s.rgb.float() for s in samples]) if samples else torch.empty(0, 3, dtype=torch.float32)
        if self.rgb_range != (-1., 1.):
            colors = (colors - self.rgb_range[0]) * (2 / (self.rgb_range[1] - self.rgb_range[0])) - 1
        frames, (height, width) = len(poses), self.image_size
        device = poses.device
        rgb = torch.full((frames, height, width, 3), -1., device=device, dtype=torch.float32)
        depth = torch.zeros((frames, height, width), device=device, dtype=torch.float32)
        coverage = torch.zeros_like(depth)
        details = None
        if diagnostics:
            source_ids = tuple(sorted(self.sources))
            source_index = {source_id: i for i, source_id in enumerate(source_ids)}
            metadata = torch.cat([
                torch.stack([torch.full_like(s.pixel_indices, source_index[s.source_id]),
                             torch.full_like(s.pixel_indices, s.instance_id), s.pixel_indices], dim=-1)
                for s in samples
            ]) if samples else torch.empty(0, 3, dtype=torch.int64)
            details = dict(source_ids=source_ids, winners=torch.full(
                (1, frames, height, width, 3), -1, dtype=torch.int64, device=device),
                nearest_depth=torch.zeros_like(depth)[None, :, None])
        for start in range(0, frames, self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, frames)
            winner, nearest = self._resolve_visibility(points, poses[start:end], intrinsics[start:end])
            frame, y, x = torch.nonzero(winner >= 0, as_tuple=True)
            rank = winner[frame, y, x].cpu()
            chosen = points[rank].to(device)
            # Camera-Z is from the chosen point, which can differ from the
            # numerical minimum within tolerance. All buffers use that winner.
            z = (chosen * poses[start:end][frame, 2, :3]).sum(-1) + poses[start:end][frame, 2, 3]
            rgb[start + frame, y, x] = colors[rank].to(device)
            depth[start + frame, y, x] = z
            coverage[start + frame, y, x] = 1
            if diagnostics:
                details['winners'][0, start + frame, y, x] = metadata[rank].to(device)
                details['nearest_depth'][0, start:end, 0] = nearest
        return rgb.permute(0, 3, 1, 2)[None, :, None], coverage[None, :, None, None], depth[None, :, None], details

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        """Render float32 RGB/depth and binary coverage on the target camera device.

        RGB [1,T,1,3,H,W] uses [-1,1], depth [1,T,1,H,W] is camera-Z,
        coverage [1,T,1,1,H,W] is shared by both. Hole values are -1/0/0.
        Static sources repeat at any nonnegative start_frame_idx.
        """
        rgb, coverage, depth, _ = self._render(target_w2cs, target_intrinsics, start_frame_idx)
        return (depth if render_depth else rgb), coverage

    def render_diagnostics(self, target_w2cs, target_intrinsics, start_frame_idx=0):
        """Return depth, coverage, nearest_depth, and categorical winner metadata.

        winners [1,T,H,W,3] stores (source index into source_ids, canonical ID,
        original pixel index), with -1 for holes. IDs always remain int64.
        """
        _, coverage, depth, details = self._render(target_w2cs, target_intrinsics, start_frame_idx, diagnostics=True)
        return dict(details, depth=depth, coverage=coverage)
