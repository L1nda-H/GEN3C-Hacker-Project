"""Selection and manual placement for static, labeled single-image caches.

The environment defines destination world coordinates. Cameras consuming these
samples use OpenCV world-to-camera conventions. Rendering/fusion is a separate
stage; no child cache tensors are changed by this wrapper.
"""

from dataclasses import dataclass
import math

import torch

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base


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
    """

    def __init__(self, *, rgb_range=(-1., 1.)):
        if (len(rgb_range) != 2 or not all(math.isfinite(x) for x in rgb_range)
                or rgb_range[0] >= rgb_range[1]):
            raise ValueError("RGB range must have two finite increasing bounds")
        self.rgb_range = tuple(rgb_range)
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
