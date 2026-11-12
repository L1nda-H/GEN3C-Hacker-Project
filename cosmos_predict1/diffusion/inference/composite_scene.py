"""Shared static scene specification and exact prepared-input persistence.

Paths in Python models are filesystem paths; JSON paths are relative to the
scene file. Loading never estimates depth, prepares images, or resizes labels.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base
from cosmos_predict1.diffusion.inference.cache3d_composite import Cache3D_Composite
from cosmos_predict1.diffusion.inference.camera_validation import validate_opencv_cameras
from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import unproject_points
from cosmos_predict1.diffusion.inference.instance_labeling import (
    load_labeled_source, prepared_source_fingerprint,
)
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    PreparedSource, SourceObservation, _validate_observation,
)


def _transform(matrix):
    # JSON numbers are doubles. Preserve their precision across save/load;
    # torch.as_tensor(list) would otherwise use the global float32 default.
    tensor = torch.as_tensor(matrix)
    if not tensor.is_complex() and tensor.dtype != torch.bool:
        tensor = torch.as_tensor(matrix, dtype=torch.float64)
    return Cache3D_Composite._similarity(tensor)


def _validate_prepared(source):
    if len(source.observations) != 1:
        raise ValueError("Composite MVP requires one static image per source")
    for key, obs in source.observations.items():
        if len(key) != 2 or any(type(i) is not int or i < 0 for i in key):
            raise ValueError("Observation keys must be nonnegative integer (frame, view) pairs")
        _validate_observation(obs)
        if obs.geometry_mask is None or obs.geometry_mask.dtype != torch.bool:
            raise ValueError("Prepared inputs require explicit boolean geometry validity")
        if (obs.geometry_mask & (~torch.isfinite(obs.depth) | (obs.depth <= 0))).any():
            raise ValueError("Prepared validity includes invalid camera-Z depth")
        if not torch.isfinite(obs.rgb).all() or (obs.rgb.abs() > 1).any():
            raise ValueError("Prepared RGB must be finite and in [-1,1]")


def save_prepared_source(path, source: PreparedSource):
    """Save the exact static PreparedSource used for labeling, without casting."""
    _validate_prepared(source)
    key, obs = next(iter(source.observations.items()))
    arrays = {name: getattr(obs, name).detach().cpu().numpy() for name in
              ('rgb', 'depth', 'intrinsics', 'w2c', 'geometry_mask')}
    metadata = dict(version=1, observation_key=key, preparation=dict(source.preparation),
                    fingerprint=prepared_source_fingerprint(source))
    # A file handle avoids NumPy silently appending .npz to the reference.
    with Path(path).open('wb') as handle:
        np.savez_compressed(handle, metadata=json.dumps(metadata, allow_nan=False), **arrays)


def load_prepared_source(path, expected_fingerprint):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays['metadata'].item()))
        if type(metadata['version']) is not int or metadata['version'] != 1:
            raise ValueError("Unsupported prepared-input version")
        obs = SourceObservation(**{name: torch.from_numpy(arrays[name].copy()) for name in
                                  ('rgb', 'depth', 'intrinsics', 'w2c', 'geometry_mask')})
    source = PreparedSource({tuple(metadata['observation_key']): obs}, metadata['preparation'])
    _validate_prepared(source)
    fingerprint = prepared_source_fingerprint(source)
    if fingerprint != expected_fingerprint or fingerprint != metadata['fingerprint']:
        raise ValueError("Prepared-source fingerprint mismatch")
    return source


@dataclass
class SceneSource:
    source_id: str
    prepared_path: str | Path
    labels_path: str | Path
    fingerprint: str
    selected_ids: list[int]
    source_transform: object = field(default_factory=lambda: torch.eye(4).tolist())
    object_transforms: dict[int, object] = field(default_factory=dict)


@dataclass
class SceneSpec:
    sources: list[SceneSource]
    environment_source: str
    target_w2cs: object
    target_intrinsics: object
    image_size: tuple[int, int]
    version: int = 1

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported scene version")
        if not self.sources or not all(isinstance(s, SceneSource) for s in self.sources):
            raise ValueError("Scene requires SceneSource entries")
        ids = [s.source_id for s in self.sources]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Source IDs must be unique nonempty strings")
        if self.environment_source not in ids:
            raise ValueError("Environment source must reference a scene source")
        if len(self.image_size) != 2 or any(type(n) is not int or n <= 0 for n in self.image_size):
            raise ValueError("image_size must be positive integer (H,W)")
        for source in self.sources:
            for path in (source.prepared_path, source.labels_path):
                if not isinstance(path, (str, Path)) or not str(path):
                    raise ValueError("Prepared and label artifact paths are required")
            if (not isinstance(source.fingerprint, str) or len(source.fingerprint) != 64
                    or any(c not in '0123456789abcdef' for c in source.fingerprint)):
                raise ValueError("Source fingerprint must be a SHA256 hex digest")
            if (not isinstance(source.selected_ids, (list, tuple))
                    or any(type(i) is not int or i < 0 for i in source.selected_ids)
                    or len(set(source.selected_ids)) != len(source.selected_ids)):
                raise ValueError("Explicit selected IDs must be unique nonnegative integers")
            if 0 in source.selected_ids and source.source_id != self.environment_source:
                raise ValueError("Only the environment source may select ID 0")
            if source.source_id == self.environment_source and 0 not in source.selected_ids:
                raise ValueError("Environment selection must explicitly include ID 0")
            matrix = _transform(source.source_transform)
            if source.source_id == self.environment_source and not torch.equal(matrix, torch.eye(4, dtype=matrix.dtype)):
                raise ValueError("Environment source transform must be identity")
            for instance_id, transform in source.object_transforms.items():
                if type(instance_id) is not int or instance_id <= 0 or instance_id not in source.selected_ids:
                    raise ValueError("Object transforms must reference selected positive IDs")
                _transform(transform)
        poses, intrinsics = self.cameras()
        if poses.ndim != 4 or poses.shape[0] != 1 or poses.shape[1] < 1 or poses.shape[2:] != (4, 4):
            raise ValueError("Target w2cs must have shape [1,T,4,4]")
        if intrinsics.shape != (*poses.shape[:2], 3, 3):
            raise ValueError("Target intrinsics must have shape [1,T,3,3]")
        validate_opencv_cameras(poses, intrinsics)
        return self

    def cameras(self, device='cpu'):
        return (torch.as_tensor(self.target_w2cs, dtype=torch.float32, device=device),
                torch.as_tensor(self.target_intrinsics, dtype=torch.float32, device=device))

    def to_dict(self):
        self.validate()
        return dict(version=self.version, environment_source=self.environment_source,
                    image_size=list(self.image_size), target_w2cs=self.cameras()[0].tolist(),
                    target_intrinsics=self.cameras()[1].tolist(), sources=[dict(
                        source_id=s.source_id, prepared_path=str(Path(s.prepared_path).resolve()),
                        labels_path=str(Path(s.labels_path).resolve()), fingerprint=s.fingerprint,
                        selected_ids=list(s.selected_ids),
                        source_transform=_transform(s.source_transform).tolist(),
                        object_transforms={str(i): _transform(t).tolist()
                                           for i, t in s.object_transforms.items()},
                    ) for s in self.sources])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if 'version' not in data:
            raise ValueError("Scene JSON requires an explicit version")
        sources = []
        for entry in data.pop('sources'):
            for name in ('prepared_path', 'labels_path'):
                entry[name] = str((path.parent / entry[name]).resolve())
            transforms = entry.get('object_transforms', {})
            if any(str(int(i)) != i for i in transforms):
                raise ValueError("Object transform keys must be canonical integer strings")
            entry['object_transforms'] = {int(i): t for i, t in transforms.items()}
            sources.append(SceneSource(**entry))
        return cls(sources=sources, **data)


def build_composite_scene(scene: SceneSpec, *, device='cpu'):
    """Validate persisted artifacts before constructing immutable static children."""
    scene.validate()
    composite = Cache3D_Composite(image_size=tuple(scene.image_size))
    # Retain provenance from validated labels; review writers need not reload JSON.
    composite.source_provenance = {}
    for source in scene.sources:
        prepared = load_prepared_source(source.prepared_path, source.fingerprint)
        labeled = load_labeled_source(source.labels_path, prepared)
        if labeled.registry.source_id != source.source_id:
            raise ValueError("Scene source ID differs from label registry ownership")
        kwargs = prepared.cache_kwargs(labeled_source=labeled)
        # Avoid the legacy cache's depth clamp (100 units): prepared geometry is
        # authoritative. Validity remains explicit even for omitted samples.
        obs = next(iter(prepared.observations.values()))
        kwargs['input_points'] = unproject_points(
            obs.depth[None, None].float(), obs.w2c[None].float(), obs.intrinsics[None].float(),
            mask=obs.geometry_mask[None],
        )
        child = Cache3D_Base(**kwargs, device=device)
        composite.add_source(source.source_id, child, is_environment=source.source_id == scene.environment_source)
        composite.set_selection(source.source_id, source.selected_ids)
        composite.set_transform(source.source_id, _transform(source.source_transform))
        for instance_id, matrix in source.object_transforms.items():
            composite.set_transform(source.source_id, _transform(matrix), instance_id=instance_id)
        composite.source_provenance[source.source_id] = dict(
            segmentation_revision=labeled.segmented_source.revision,
            provenance=None if labeled.segmented_source.provenance is None else asdict(labeled.segmented_source.provenance),
        )
    return composite
