"""Canonical source-local identities, geometry-aware labels, and NPZ/JSON persistence."""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.instance_segmentation import (
    InstanceObservation, ModelProvenance, PreparedSource, SegmentedObservation, SegmentedSource, file_sha256,
)


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty identifier string")
    return value


def _positive_id(value):
    if type(value) is not int or not 0 < value <= torch.iinfo(torch.int64).max:
        raise ValueError("Object IDs must be positive int64 integers")
    return value


def _observation_key(key):
    if len(key) != 3 or any(type(i) is not int or i < 0 for i in key[:2]):
        raise ValueError("Observation keys must be (frame_idx, view_id, positive local_id)")
    _positive_id(key[2])
    return tuple(key)


def _track_key(track, source_id, view_id=None):
    if (len(track) != 4 or track[0] != source_id or type(track[1]) is not int or track[1] < 0
            or (view_id is not None and track[1] != view_id)):
        raise ValueError("Track must belong to this source/camera namespace")
    _text(track[2])
    _positive_id(track[3])
    return tuple(track)


@dataclass
class ObjectRegistry:
    """Persistent IDs scoped to one source. Retired IDs remain resolvable aliases.

    Observation bindings include the segmentation revision; only scoped tracks
    or explicit caller links carry identity into a different revision.
    """

    source_id: str
    names: dict[int, str] = field(default_factory=dict)
    observation_to_id: dict[tuple[str, int, int, int], int] = field(default_factory=dict)
    track_to_id: dict[tuple[str, int, str, int], int] = field(default_factory=dict)
    aliases: dict[int, int] = field(default_factory=dict)
    revision_fingerprints: dict[str, str] = field(default_factory=dict)
    next_id: int = 1

    def __post_init__(self):
        _text(self.source_id)

    def allocate(self, name=None):
        instance_id = _positive_id(self.next_id)
        if name is not None and not isinstance(name, str):
            raise ValueError("Object name must be a string")
        self.names[instance_id] = name if name is not None else f"Object {instance_id}"
        self.next_id += 1
        return instance_id

    def resolve(self, instance_id):
        _positive_id(instance_id)
        seen = set()
        while instance_id in self.aliases:
            if instance_id in seen:
                raise ValueError("Cyclic merge aliases")
            seen.add(instance_id)
            instance_id = self.aliases[instance_id]
        if instance_id not in self.names:
            raise ValueError(f"Unknown canonical object ID {instance_id}")
        return instance_id

    def bind_observation(self, key, instance_id, *, revision, track_id=None):
        key = _observation_key(key)
        binding = (_text(revision), *key)
        instance_id = self.resolve(instance_id)
        track_id = None if track_id is None else _track_key(track_id, self.source_id, key[1])
        existing = self.observation_to_id.get(binding)
        track_existing = self.track_to_id.get(track_id)
        if any(value is not None and self.resolve(value) != instance_id for value in (existing, track_existing)):
            raise ValueError("Conflicting identity binding; merge IDs explicitly first")
        self.observation_to_id[binding] = instance_id
        if track_id is not None:
            self.track_to_id[track_id] = instance_id

    def merge_ids(self, retired_id, surviving_id):
        retired_id, surviving_id = self.resolve(retired_id), self.resolve(surviving_id)
        if retired_id != surviving_id:
            del self.names[retired_id]
            self.aliases[retired_id] = surviving_id
            self.aliases = {old: self.resolve(target) for old, target in self.aliases.items()}
            for bindings in (self.observation_to_id, self.track_to_id):
                bindings.update({key: self.resolve(value) for key, value in bindings.items()})
        return surviving_id


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _hash_tensor(digest, tensor):
    # uint8 view also supports bfloat16, which NumPy cannot represent directly.
    tensor = tensor.detach().cpu().contiguous()
    digest.update(_json_bytes([str(tensor.dtype), list(tensor.shape)]))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())


def prepared_source_fingerprint(source: PreparedSource):
    """Exact prepared pixels/geometry and preparation settings, excluding labels."""
    digest = hashlib.sha256(_json_bytes(dict(source.preparation)))
    for key, obs in sorted(source.observations.items()):
        digest.update(_json_bytes(key))
        for tensor in (obs.rgb, obs.depth, obs.intrinsics, obs.w2c, obs.geometry_mask):
            if tensor is None:
                raise ValueError("Expected prepared geometry validity")
            _hash_tensor(digest, tensor)
    return digest.hexdigest()


def _instance_metadata(instance):
    for quality in (instance.confidence, instance.stability_score):
        if quality is not None and not math.isfinite(quality):
            raise ValueError("Quality must be finite when supplied")
    if not isinstance(instance.name, str):
        raise ValueError("Object name must be a string")
    return dict(local_id=instance.local_id, name=instance.name, confidence=instance.confidence,
                stability_score=instance.stability_score, track_id=instance.track_id)


def _confirmed(obs):
    return obs.environment_mask if obs.confirmed_environment_mask is None else obs.confirmed_environment_mask


def _snapshot_fingerprint(segmented, source_fingerprint):
    digest = hashlib.sha256(source_fingerprint.encode())
    for key, obs in sorted(segmented.observations.items()):
        digest.update(_json_bytes(key))
        for instance in sorted(obs.instances, key=lambda i: i.local_id):
            digest.update(_json_bytes(_instance_metadata(instance)))
            _hash_tensor(digest, instance.mask)
        for mask in (obs.ambiguous_mask, obs.unassigned_mask, obs.environment_mask, _confirmed(obs)):
            _hash_tensor(digest, mask)
    return digest.hexdigest()


def _mask(mask, reference):
    if mask.dtype != torch.bool or mask.shape != reference.shape or mask.device != reference.device:
        raise ValueError("Segmentation/validity masks must be boolean on the prepared [H,W] grid/device")
    return mask


def _rasterize(key, observation, prepared, registry, revision):
    valid = prepared.geometry_mask
    if valid is None or valid.shape != prepared.depth.shape or valid.dtype != torch.bool:
        raise ValueError("Expected prepared boolean geometry validity [H,W]")
    uncertain = _mask(observation.ambiguous_mask, valid).clone()
    occupied = torch.zeros_like(valid)
    local_ids = set()
    for instance in observation.instances:
        _observation_key((*key, instance.local_id))
        if (instance.frame_idx, instance.view_id) != key or instance.local_id in local_ids:
            raise ValueError("Duplicate or mismatched instance observation key")
        local_ids.add(instance.local_id)
        mask = _mask(instance.mask, valid)
        uncertain |= occupied & mask
        occupied |= mask
    uncertain |= _mask(observation.unassigned_mask, valid)
    environment = _mask(observation.environment_mask, valid) & _mask(_confirmed(observation), valid)
    labels = torch.full_like(valid, -1, dtype=torch.int64)
    labels[valid & environment & ~occupied & ~uncertain] = 0
    for instance in observation.instances:
        instance_id = registry.observation_to_id[(revision, *key, instance.local_id)]
        labels[valid & instance.mask & ~uncertain] = registry.resolve(instance_id)
    return labels


@dataclass(frozen=True)
class LabeledSource:
    segmented_source: SegmentedSource
    registry: ObjectRegistry
    labels: dict[tuple[int, int], torch.Tensor]
    source_fingerprint: str

    @property
    def prepared_source(self):
        return self.segmented_source.prepared_source


def label_source(segmented: SegmentedSource, *, source_id: str, registry=None, identity_links=None) -> LabeledSource:
    """Label a snapshot without mutating it or the supplied registry.

    identity_links explicitly maps current (frame,view,local_id) to an existing
    canonical ID, e.g. after an image prompt correction. New video sessions and
    image rediscovery otherwise allocate new identities. After merges, call this
    again with the merged registry to rebuild all current maps.
    """
    _text(segmented.revision)
    if set(segmented.observations) != set(segmented.prepared_source.observations):
        raise ValueError("Segmentation must cover exactly the prepared observation keys")
    registry = ObjectRegistry(source_id) if registry is None else deepcopy(registry)
    if registry.source_id != source_id:
        raise ValueError("Registry belongs to a different source")
    fingerprint = prepared_source_fingerprint(segmented.prepared_source)
    snapshot = _snapshot_fingerprint(segmented, fingerprint)
    previous = registry.revision_fingerprints.setdefault(segmented.revision, snapshot)
    if previous != snapshot:
        raise ValueError("Changed segmentation needs a new revision")
    identity_links = identity_links or {}
    keys = {(*key, i.local_id) for key, obs in segmented.observations.items() for i in obs.instances}
    if set(identity_links) - keys:
        raise ValueError("Identity links reference unknown observations")
    labels = {}
    for key, obs in sorted(segmented.observations.items()):
        for instance in sorted(obs.instances, key=lambda i: i.local_id):
            local_key = (*key, instance.local_id)
            track = instance.track_id
            if track is not None:
                track = _track_key(track, source_id, key[1])
            instance_id = identity_links.get(local_key)
            if instance_id is None:
                instance_id = registry.observation_to_id.get((segmented.revision, *local_key))
            if instance_id is None:
                instance_id = registry.track_to_id.get(track)
            if instance_id is None:
                instance_id = registry.allocate(instance.name)
            registry.bind_observation(local_key, instance_id, revision=segmented.revision, track_id=track)
        labels[key] = _rasterize(key, obs, segmented.prepared_source.observations[key], registry, segmented.revision)
    return LabeledSource(segmented, registry, labels, fingerprint)


def _registry_metadata(registry):
    return dict(source_id=registry.source_id, next_id=registry.next_id,
                names=[[i, name] for i, name in sorted(registry.names.items())],
                aliases=sorted(registry.aliases.items()),
                observations=[[list(key), value] for key, value in sorted(registry.observation_to_id.items())],
                tracks=[[list(key), value] for key, value in sorted(registry.track_to_id.items())],
                revisions=registry.revision_fingerprints)


def _unique_pairs(rows):
    result = {}
    for key, value in rows:
        key = tuple(key) if isinstance(key, list) else key
        if key in result:
            raise ValueError("Duplicate registry entry")
        result[key] = value
    return result


def _load_registry(data):
    registry = ObjectRegistry(data['source_id'], names=_unique_pairs(data['names']),
                              aliases=_unique_pairs(data['aliases']), next_id=data['next_id'],
                              revision_fingerprints=data['revisions'])
    for instance_id, name in registry.names.items():
        _positive_id(instance_id)
        if not isinstance(name, str):
            raise ValueError("Object name must be a string")
    if set(registry.names) & set(registry.aliases):
        raise ValueError("Retired IDs cannot also be active")
    for old, target in registry.aliases.items():
        _positive_id(target)
        registry.resolve(old)
    allocated = set(registry.names) | set(registry.aliases)
    if type(registry.next_id) is not int or not max(allocated, default=0) < registry.next_id <= 2**63:
        raise ValueError("Invalid next object ID")
    for revision, fingerprint in registry.revision_fingerprints.items():
        _text(revision)
        _text(fingerprint)
    for binding, instance_id in _unique_pairs(data['observations']).items():
        if len(binding) != 4 or binding[0] not in registry.revision_fingerprints:
            raise ValueError("Observation binding references an unknown revision")
        registry.bind_observation(binding[1:], instance_id, revision=binding[0])
    for track, instance_id in _unique_pairs(data['tracks']).items():
        registry.track_to_id[_track_key(track, registry.source_id)] = registry.resolve(instance_id)
    return registry


def save_labeled_source(path: str | Path, result: LabeledSource):
    """Save matching .npz/.json siblings, including masks needed for later editing.

    Does not serialize SAM inference state or source RGB/depth/cameras. Loading
    requires the exact matching PreparedSource; SAM editing restarts from masks.
    """
    checked = label_source(result.segmented_source, source_id=result.registry.source_id, registry=result.registry)
    if (result.source_fingerprint != checked.source_fingerprint or result.registry != checked.registry
            or set(result.labels) != set(checked.labels)
            or any(result.labels[k].dtype != torch.int64 or not torch.equal(result.labels[k], checked.labels[k])
                   for k in checked.labels)):
        raise ValueError("Stale label maps; rebuild after merges, corrections, or source changes")
    path = Path(path)
    arrays, manifest = {}, []
    for index, (key, obs) in enumerate(sorted(result.segmented_source.observations.items())):
        prefix = f"observation_{index}"
        tensors = dict(labels=result.labels[key], validity=result.prepared_source.observations[key].geometry_mask,
                       ambiguous=obs.ambiguous_mask, unassigned=obs.unassigned_mask,
                       environment=obs.environment_mask, confirmed_environment=_confirmed(obs))
        for instance in obs.instances:
            tensors[f"object_{instance.local_id}"] = instance.mask
        for name, tensor in tensors.items():
            arrays[f"{prefix}_{name}"] = tensor.detach().cpu().numpy()
        manifest.append(dict(key=key, prefix=prefix, instances=[_instance_metadata(i) for i in obs.instances]))
    provenance = result.segmented_source.provenance
    metadata = dict(schema_version=1, source_fingerprint=result.source_fingerprint,
                    segmentation_revision=result.segmented_source.revision,
                    provenance=None if provenance is None else asdict(provenance),
                    registry=_registry_metadata(result.registry), observations=manifest)
    # Validate JSON before writing arrays; the NPZ digest detects interrupted or mismatched pairs.
    _json_bytes(metadata)
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    metadata['npz_sha256'] = file_sha256(path.with_suffix('.npz'))
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def load_labeled_source(path: str | Path, prepared: PreparedSource) -> LabeledSource:
    """Validate source identity and persistence invariants before exposing labels."""
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata['schema_version'] != 1:
        raise ValueError("Unsupported label schema version")
    if metadata['source_fingerprint'] != prepared_source_fingerprint(prepared):
        raise ValueError("Saved labels do not match the prepared-source fingerprint")
    if metadata['npz_sha256'] != file_sha256(path.with_suffix('.npz')):
        raise ValueError("NPZ/JSON artifact mismatch")
    registry = _load_registry(metadata['registry'])
    revision = _text(metadata['segmentation_revision'])
    if revision not in registry.revision_fingerprints:
        raise ValueError("Missing segmentation revision fingerprint")
    provenance = metadata['provenance']
    provenance = None if provenance is None else ModelProvenance(**provenance)
    observations, labels = {}, {}
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
        for entry in metadata['observations']:
            key = tuple(entry['key'])
            if key not in prepared.observations or key in observations:
                raise ValueError("Unknown or duplicate saved observation key")
            valid = prepared.observations[key].geometry_mask

            def tensor(name, dtype=torch.bool):
                value = torch.from_numpy(arrays[f"{entry['prefix']}_{name}"].copy()).to(valid.device)
                if value.dtype != dtype or value.shape != valid.shape:
                    raise ValueError("Saved array has incorrect dtype or prepared-grid shape")
                return value

            if not torch.equal(tensor('validity'), valid):
                raise ValueError("Saved validity differs from prepared geometry")
            instances = []
            for record in entry['instances']:
                record = dict(record)
                track = record['track_id']
                record['track_id'] = None if track is None else tuple(track)
                instance = InstanceObservation(*key, mask=tensor(f"object_{record['local_id']}"), **record)
                binding = (revision, *key, instance.local_id)
                if binding not in registry.observation_to_id:
                    raise ValueError("Missing observation identity binding")
                if track is not None and registry.track_to_id.get(tuple(track)) != registry.observation_to_id[binding]:
                    raise ValueError("Track/observation identity mismatch")
                instances.append(instance)
            observations[key] = SegmentedObservation(tuple(instances), tensor('ambiguous'), tensor('unassigned'),
                                                     tensor('environment'), tensor('confirmed_environment'))
            labels[key] = tensor('labels', torch.int64)
    segmented = SegmentedSource(prepared, observations, revision=revision, provenance=provenance)
    checked = label_source(segmented, source_id=registry.source_id, registry=registry)
    if any(not torch.equal(labels[key], checked.labels[key]) for key in labels):
        raise ValueError("Saved labels violate geometry, uncertainty, or registry bindings")
    return checked
