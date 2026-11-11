"""Single-camera SAM 2.1 propagation, retaining prepared observation keys."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from PIL import Image
import torch

from cosmos_predict1.diffusion.inference.instance_segmentation import (
    SAM21_MODELS, SAM21_REVISION, InstanceProposal, ModelProvenance, PreparedSource, SegmentedSource, SegmentPrompts,
    file_sha256,
    _sam_mask, _sam_prompts, _sam_rgb, _segmented_observation,
)


class SAM21VideoSegmenter:
    """One source/camera/session per wrapper; public state operations are serialized.

    Object IDs are caller-owned positive integers passed unchanged to SAM. Each
    propagation returns a fresh snapshot; edits invalidate ``result``. Future
    labels/render masks must be rebuilt from that snapshot (steps 5/6).
    """

    def __init__(self, predictor, *, provenance=None):
        self.predictor = predictor
        self.provenance = provenance
        self._lock = RLock()
        self.state = None
        self.result = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, model_size="large", device="cuda"):
        if model_size not in SAM21_MODELS:
            raise ValueError(f"model_size must be one of {tuple(SAM21_MODELS)}")
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"SAM 2.1 checkpoint does not exist: {checkpoint_path}")
        from cosmos_predict1.utils.model_config import model_config_session

        with model_config_session(sam_config=SAM21_MODELS[model_size][0]):
            try:
                from sam2.build_sam import build_sam2_video_predictor
            except ModuleNotFoundError as exc:
                if exc.name == "sam2" or (exc.name or "").startswith("sam2."):
                    raise ImportError("Install SAM 2.1 with pip install -r requirements.txt") from exc
                raise
            return cls(build_sam2_video_predictor(
                SAM21_MODELS[model_size][0], str(checkpoint_path), device=device, mode="eval",
            ), provenance=ModelProvenance(SAM21_REVISION, SAM21_MODELS[model_size][0], file_sha256(checkpoint_path)))

    @torch.inference_mode()
    def initialize(self, source: PreparedSource, *, source_id: str, rgb_range: tuple[float, float]):
        """Export numeric JPEGs for SAM; sparse original frame keys remain explicit.

        SAM loads frames synchronously, so temporary files can be removed after
        init_state. RGB/depth/cameras/validity in the prepared source are untouched.
        Reinitializing starts a distinct identity namespace.
        """
        with self._lock:
            keys = tuple(sorted(source.observations))
            if not keys or len({key[1] for key in keys}) != 1:
                raise ValueError("Video propagation requires one nonempty camera stream")
            if len({tuple(obs.rgb.shape) for obs in source.observations.values()}) != 1:
                raise ValueError("Video propagation requires a common prepared image size")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("source_id must be a nonempty string")
            with TemporaryDirectory(prefix="sam21-") as directory:
                for offset, key in enumerate(keys):
                    Image.fromarray(_sam_rgb(source.observations[key].rgb, rgb_range)).save(
                        Path(directory) / f"{offset:08d}.jpg", quality=100, subsampling=0,
                    )
                state = self.predictor.init_state(video_path=directory, async_loading_frames=False)
            self.source, self.source_id = source, source_id
            self._rgb_range = tuple(rgb_range)
            self.frame_keys = keys
            self._offsets = {key: offset for offset, key in enumerate(keys)}
            self.session_id = uuid4().hex
            self.state = state
            self.objects = {}  # ID -> name, including invisible objects.
            self._seeds = {}  # (offset, ID) -> public SAM API arguments.
            self._propagated = False
            self.result = None

    def seed_objects(self, key, objects):
        """Seed step-2 InstanceObservations or grouped SegmentPrompts at ``key``.

        Instance masks seed geometry-independent tracks. Prompt groups use full
        replacement clicks/box; video SAM returns one mask per object.
        """
        with self._lock:
            objects = tuple(objects)
            if len({obj.local_id for obj in objects}) != len(objects):
                raise ValueError("Seed objects require unique local_id values")
            for obj in objects:
                self._seed(key, obj, correction=False)

    def correct_object(self, key, obj):
        """Replace this object's mask or full prompt set, retaining its track ID."""
        with self._lock:
            self._seed(key, obj, correction=True)

    @torch.inference_mode()
    def _seed(self, key, obj, *, correction):
        if self.state is None:
            raise RuntimeError("Initialize a video source first")
        if key not in self._offsets:
            raise ValueError("Seed references an unknown observation key")
        obj_id = obj.local_id
        if type(obj_id) is not int or obj_id <= 0:
            raise ValueError("Seed objects require a positive integer local_id")
        if correction and obj_id not in self.objects:
            raise ValueError("Corrections require an existing object ID")
        obs = self.source.observations[key]
        if isinstance(obj, SegmentPrompts):
            args = _sam_prompts(obj, obs.rgb.shape[-2:])
            method = "add_new_points_or_box"
            args = dict(points=args['point_coords'], labels=args['point_labels'], box=args['box'],
                        clear_old_points=True, normalize_coords=True)
        else:
            if (obj.frame_idx, obj.view_id) != key:
                raise ValueError("Seed mask observation does not match its key")
            method = "add_new_mask"
            args = dict(mask=_sam_mask(obj.mask, obs.rgb.shape[-2:], obs.rgb.device))
        if obj.name is not None and not isinstance(obj.name, str):
            raise ValueError("Object name must be a string")
        self.result = None
        offset = self._offsets[key]
        getattr(self.predictor, method)(inference_state=self.state, frame_idx=offset, obj_id=obj_id, **args)
        self.objects[obj_id] = obj.name if obj.name is not None else self.objects.get(obj_id, f"Object {obj_id}")
        self._seeds[offset, obj_id] = (method, args)

    @torch.inference_mode()
    def propagate(self, *, environment_masks=None, boundary_width=0) -> SegmentedSource:
        """Track the full clip in both directions, including before late seeds.

        After any earlier propagation, reset/replay public seed APIs to discard
        stale temporal memory. This MVP recomputes the whole clip after edits.
        Returned rows are resolved by SAM object IDs, never by row position.
        """
        with self._lock:
            if self.state is None or not self.objects:
                raise RuntimeError("Initialize and seed at least one object first")
            if type(boundary_width) is not int or boundary_width < 0:
                raise ValueError("boundary_width must be a nonnegative integer")
            environment_masks = environment_masks or {}
            if set(environment_masks) - set(self.frame_keys):
                raise ValueError("Environment masks reference unknown observation keys")
            self.result = None
            if self._propagated:
                self.predictor.reset_state(self.state)
                for (offset, obj_id), (method, args) in self._seeds.items():
                    getattr(self.predictor, method)(
                        inference_state=self.state, frame_idx=offset, obj_id=obj_id, **args,
                    )
            # Also replay after an interrupted/failed propagation.
            self._propagated = True
            first = {obj_id: min(t for t, i in self._seeds if i == obj_id) for obj_id in self.objects}
            masks = {}
            for reverse, start in ((False, min(first.values())), (True, max(first.values()))):
                if reverse and start == 0:
                    continue
                for offset, ids, logits in self.predictor.propagate_in_video(
                    self.state, start_frame_idx=start, reverse=reverse,
                ):
                    if not 0 <= offset < len(self.frame_keys):
                        raise ValueError("SAM returned an unknown frame offset")
                    ids = list(ids)
                    if len(set(ids)) != len(ids) or set(ids) != set(self.objects):
                        raise ValueError("SAM returned unexpected object IDs")
                    if tuple(logits.shape) != (len(ids), 1, *self.source.observations[self.frame_keys[offset]].rgb.shape[-2:]):
                        raise ValueError("SAM must return logits [objects,1,H,W] on the prepared grid")
                    obs = self.source.observations[self.frame_keys[offset]]
                    for obj_id, row in zip(ids, logits):
                        if not reverse or offset < first[obj_id]:
                            masks[offset, obj_id] = (row[0] > 0).to(obs.rgb.device).clone()
            results = {}
            for offset, key in enumerate(self.frame_keys):
                obs = self.source.observations[key]
                proposals = []
                for obj_id, name in self.objects.items():
                    if (offset, obj_id) not in masks:
                        raise ValueError("SAM did not cover every requested frame/object")
                    proposals.append(InstanceProposal(
                        masks[offset, obj_id], None, obj_id, name,
                        track_id=(self.source_id, key[1], self.session_id, obj_id),
                    ))
                results[key] = _segmented_observation(
                    key, obs, proposals, environment_masks.get(key), boundary_width,
                )
            provenance = self.provenance
            if provenance is not None:
                provenance = provenance.with_settings(mode="video", rgb_range=self._rgb_range,
                                                       boundary_width=boundary_width)
            self.result = SegmentedSource(self.source, results, provenance=provenance)
            return self.result
