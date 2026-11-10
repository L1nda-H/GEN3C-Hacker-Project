# Cache3D_Composite implementation plan

## Qualities
- Use [SAM 2.1](https://github.com/facebookresearch/sam2) for all instance segmentation and video mask propagation.
- Wrap existing single-view, multiview, and dynamic caches; preserve each source's views and frames.
- Give each object a persistent `(source_id, instance_id)`; reserve `0` for environment and `-1` for unresolved labels. Require one designated environment source.
- Keep instance labels separate from geometry-validity masks; preserve source caches when changing selections or placement.
- Expose the existing `render_cache()` camera interface, with explicit object selections and a replaceable fusion policy.

## Implementation steps
1. **Segment, match, and label inputs.** Use `SAM21Segmenter` in `instance_segmentation.py` and add `instance_association.py` under `cosmos_predict1/diffusion/inference/`. Produce reusable integer label maps and a per-source object registry before constructing child caches. For videos, propagate SAM masks before final cross-view association and labeling.

2. **Attach labels to child caches.** In `cache_3d.py`, extend `Cache3D_Base.__init__()` to accept integer labels aligned with `input_points`; retain `_compute_input_points()` and `unproject_points()` geometry unchanged. Extend `render_cache()` and subclass overrides to accept selected IDs. In `Cache3D_Buffer.update_cache()`, maintain labels alongside inserted/replaced points. Existing `Cache3D_Labeled` motion flags do not replace instance IDs.

3. **Build the composite.** Add `cache3d_composite.py` containing `Cache3D_Composite`: `add_source()`, `set_selection()`, `set_transform()`, `set_time_mapping()`, and `render_cache()`. Store child caches by source ID, object metadata, source/object placement transforms, and dynamic time mappings. Place selected geometry in one destination coordinate system, including explicit scale calibration; repeat static sources across requested times.

## Step 1: segmentation, matching, and labeling

Status: input alignment and the SAM 2.1 image wrapper/checkpoint loader are implemented. Real-checkpoint accuracy validation, GUI integration, geometric association, video propagation, persistence, and cache label attachment remain pending. The video and geometric snippets below remain partial integration sketches.

### 1. Align inputs

Current status:

- **Complete:** `prepare_source()` preserves `(frame_idx, view_id)` observations and packs frame/view inputs for existing child caches.
- Validates alignment and OpenCV cameras; applies crop/resize with updated intrinsics and integer-safe half-pixel nearest sampling for depth, masks, and labels.
- Preserves geometry validity as `input_mask` and converts ray distance to camera-Z using existing unprojection. Callers ensure synchronization and consistent source coordinates/units.
- **Verified:** 8 CPU tests pass, including noninteger resize regressions and cache geometry checks. Cross-view association and cache label attachment are next-stage work.

Requirements:

- Retain `prepare_source()` in `instance_segmentation.py` as the preprocessing adapter for one source: a single image, video, static multiview set, or synchronized multiview videos.
- Index each observation's RGB, depth, intrinsics, world-to-camera pose, and optional masks/labels by `(frame_idx, view_id)`; preserve this mapping through segmentation and cache construction.
- Validate pixel alignment and camera conventions. If cropping/resizing is needed, apply matching spatial transforms to RGB, depth, masks, and labels; update intrinsics and use nearest-neighbor interpolation for categorical labels and validity masks. Accept already aligned inputs directly.
- Preserve finite, positive depth validity, combined with any supplied geometry mask, before resizing or the cache's depth sanitization. Keep validity separate from instance labels and pass it as `input_mask` at cache construction.
- Require consistent coordinates and scale within the source. Reuse `unproject_points(..., is_depth=...)` for camera-Z or ray-distance input; ensure depth used for association is camera-Z, converting ray-distance values when necessary. Cross-source placement and scale calibration belong to the composite stage.
- Reuse `Cache3D_Base.__init__()` for tensor dimension normalization, storage, depth sanitization, and existing depth-reliability filtering; reuse `_compute_input_points()` / `unproject_points()` for geometry. Extend label handling in step 6 rather than duplicating these cache operations in `prepare_source()`.

### 2. Segment instances with SAM 2.1

Current status:

- **Implemented:** `SAM21Segmenter.from_checkpoint()` constructs the official image predictor and automatic mask generator. `segment_source()` retains prepared observation keys, grouped prompts, optional quality metadata, names, ambiguity, unassigned regions, and confirmed environment masks.
- **Verified:** 25 local tests cover input alignment/cache geometry and mocked public SAM API contracts. The checkpoint factory is tested with CUDA as its default device. The 399/400-pixel alternate-mask regression now produces one instance without duplicate-induced ambiguity. Separate contained prompt groups remain separate.

Requirements:

- Use `SAM21Segmenter` as a thin wrapper around `SAM2ImagePredictor` and `SAM2AutomaticMaskGenerator`. `requirements.txt` includes SAM as a pipeline dependency, pins the official repository at `2b90b9f5ceec907a1c18123530e92e794ad901a4`, and preserves GEN3C's existing dependency pins. `SAM21_MODELS` pairs each 2.1 checkpoint variant with its matching packaged configuration. Weights are supplied explicitly; loading never silently downloads them.
- Refactor `segment_source()` to consume `PreparedSource`, preserve `(frame_idx, view_id)` keys, and convert prepared RGB to uint8 HWC. Let SAM's predictor perform internal resizing and return masks at prepared-image resolution; remove the generic adapter's independent backend resizing/prompt scaling.
- Use `SAM2AutomaticMaskGenerator.generate()` for class-agnostic candidate discovery. Pass quality/stability, point sampling, and box-NMS options directly through `automatic_mask_options`. Keep only project-specific overlap/boundary uncertainty handling after SAM postprocessing. Missing objects need explicit prompts; automatic NMS cannot guarantee preservation of every contained object.
- For GUI initialization/refinement, call `SAM2ImagePredictor.set_image()` and `predict()` with grouped object-specific clicks/boxes in prepared-image coordinates. Use `multimask_output=False` for a single mask, or select the best-quality alternative per prompt group. Different hypotheses for one prompt are not distinct objects.
- Return boolean `[H, W]` masks with observation-local IDs and SAM quality scores. These are predicted IoU/stability, not calibrated identity confidence. Retain `track_id` when supplied by the future video path. `InstanceObservation` has no appearance-vector requirement and allows missing confidence for propagated frames because the public video API returns mask logits, not calibrated per-frame quality scores.
- Keep segmentation independent of geometry validity. Preserve unresolved overlaps as ambiguous and unclaimed regions as unassigned. Establish environment regions through GUI mask prompts/painting or user confirmation of the residual after reviewing object masks; label 0 only on confirmed, valid, unambiguous pixels. Automatic proposals alone do not confirm the environment.
- Expose object overlays/list entries with default names such as “Object 3” and optional user names. After association/labeling, GUI selection addresses canonical IDs; mask corrections update affected association and point labels while retaining an existing ID when identity is unchanged.
- SAM-wrapper contract tests cover keyed prompts and object grouping, prepared-grid masks, one chosen nonempty hypothesis per prompt, retained empty observations, independent depth validity, uncertainty/environment handling, and replacement of an object's prompts while preserving its explicit local ID. GUI corrections themselves remain unimplemented. Inspect real masks on single-image and two-view examples.

Implemented scaffold changes in `instance_segmentation.py`:

- Replaced `SegmenterAdapter` with `SAM21Segmenter`; `SegmentPrompts` describes one object's clicks/box, explicit positive `local_id`, optional name, and multimask setting. All prompts stay in prepared-image coordinates; SAM handles its own resizing.
- Removed `AppearanceEncoderAdapter`, appearance-vector validation, and the mandatory appearance field/argument from `InstanceObservation` / `segment_source()`.
- Removed `clean_instance_proposals()` and its custom duplicate-suppression tests. Only unresolved overlap/boundary masks are computed locally; proposal filtering is delegated to SAM.

Install into an existing GEN3C environment (Python >=3.10; current torch 2.6.0 and torchvision 0.21.0 pins satisfy SAM's minimums):

```bash
python -m pip install -r requirements.txt
```

SAM can build an optional CUDA postprocessing extension; follow the [official installation notes](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md) for matching CUDA/PyTorch tooling on GCP. Download the [official large 2.1 checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) to `checkpoints/sam2.1_hiera_large.pt`. Validate image quality on representative project inputs rather than inferring it from the upstream video benchmark.

Image wrapper usage (`prepared` is the output of `prepare_source()`):

```python
import torch
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    SAM21Segmenter, SegmentPrompts, segment_source,
)

segmenter = SAM21Segmenter.from_checkpoint(
    "checkpoints/sam2.1_hiera_large.pt", model_size="large", device="cuda",
    automatic_mask_options={"pred_iou_thresh": 0.8, "stability_score_thresh": 0.95},
)
with torch.autocast("cuda", dtype=torch.bfloat16):
    # Automatic discovery on every prepared observation:
    discovered = segment_source(prepared, segmenter, rgb_range=(-1, 1))
    # Prompted replacement for one observation; each group describes one object.
    refined = segment_source(
        prepared, segmenter, rgb_range=(-1, 1),  # Use the actual prepared RGB range.
        prompts={(frame_idx, view_id): [SegmentPrompts(
            local_id=3, points=torch.tensor([[x, y]]), point_labels=torch.tensor([1]),
            name="Selected object", multimask_output=True,
        )]},
    )
```

Multimask output selects the highest-quality nonempty alternative **within one group**. Separate groups are never merged by local NMS. Prompted calls replace that observation's object list; supply all objects to retain, reusing IDs with their full revised clicks/boxes. An explicit empty list requests no objects; omitted observation keys use automatic discovery. Automatic IDs are provisional enumeration values and may change after rediscovery. Reusing a prompt ID across independent images does not establish identity.

Pass user-confirmed boolean `[H,W]` masks through `environment_masks`; SAM never confirms environment automatically. Rerunning segmentation returns a new result, leaves source geometry unchanged, and requires downstream association/label invalidation once those stages exist. Shared predictors have mutable image state; serialize access to a wrapper. Run the contract suite with `python -m unittest discover -s tests -p 'test_*seg*.py' -v`.

### 3. Match views geometrically

Current status:

- **Pending:** `instance_association.py`, geometric correspondences, pair scoring, and assignment are not implemented.
- Consume `SegmentedSource`, reusing its `prepared_source` cameras/depth/validity and SAM masks, optional quality scores, and scoped video track IDs. Keep segmentation results and child caches unchanged.

Requirements:

- Implement `match_source_views()` in `instance_association.py` to enumerate observation pairs within one source. For videos, run SAM propagation in step 4 before final cross-view association; match synchronized views at the same frame. Allow different capture times only for explicitly static content. Single-view sources retain their candidates without cross-view links; unrelated scenes remain independent.
- Implement `visible_correspondences()` using existing `unproject_points()` / `project_points()` in `forward_warp_utils_pytorch.py`. Unproject valid camera-Z samples into the source's shared world frame and project into the destination camera. Exclude ambiguous pixels in both observations; require valid destination depth. Reject nonfinite coordinates, points behind the camera, and out-of-grid samples before gathering; use nearest sampling consistent with integer pixel centers.
- Test projected camera-Z against destination depth with configurable absolute and relative tolerance: `abs(z_projected - z_observed) <= abs_tol + rel_tol * z_observed`. Treat points behind an observed foreground surface as occluded. Invalid depth, occlusion, and missing coverage supply no matching evidence; distinguish these from a supported mismatch.
- Resolve projected point collisions with nearest-depth selection and count unique destination pixels, preventing multiple source samples from inflating support. Compute correspondences in both directions and report depth residuals, support counts, and visible-support fractions.
- Implement `score_instance_pairs()` using bidirectional mask agreement on visible, valid, unambiguous surfaces, depth residuals, and available SAM mask quality. Require minimum support/count fractions and overlap quality; exclude unassigned/ambiguous destination pixels. Do not use a separate appearance encoder or semantic class names for association.
- Implement `associate_instances()` with deterministic maximum-score one-to-one assignment per observation pair, allowing unmatched candidates. Gate edges by support, score, and runner-up margins on both sides. With absent/insufficient overlap, retain separate candidates or accept an explicit GUI identity link; do not infer identity from matching native IDs in independent SAM sessions.
- Return an `AssociationResult` containing accepted links between `(frame_idx, view_id, local_id)` keys, unmatched/ambiguous candidates, and score/support diagnostics. Keep confirmed environment masks outside object assignment. Flag multi-view link conflicts that would identify two distinct objects in one observation as the same object; canonical IDs and registry merges remain step 5 work.
- Verify with synthetic calibrated scenes: known same-object views, two similar objects, partial occlusion, no overlap, invalid depth, ambiguous masks, unequal resolutions, point collisions, noisy depth, assignment ties, empty/single-view inputs, and static/dynamic pair eligibility. Inspect correspondence overlays and accepted/rejected links on real segmentation outputs after backend validation.

**Partial implementation sketch:** this illustrates one-direction projection and depth agreement for points already produced by `unproject_points()`. It does not yet implement destination geometry-mask/ambiguity filtering, explicit finite-coordinate checks before indexing, collision resolution, bidirectional scoring, or instance assignment. Complete the requirements above before integration.

```python
# points: [H, W, 3]; source_valid, dst_depth: [H, W]
# dst_w2c: [4, 4]; dst_K: [3, 3]; depth tolerances are configurable.
def visible_correspondences(points, source_valid, dst_depth,
                            dst_w2c, dst_K, abs_tol, rel_tol):
    """Partial sketch: project_points is the existing GEN3C helper."""
    p, camera = project_points(
        points[None], dst_w2c[None], dst_K[None],
        return_cam_points=True,
    )
    z = camera[0, ..., 2]
    uv = p[0, ..., :2, 0] / z.clamp_min(1e-6)[..., None]
    x, y = uv.round().long().unbind(-1)
    h, w = dst_depth.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    observed = dst_depth[y.clamp(0, h - 1), x.clamp(0, w - 1)]
    visible = (source_valid & inside & torch.isfinite(z) & (z > 0)
               & torch.isfinite(observed) & (observed > 0)
               & ((z - observed).abs() <= abs_tol + rel_tol * observed))
    return x, y, visible  # Only use returned pixel indices where visible is true.
```

Accepted links combine per-view SAM tracks/candidates into canonical identities. Once labels are attached to cache points, selecting one canonical object includes its observations from all matched views; placement, visibility, and rendering fusion occur later in the composite.

### 4. Propagate video masks with SAM 2.1

Current status: integration pending. Use SAM's video predictor instead of implementing a separate tracking/reidentification algorithm.

Requirements:

- Build one `SAM2VideoPredictor` session per source camera using `build_sam2_video_predictor()` with the SAM 2.1 checkpoint/configuration. Export prepared frames in chronological order and preserve the mapping from SAM's zero-based frame offsets to original observation keys.
- Seed objects from selected automatic/image masks with `add_new_mask()`, or from GUI clicks/boxes with `add_new_points_or_box()`. Assign distinct session object IDs; propagate masks using `propagate_in_video()` and threshold output logits at zero.
- Store native IDs as tracks scoped by `(source_id, view_id, session_id, obj_id)`. Map tracks and accepted cross-view links into canonical IDs in step 5; the same numeric ID from two sessions does not imply the same object.
- Add newly discovered/user-selected objects as new SAM seeds. Apply corrections through SAM's APIs and rerun affected propagation. A lost/split track remains unresolved until supported association or GUI confirmation; add no separate learned reidentification model. The pinned revision includes the December 2024 predictor update supporting new objects after propagation starts; do not substitute an older September-only checkout.
- Verify frame-key mapping, two-object propagation, temporary occlusion, new-object seeds, and correction behavior. Inspect real video overlays before relying on track continuity.

Official video API sketch (partial integration; frame-key mapping and result storage omitted):

```python
from sam2.build_sam import build_sam2_video_predictor

video = build_sam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint_path,
)
state = video.init_state(video_path=prepared_frame_directory)
video.add_new_mask(
    inference_state=state, frame_idx=seed_offset,
    obj_id=session_object_id, mask=seed_mask,
)
for frame_offset, object_ids, mask_logits in video.propagate_in_video(state):
    masks = mask_logits > 0  # Resolve frame_offset through the saved key mapping.
```

### 5. Assign canonical labels and persist
Implement `ObjectRegistry` with `allocate()`, `bind_observation()`, and `merge_ids()`; allocate positive IDs within each source and preserve observation-to-ID mappings. Track merges/corrections update all affected maps; names are user-supplied or generic. Rasterize accepted disjoint masks; reserve `0` for confirmed, valid, unambiguous environment pixels and `-1` for unresolved/invalid regions. Persist integer maps and validity in NPZ, and names, optional quality scores, observations, ID mappings, SAM code revision, configuration, and checkpoint hash in JSON.

```python
# Rasterize one image; `observations` contains only its instance masks.
labels = torch.full(valid.shape, -1, dtype=torch.int64)
labels[valid & environment_mask & ~uncertain] = 0
for obs in observations:
    key = (obs.frame_idx, obs.view_id, obs.local_id)
    instance_id = observation_to_id.get(key)
    if instance_id is not None:
        labels[valid & obs.mask & ~uncertain] = instance_id
# `environment_mask` is confirmed and excludes candidates; uncertainty includes conflicts.
```

### 6. Attach to cache points and check consistency
Add label-loading arguments at cache construction in `gen3c_multiview.demo()` and `Gen3cPersistentModel.seed_model_from_values()`. Normalize labels with the same dimension mapping as RGB into `[B, F, N, V, 1, H, W]`; the static multiview path stores source views along `N`. In `Cache3D_Base.__init__()`, store them as `int64`, separate from `input_mask`. Pixel `(y, x)` labels `input_points[..., y, x, :]` directly, so unprojection mathematics stays unchanged. Keep separate view samples carrying the same canonical ID.

Add `check_label_consistency()` to reproject labels and flag disagreements on mutually visible surfaces, excluding unknowns. Verify with a two-view scene containing two similar objects, plus a moving object that disappears and reappears. Also check environment coverage, label/RGB alignment, and save/load ID preservation. First deliverable: saved label maps, registry, and disagreement overlays for one source; then attach them to a child cache.
