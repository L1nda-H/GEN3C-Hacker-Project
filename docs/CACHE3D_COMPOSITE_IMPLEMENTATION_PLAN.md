# Cache3D_Composite implementation plan

## Qualities
- Wrap existing single-view, multiview, and dynamic caches; preserve each source's views and frames.
- Give each object a persistent `(source_id, instance_id)`; reserve `0` for environment and `-1` for unresolved labels. Require one designated environment source.
- Keep instance labels separate from geometry-validity masks; preserve source caches when changing selections or placement.
- Expose the existing `render_cache()` camera interface, with explicit object selections and a replaceable fusion policy.

## Implementation steps
1. **Segment, match, and label inputs.** Implement the six components below in new `instance_segmentation.py` and `instance_association.py` modules under `cosmos_predict1/diffusion/inference/`. Produce reusable integer label maps and a per-source object registry before constructing child caches.

2. **Attach labels to child caches.** In `cache_3d.py`, extend `Cache3D_Base.__init__()` to accept integer labels aligned with `input_points`; retain `_compute_input_points()` and `unproject_points()` geometry unchanged. Extend `render_cache()` and subclass overrides to accept selected IDs. In `Cache3D_Buffer.update_cache()`, maintain labels alongside inserted/replaced points. Existing `Cache3D_Labeled` motion flags do not replace instance IDs.

3. **Build the composite.** Add `cache3d_composite.py` containing `Cache3D_Composite`: `add_source()`, `set_selection()`, `set_transform()`, `set_time_mapping()`, and `render_cache()`. Store child caches by source ID, object metadata, source/object placement transforms, and dynamic time mappings. Place selected geometry in one destination coordinate system, including explicit scale calibration; repeat static sources across requested times.

## Step 1: segmentation, matching, and labeling

Status: input alignment and the backend-independent segmentation contract are complete; real-backend overlay validation, association, tracking, persistence, and label attachment remain pending. Remaining snippets are implementation sketches.

### 1. Align inputs

Current status:

- **Complete:** `prepare_source()` preserves `(frame_idx, view_id)` observations and packs frame/view inputs for existing child caches.
- Validates alignment and OpenCV cameras; applies crop/resize with updated intrinsics and integer-safe half-pixel nearest sampling for depth, masks, and labels.
- Preserves geometry validity as `input_mask` and converts ray distance to camera-Z using existing unprojection. Callers ensure synchronization and consistent source coordinates/units.
- **Verified:** 8 CPU tests pass, including noninteger resize regressions and cache geometry checks. Segmentation and cache label attachment are next-stage work.

Requirements:

- Implement `prepare_source()` in `instance_segmentation.py` as a thin preprocessing adapter for one source: a single image, video, static multiview set, or synchronized multiview videos.
- Index each observation's RGB, depth, intrinsics, world-to-camera pose, and optional masks/labels by `(frame_idx, view_id)`; preserve this mapping through segmentation and cache construction.
- Validate pixel alignment and camera conventions. If cropping/resizing is needed, apply matching spatial transforms to RGB, depth, masks, and labels; update intrinsics and use nearest-neighbor interpolation for categorical labels and validity masks. Accept already aligned inputs directly.
- Preserve finite, positive depth validity, combined with any supplied geometry mask, before resizing or the cache's depth sanitization. Keep validity separate from instance labels and pass it as `input_mask` at cache construction.
- Require consistent coordinates and scale within the source. Reuse `unproject_points(..., is_depth=...)` for camera-Z or ray-distance input; ensure depth used for association is camera-Z, converting ray-distance values when necessary. Cross-source placement and scale calibration belong to the composite stage.
- Reuse `Cache3D_Base.__init__()` for tensor dimension normalization, storage, depth sanitization, and existing depth-reliability filtering; reuse `_compute_input_points()` / `unproject_points()` for geometry. Extend label handling in step 6 rather than duplicating these cache operations in `prepare_source()`.

### 2. Segment instances

Current status:

- **Complete:** injectable segmentation/appearance adapters, proposal cleanup, and per-observation instance, ambiguity, unassigned, and confirmed-environment outputs in `instance_segmentation.py`.
- **Verified:** 23 CPU tests cover alignment/cache geometry and fake-backend segmentation contracts. Real model wrappers/checkpoints and inspection of their mask overlays remain pending before association.
- Reuse `PreparedSource.observations` from `prepare_source()`; segmentation consumes prepared RGB and preserves its `(frame_idx, view_id)` keys and geometry masks.

Requirements:

- Implement `segment_source()` in `instance_segmentation.py` to process each prepared observation and return its instance records, ambiguous-pixel mask, and unassigned-pixel mask. Preserve observation keys even when no instances are found; do not change cameras, image size, or cache packing.
- Define `SegmenterAdapter.segment()` around a promptable instance segmenter. Start with box/point prompts keyed by observation and expressed in prepared-image coordinates; allow automatic proposals through the same output contract. Explicitly convert the supplied RGB range/layout to the backend's expected input and map predicted masks back to the prepared `[H, W]` grid.
- Return one `InstanceObservation` per candidate: a unique observation-local ID, boolean mask, segmentation confidence, and appearance embedding. Validate mask shape and nonempty area; expose configurable quality thresholds. Class names are optional metadata, not instance identity. Discover candidate objects before the composite's final object selection.
- Add `clean_instance_proposals()` to suppress repeated proposals for the same object using mask overlap and confidence. Preserve distinct contained objects; do not suppress a candidate solely because it lies inside another mask. Mark unresolved overlaps/boundaries as ambiguous rather than assigning an arbitrary owner.
- Add `AppearanceEncoderAdapter.encode()` to compute comparable, normalized embeddings from masked RGB crops using one encoder across the source. Exclude surrounding background from the object representation. These features support later association when geometric overlap is limited; they do not establish identity by themselves.
- Keep segmentation masks independent of `geometry_mask`: missing depth must not erase an RGB object candidate. Use geometry validity when matching/unprojecting later. Retain unassigned and ambiguous regions for step 5 to distinguish environment from unknown; do not assign every unclaimed pixel to environment here.
- Retain an explicit environment mask from segmentation or user confirmation. In step 5, assign environment label `0` only to confirmed, valid, unambiguous environment pixels; unassigned pixels remain unknown (`-1`) until resolved.
- Keep IDs local to each observation at this stage. Cross-view association belongs to step 3, temporal identity to step 4, and canonical `(source_id, instance_id)` assignment to step 5. A future tracker may populate optional `track_id`; segmentation alone must not claim consistent identity across views or frames.
- Verify the contract with a fake backend: observation-key preservation, backend mask remapping, two same-class instances, duplicate proposals, distinct contained objects, overlap uncertainty, empty results, and object masks spanning invalid depth. Inspect real-backend mask overlays on prepared RGB before using them for association.

```python
@dataclass
class InstanceObservation:
    frame_idx: int
    view_id: int
    local_id: int
    mask: torch.Tensor        # bool [H, W], aligned with depth
    confidence: float
    appearance: torch.Tensor
    track_id: int | None = None
```

`segment_source()` groups these records by observation; ambiguity and unassigned masks are separate per-image outputs. Final point labels are produced in step 5 and attached in step 6.

Adapter usage (model loading stays with the caller):

```python
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    AppearanceEncoderAdapter, SegmenterAdapter, SegmentPrompts, segment_source,
)

# segment_backend(image, prompts) returns SegmentPrediction with full-input-grid
# InstanceProposal masks and optional explicitly confirmed environment_mask.
# appearance_backend(masked_crop, crop_mask) returns a nonzero floating vector;
# it must honor crop_mask when preprocessing/pooling the object representation.
segmenter = SegmenterAdapter(
    segment_backend, input_layout="HWC", input_range="uint8", input_size=None,
)
encoder = AppearanceEncoderAdapter(appearance_backend, input_layout="CHW", input_range="0_1")
segmented = segment_source(
    prepared, segmenter, encoder, rgb_range=(-1, 1),  # Actual prepared RGB range.
    prompts={(frame_idx, view_id): SegmentPrompts(boxes=boxes_xyxy)},
    environment_masks=confirmed_environment_masks,  # Optional keyed bool [H,W].
    min_confidence=0.5, min_area=16, boundary_width=1,
)
image_result = segmented.observations[frame_idx, view_id]
```

Omit prompts for automatic proposals. Boxes/points use prepared-image integer
pixel centers; the adapter scales them with its optional image resize. Backend
wrappers must undo internal padding/cropping and convert logits to probabilities
before returning masks. Local IDs restart per observation and do not imply
cross-image identity. Candidate masks keep overlaps; downstream rasterization
must exclude `ambiguous_mask`. `environment_mask` is confirmed RGB evidence;
step 5 must intersect geometry validity before assigning label `0`.

Run the CPU contract suite with
`python -m unittest discover -s tests -p 'test_*seg*.py' -v`.

### 3. Match views geometrically
In `instance_association.py`, implement `visible_correspondences()` using existing `unproject_points()` and `project_points()` from `forward_warp_utils_pytorch.py`. Use valid, in-bounds points whose projected depth agrees with destination depth. Project in both directions; compare masks only on mutually observed surfaces, so unseen portions do not count as mismatches.

```python
# points: [H, W, 3]; source_valid, dst_depth: [H, W]
# dst_w2c: [4, 4]; dst_K: [3, 3]; depth tolerances are configurable.
def visible_correspondences(points, source_valid, dst_depth,
                            dst_w2c, dst_K, abs_tol, rel_tol):
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
    return x, y, visible
```

Add `score_instance_pairs()` combining symmetric visible overlap, appearance similarity, and segmentation confidence. Require minimum geometric support when using overlap. `associate_instances()` performs one-to-one assignment with unmatched candidates allowed, a score threshold, and a runner-up margin. No-overlap appearance matches remain tentative; ambiguous matches become unknown pending correction. Do not merge instances solely because their class names agree.

### 4. Track across time
Implement `TrackerAdapter.propagate()` per camera: propagate masks, periodically segment again to discover new objects, and reconcile detections with tracks. Keep IDs through occlusion; use appearance and motion-aware tracking for reappearance. Associate tracks across synchronized views at corresponding times. Use static reprojection across time only for static content; moving objects require tracking. Reject registry merges that would give two distinct objects in the same image one ID.

### 5. Assign canonical labels and persist
Implement `ObjectRegistry` with `allocate()`, `bind_observation()`, and `merge_ids()`; allocate positive IDs within each source and preserve observation-to-ID mappings. Track merges/corrections update all affected maps. Rasterize accepted disjoint masks; reserve `0` for the environment outside candidate object regions and `-1` for unresolved/invalid regions. Persist integer maps and validity in NPZ, and names, confidence, observations, and ID mappings in JSON.

```python
# Rasterize one image; `observations` contains only its instance masks.
labels = torch.full(valid.shape, -1, dtype=torch.int64)
labels[valid & background & ~uncertain] = 0
for obs in observations:
    key = (obs.frame_idx, obs.view_id, obs.local_id)
    instance_id = observation_to_id.get(key)
    if instance_id is not None:
        labels[valid & obs.mask & ~uncertain] = instance_id
# `background` excludes candidate masks; `uncertain` includes mask conflicts.
```

### 6. Attach to cache points and check consistency
Add label-loading arguments at cache construction in `gen3c_multiview.demo()` and `Gen3cPersistentModel.seed_model_from_values()`. Normalize labels with the same dimension mapping as RGB into `[B, F, N, V, 1, H, W]`; the static multiview path stores source views along `N`. In `Cache3D_Base.__init__()`, store them as `int64`, separate from `input_mask`. Pixel `(y, x)` labels `input_points[..., y, x, :]` directly, so unprojection mathematics stays unchanged. Keep separate view samples carrying the same canonical ID.

Add `check_label_consistency()` to reproject labels and flag disagreements on mutually visible surfaces, excluding unknowns. Verify with a two-view scene containing two similar objects, plus a moving object that disappears and reappears. Also check environment coverage, label/RGB alignment, and save/load ID preservation. First deliverable: saved label maps, registry, and disagreement overlays for one source; then attach them to a child cache.
