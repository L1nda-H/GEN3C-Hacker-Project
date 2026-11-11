# Cache3D_Composite implementation plan

## Qualities
- Use [SAM 2.1](https://github.com/facebookresearch/sam2) for all instance segmentation and video mask propagation.
- Wrap existing single-view, multiview, and dynamic caches; preserve each source's views and frames.
- Give each object a persistent `(source_id, instance_id)`; reserve `0` for environment and `-1` for unresolved labels. Require one designated environment source.
- Keep instance labels separate from geometry-validity masks; preserve source caches when changing selections or placement.
- Expose the existing `render_cache()` camera interface, with explicit object selections and a replaceable fusion policy.

## Implementation steps
1. **Segment and label single-camera inputs for the MVP.** Use `SAM21Segmenter` in `instance_segmentation.py`; add SAM video propagation for single-camera clips. Produce reusable integer label maps and a per-source object registry before constructing child caches. Temporarily skip geometric cross-view association (step 3 below); retain its plan for later multiview support.

2. **Attach labels to child caches.** In `cache_3d.py`, extend `Cache3D_Base.__init__()` to accept integer labels aligned with `input_points`; retain `_compute_input_points()` and `unproject_points()` geometry unchanged. Extend `render_cache()` and subclass overrides to accept selected IDs. In `Cache3D_Buffer.update_cache()`, maintain labels alongside inserted/replaced points. Existing `Cache3D_Labeled` motion flags do not replace instance IDs.

3. **Build the composite.** Add `cache3d_composite.py` containing `Cache3D_Composite`: `add_source()`, `set_selection()`, `set_transform()`, `set_time_mapping()`, and `render_cache()`. Store child caches by source ID, object metadata, source/object placement transforms, and dynamic time mappings. Place selected geometry in one destination coordinate system, including explicit scale calibration; repeat static sources across requested times.

## Step 1: segmentation, matching, and labeling

Status: input alignment, SAM 2.1 image/video wrappers, canonical labeling, and editable persistence are implemented with local contract tests. Real-checkpoint accuracy validation, GUI integration, geometric association, and cache label attachment remain pending. The geometric snippet below remains a partial integration sketch.

**MVP order:** temporarily skip step 3. Single images follow steps 1 → 2 → 5 → 6; single-camera videos follow steps 1 → 2 (object seeding) → 4 → 5 → 6. Sources retain independent object-ID namespaces when combined in the composite.

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
- Share configuration initialization through `cosmos_predict1.utils.model_config.model_config_session()`. Application startup exposes SAM's package configs alongside GEN3C's ConfigStore registrations; both loaders compose independently without nested initialization. `gen3c_multiview` owns this session at its CLI entry point. Reuse a prior `import sam2` session when the requested config is accessible; reject incompatible caller sessions without clearing or replacing them. Standalone loaders own a temporary session only when none exists.
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

For combined segmentation and diffusion, application startup owns one shared Hydra session (`pipeline_kwargs` contains the caller's generation settings):

```python
from cosmos_predict1.utils.model_config import model_config_session
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline

with model_config_session(sam_config="configs/sam2.1/sam2.1_hiera_l.yaml"):
    segmenter = SAM21Segmenter.from_checkpoint("checkpoints/sam2.1_hiera_large.pt")
    pipeline = Gen3cPipeline(**pipeline_kwargs)
    # Run segmentation, conditioning, and generation here.
```

GEN3C registers its runtime root under `_cosmos_predict1_runtime_config`, preserving a caller's root named `config`. Configuration registration/composition must be serialized; the helper never resets caller-owned Hydra state. A caller-owned session must already expose the requested SAM config (normally through `pkg://sam2`). Review verified the 25 existing tests plus five focused lifecycle smoke checks: standalone composition/cleanup, shared startup/repeated composition/root preservation, prior SAM import, incompatible caller preservation, and construction-failure cleanup. The smoke checks used the pinned SAM package with a synthetic checkpoint and GEN3C's override function with a small configuration fixture; pretrained mask quality and full diffusion inference remain unvalidated.

### 3. Match views geometrically

**Temporarily skipped for the single-camera MVP.** Preserve the requirements, milestone, and partial code sketch below for later multiview support; this step does not block steps 4–6.

Current status:

- **Pending:** `instance_association.py`, geometric correspondences, pair scoring, and assignment are not implemented.
- Consume `SegmentedSource`, reusing its `prepared_source` cameras/depth/validity and SAM masks, optional quality scores, and scoped video track IDs. Keep segmentation results and child caches unchanged.

Next implementation milestone: static two-view association.

1. Define `AssociationResult` and pair diagnostics keyed by observation/local IDs and the segmentation revision. Begin with one static source and two calibrated views.
2. Implement `visible_correspondences()` using the existing geometry helpers, applying validity, ambiguity, depth agreement, and nearest-depth collision handling in both directions. Keep the partial sketch below as a reference, not the complete implementation.
3. Implement `score_instance_pairs()` and `associate_instances()` with support gates, deterministic one-to-one assignment, and explicit unmatched/ambiguous results. Retain the optional-quality and invalidation requirements below.
4. Validate synthetic known matches, similar objects, partial occlusion, and no-overlap cases; include unequal image sizes and reordered/corrected segmentation outputs. Then inspect overlays and link diagnostics on a real two-view SAM result.

Completion criterion: correct supported links and explicit unresolved cases, ready for the canonical registry in step 5. Video mask propagation is not a prerequisite for this static milestone; synchronized video association follows once step 4 supplies tracks.

Requirements:

- Implement `match_source_views()` in `instance_association.py` to enumerate observation pairs within one source. For videos, run SAM propagation in step 4 before final cross-view association; match synchronized views at the same frame. Allow different capture times only for explicitly static content. Single-view sources retain their candidates without cross-view links; unrelated scenes remain independent.
- Implement `visible_correspondences()` using existing `unproject_points()` / `project_points()` in `forward_warp_utils_pytorch.py`. Unproject valid camera-Z samples into the source's shared world frame and project into the destination camera. Exclude ambiguous pixels in both observations; require valid destination depth. Reject nonfinite coordinates, points behind the camera, and out-of-grid samples before gathering; use nearest sampling consistent with integer pixel centers.
- Test projected camera-Z against destination depth with configurable absolute and relative tolerance: `abs(z_projected - z_observed) <= abs_tol + rel_tol * z_observed`. Treat points behind an observed foreground surface as occluded. Invalid depth, occlusion, and missing coverage supply no matching evidence; distinguish these from a supported mismatch.
- Resolve projected point collisions with nearest-depth selection and count unique destination pixels, preventing multiple source samples from inflating support. Compute correspondences in both directions and report depth residuals, support counts, and visible-support fractions.
- Implement `score_instance_pairs()` using bidirectional mask agreement on visible, valid, unambiguous surfaces, depth residuals, and available SAM mask quality. Require minimum support/count fractions and overlap quality; exclude unassigned/ambiguous destination pixels. Treat predicted IoU/stability as auxiliary quality, not identity probabilities; handle missing quality without rejecting an otherwise supported match or assuming perfect confidence. Do not require raw predicted IoU to lie in `[0, 1]`. Do not use a separate appearance encoder or semantic class names for association.
- Implement `associate_instances()` with deterministic maximum-score one-to-one assignment per observation pair, allowing unmatched candidates. Gate edges by support, score, and runner-up margins on both sides. With absent/insufficient overlap, retain separate candidates or accept an explicit GUI identity link; do not infer identity from matching native IDs in independent SAM sessions.
- Return an `AssociationResult` containing accepted links between `(frame_idx, view_id, local_id)` keys, unmatched/ambiguous candidates, and score/support diagnostics. Bind results to the exact segmentation snapshot/revision and invalidate them after prompt corrections or rediscovery: automatic enumeration IDs may change, and a reused prompt ID may now have a different mask. Keep confirmed environment masks outside object assignment. Flag multi-view link conflicts that would identify two distinct objects in one observation as the same object; canonical IDs and registry merges remain step 5 work.
- Verify with synthetic calibrated scenes: known same-object views, two similar objects, partial occlusion, no overlap, invalid depth, ambiguous masks, unequal resolutions, point collisions, noisy depth, assignment ties, empty/single-view inputs, and static/dynamic pair eligibility. Include missing/out-of-range predicted quality and association invalidation after reordered automatic proposals or same-ID mask corrections. Inspect correspondence overlays and accepted/rejected links on real segmentation outputs after backend validation.

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

Purpose:

Keep one selected object's identity consistent across frames of a single-camera video. A GUI selection can then include that object's labeled points throughout the clip. Still-image sources use step 2 directly and bypass propagation.

Current status: `SAM21VideoSegmenter` is implemented in `instance_tracking.py`, reusing the segmentation result containers and image/video uncertainty policy. Seven predictor-contract tests cover numeric frame export, sparse keys, object-row ordering, occlusion, late seeds, prompts, correction invalidation/replay, and checkpoint construction; all 32 local tests pass. Real-checkpoint clip inspection on CUDA remains pending. Step 3 remains deferred.

Usage after step-2 segmentation (`segmented.prepared_source` is one camera with a common image size):

```python
from cosmos_predict1.diffusion.inference.instance_tracking import SAM21VideoSegmenter

tracker = SAM21VideoSegmenter.from_checkpoint("checkpoints/sam2.1_hiera_large.pt")
tracker.initialize(segmented.prepared_source, source_id="clip-a", rgb_range=(-1, 1))
seed_key = (frame_idx, view_id)
tracker.seed_objects(seed_key, segmented.observations[seed_key].instances)  # Or selected instances only.
with torch.autocast("cuda", dtype=torch.bfloat16):
    tracked = tracker.propagate(environment_masks=confirmed_environment_masks)
    # corrected_instance has the same local_id; SegmentPrompts is also accepted.
    tracker.correct_object(correction_key, corrected_instance)
    tracked = tracker.propagate(environment_masks=confirmed_environment_masks)
```

`frame_keys[offset]` maps SAM offsets to original keys. Caller IDs pass unchanged to SAM; names and scoped track IDs persist through empty masks. Each initialization creates a distinct session namespace. Calls on one wrapper are serialized. Edits set `tracker.result` to `None`; the next propagation resets/replays public SAM seeds and recomputes the full clip in both directions to discard stale temporal memory. Old returned snapshots remain unchanged. Downstream label/render consumers must rebuild from the new snapshot once steps 5–6 exist. Confirmed environment masks remain explicit and per-frame; they are not propagated or inferred. Temporary numeric JPEG export uses synchronous loading and does not alter prepared geometry/RGB tensors.

Requirements:

- Accept one `PreparedSource` camera stream with a common prepared image size. Keep RGB, depth, poses, and validity unchanged; export SAM-supported, numerically ordered frames and preserve the offset-to-`(frame_idx, view_id)` mapping, including sparse original frame keys.
- Build `SAM2VideoPredictor` with the pinned SAM 2.1 checkpoint/configuration inside the shared `model_config_session()`. One source/camera owns one predictor state; serialize changes to that state.
- Seed selected step-2 masks with `add_new_mask()`, or grouped clicks/boxes with `add_new_points_or_box()`. Keep a stable object ID and name for each seed. Run `propagate_in_video()` instead of independently segmenting and renumbering every frame.
- Map output rows using the returned `object_ids`; convert logits to prepared-grid boolean masks with `> 0`. Populate `SegmentedSource` / `InstanceObservation` with stable per-session `local_id`, scoped `track_id=(source_id, view_id, session_id, obj_id)`, and optional quality metadata. Propagated quality remains `None` unless explicitly available.
- Keep absent/fully occluded objects in the session's track metadata, with empty visible masks and no assigned foreground points. Recompute overlap/boundary ambiguity and unassigned regions per frame; retain the existing confirmed-environment policy. Missing depth must not erase RGB tracks.
- Support later object seeds and corrections through SAM's APIs, reusing IDs for the same object. Repropagate affected frames, including reverse propagation when earlier requested frames need coverage, and refresh derived label maps/render masks while retaining object selection by ID. Unresolved identity switches require GUI correction.
- Step 5 can map a scoped single-camera track directly to a canonical ID; no step-3 result is required for this MVP. Separate scenes/sessions keep distinct namespaces.

Implementation steps:

1. Add `SAM21VideoSegmenter` in `cosmos_predict1/diffusion/inference/instance_tracking.py`. Implement `from_checkpoint()` and `initialize()` using the shared Hydra helper, `SAM21_MODELS`, ordered frame export, and an explicit frame-key lookup.
2. Implement `seed_objects()` and `correct_object()` as thin calls to SAM's mask/click/box APIs. Preserve seed names and the mapping between caller object IDs and SAM session IDs.
3. Implement `propagate()` to collect outputs by original observation key and object ID. Assemble the existing segmentation result types and per-frame uncertainty/environment masks; retain seeded identities through empty frames.
4. Verify frame ordering, nonconsecutive keys, returned object-row ordering, empty masks, two objects, late seeds, and correction invalidation with predictor-contract tests. Then inspect a real clip with motion and temporary occlusion.

Completion criterion: seed two objects once, produce aligned masks with stable IDs across one camera's clip, and correct a mask while preserving its identity. The resulting records are ready for step-5 labeling and step-6 point attachment.

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

Step 5 should turn a `SegmentedSource` into persistent object identities and integer label maps. It can work directly with single images and single-camera tracks while step 3 stays deferred.

1. **Add `ObjectRegistry` in `instance_labeling.py`.**
   - `allocate(name)` creates a positive canonical ID within one source.
   - `bind_observation((frame_idx, view_id, local_id), instance_id)` records identity.
   - Map each scoped video `track_id` to one canonical ID across frames, including empty masks.
   - Treat still-image candidates independently; matching local IDs across images/sessions do not establish identity.
   - `merge_ids()` redirects bindings to a surviving ID. Never reuse retired IDs.

2. **Rasterize labels using existing masks.**
   - Start with an `int64 [H,W]` map filled with `-1`.
   - Assign `0` only to confirmed, geometry-valid, unambiguous environment.
   - Assign positive IDs only where the bound object mask is valid and unambiguous.
   - Leave overlaps, uncertain boundaries, unassigned regions, and invalid geometry at `-1`.
   - Preserve geometry validity separately and keep the original RGB masks unchanged.

3. **Support corrections without losing identity.**
   - Accept the previous registry when labeling a corrected tracking result.
   - Preserve canonical IDs for unchanged scoped tracks; allocate IDs for new tracks.
   - Rebuild label maps after corrections or merges.
   - Require explicit identity preservation for still-image rediscovery, whose automatic IDs may change.
   - **Rediscovery** means rerunning automatic image segmentation: local ID `3` may now identify a different object. A **new video session** means initializing the tracker again, which creates a fresh track-ID namespace even for the same clip.
   - A **segmentation revision** uniquely identifies one result snapshot; changed masks produce a new revision. Scope observation bindings to that revision so reused `(frame_idx, view_id, local_id)` keys cannot silently inherit an old identity. Across revisions, preserve identities through unchanged scoped tracks or explicit confirmation.

4. **Persist and reload the result.**
   - **NPZ:** label maps and geometry validity, keyed through an explicit observation manifest.
   - **JSON:** schema version, source ID, object names, observation/track bindings, merge aliases, optional quality scores, segmentation revision, SAM revision/configuration, and checkpoint SHA-256.
   - Validate shapes, integer values, referenced IDs, and prepared-source alignment when loading. Sparse frame keys must survive unchanged.
   - **Source-alignment validation** checks that saved labels belong to the exact prepared inputs being loaded. Observation keys locate frames/views; a prepared-source fingerprint checks input identity, crop/resize settings, depth, and camera metadata. Matching shapes and frame keys alone cannot distinguish two clips or detect changed pixel alignment. Store and verify this fingerprint before attaching labels to points.
   - **Reload** means reopening saved results, for example after an application restart. For cache loading/rendering, label maps, validity, and registry metadata are sufficient when the matching prepared source is available. To resume **editing** (correcting object masks or prompts), also persist the original RGB-grid object masks, ambiguity masks, and confirmed environment masks, or explicitly require resegmentation. Confirmed environment masks record the user's acceptance of environment pixels before geometry/ambiguity filtering; final labels cannot recover confirmations or object-mask pixels excluded by invalid depth or ambiguity. Saving masks preserves editable segmentation data; resuming SAM video inference additionally requires reinitializing and seeding the predictor.

Current status: `instance_labeling.py` implements `ObjectRegistry`, `label_source()`, and NPZ/JSON save/load. Image/video results carry a fresh segmentation revision; checkpoint loaders record SAM revision/configuration and compute the checkpoint hash once. Injected predictors may explicitly leave provenance unknown (`None`). Original environment confirmations are retained separately from the filtered environment mask. Source fingerprints include prepared RGB/depth/cameras/validity and preparation settings, excluding existing labels.

```python
from cosmos_predict1.diffusion.inference.instance_labeling import (
    label_source, save_labeled_source, load_labeled_source,
)

labeled = label_source(segmented, source_id="clip-a")  # Must match video track source_id.
save_labeled_source("outputs/clip-a", labeled)  # Creates .npz and .json siblings.
loaded = load_labeled_source("outputs/clip-a", segmented.prepared_source)
updated = label_source(corrected_segmented, source_id="clip-a", registry=loaded.registry)
```

`label_source()` copies the supplied registry and creates new label maps; prior results remain unchanged. For image corrections or a restarted video session, pass explicit `identity_links={(frame_idx, view_id, local_id): canonical_id}` to retain a confirmed identity. Bindings otherwise carry across revisions only through the same scoped video track. Changed masks or confirmation/uncertainty data require a new `SegmentedSource` revision; reusing a revision with changed content is rejected. After `registry.merge_ids(retired_id, surviving_id)`, rerun `label_source()` on each affected snapshot using the merged registry. Saving rejects stale maps, and loading verifies the source fingerprint, NPZ/JSON pairing, registry references, shapes/dtypes, and rasterization semantics. All saves include masks for editing; SAM runtime state and source RGB/depth/cameras are supplied separately.

Tests should cover deterministic allocation, independent source/session namespaces, disappearance/reappearance, corrections and merges, overlap/environment rules, missing depth, sparse keys, exact save/load preservation, and rejection of mismatched metadata.

Verification: 52 CPU tests pass, including image/video API contracts, tracking-to-label persistence, large integer IDs, and environment confirmations at pixels excluded from final labels. Run `python -m unittest discover -s tests -v`. Real-checkpoint mask accuracy remains a CUDA validation task.

The deliverable is a registry plus reloadable label maps ready for step 6.

### 6. Attach to cache points and check consistency

- **Wire inputs:** add label loading to `gen3c_multiview.demo()`, `Gen3cPersistentModel.seed_model_from_values()`, and `PreparedSource.cache_kwargs()`.
- **Store labels:** normalize like RGB to CPU `int64 [B,F,N,V,1,H,W]`, separate from `input_mask`. Reject noninteger labels, values below `-1`, shape mismatches, and unregistered positive IDs. Never cast IDs through RGB/`weight_dtype` or interpolate them.
- **Preserve alignment:** map observation keys to cache offsets; retain separate samples sharing an ID. Verify source fingerprints, use integer-safe nearest sampling for crop/resize, and label `input_points[...,y,x,:]` directly without changing unprojection.
- **Preserve validity:** combine pre-sanitization geometry validity with cache reliability filtering. Labels cannot enable invalid points. Pass MoGE's validity mask and align labels to its output grid.
- **Render selections:** update every `render_cache()` override. `None` keeps existing behavior; empty selections give zero coverage; `0` selects environment; reject `-1`, unknown IDs, and explicit selection on unlabeled caches.
- **Apply selection before warping/ranking:** intersect selected IDs with validity, preserve stored geometry and RGB/depth/coverage contracts, and prevent the selector's near-full-view shortcut from suppressing selected objects.
- **Update buffers:** insert/replace labels and validity with geometry, including single-slot buffers. Require associated labels for labeled-cache updates; disable generated-frame updates until association exists. Preserve unlabeled behavior.
- **Refresh corrections:** replace maps without recomputing geometry; validate source/revision, reject stale results, and resolve selections through merge aliases.
- **Limit MVP scope:** support images and single-camera clips; static observations use `N`, time uses `F`. Validate render windows and repeat static sources across target times. Defer `V > 1` rendering and two-view reprojection checks until step 3.
- **Verify:** add basic `check_label_consistency()` for shape/dtype, registry, source/revision, and render validity. Test layouts, sparse keys/windows, invalid depth/filtering, large IDs, corrections/merges, selected-only RGB/depth coverage, buffer updates, save/load, and unlabeled compatibility.
- **Deliver:** labeled points, maps, registry, and overlays; confirm environment coverage and stable video IDs through disappearance/reappearance.
