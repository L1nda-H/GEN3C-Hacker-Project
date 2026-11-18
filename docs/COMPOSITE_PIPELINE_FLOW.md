# Composite pipeline: images and bounding boxes to video

This traces the current composite implementation on `sandbox`, compared with
`main` at `ff53f0a` (`git diff main...HEAD`). New functions and changed functions
on the executed path are expanded. **BLACK BOX** means an unchanged repository
function or an external model API; its input/output contract is shown without
expanding its internals. A changed file does not make every function in it new.

Preparation and generation are separate operations. The generation CLI consumes
saved sources and a scene specification; it does not accept raw bounding boxes
or run segmentation. Composite GUI integration is proposed, not implemented.
The box-driven preparation entry point is
[`scripts/evaluate_real_cache_mvp.py`](../scripts/evaluate_real_cache_mvp.py).
Raw photographs alone are insufficient: each source also needs aligned depth,
validity, intrinsics, and an OpenCV world-to-camera matrix. The example loader
reads these from existing dataset artifacts; it does not estimate depth.

## 1. End-to-end overview

```mermaid
flowchart TD
    images["Selected environment and object images + aligned depth and cameras"]
    boxes["Object boxes, optional removal boxes / clicks, confirmed environment"]
    prep["prepare_source -> PreparedSource"]
    segmentation["segment_source + SAM21Segmenter.segment -> SegmentedSource"]
    labeling["label_source + ObjectRegistry -> LabeledSource with canonical IDs"]
    artifacts["save_prepared_source + save_labeled_source -> persisted source artifacts"]
    scene["SceneSource + SceneSpec: selections, transforms, trajectory, resolution"]
    build["build_composite_scene -> labeled Cache3D_Base children + Cache3D_Composite"]
    generate["Gen3cCompositePipeline.generate: validate and iterate camera chunks"]
    render["render_conditioning: select, transform, project and jointly resolve visibility"]
    buffers["Visible per-source RGB + coverage buffers"]
    seed["Initial RGB seed or previous chunk's final generated frame"]
    model[["BLACK BOX: Gen3cPipeline.generate"]]
    chunks["Check output; remove repeated boundary frames; concatenate chunks"]
    save[["BLACK BOX: save_video"]]
    output["Final MP4 + review artifacts"]
    images --> prep --> segmentation --> labeling --> artifacts --> scene --> build --> generate --> render --> buffers
    boxes --> segmentation
    buffers --> model
    build -->|"Source RGB, masks, transforms and depth"| imageSeed["build_image_space_seed: image-space composite"]
    imageSeed -->|"First chunk"| seed
    seed --> model --> chunks --> save --> output
    chunks -->|"Next chunk: retain last frame"| seed
    chunks -->|"Next camera slice"| render
    render -.->|"save_render"| output
    generate -.->|"scene, seed, manifest"| output
    classDef blackBox fill:#222222,color:#ffffff,stroke:#000000
    class model,save blackBox
```

## 2. Images and boxes become labeled, reproducible sources

Repeat this stage independently for each source. The current composite builder
requires exactly one static observation per source.

```mermaid
flowchart TD
    load["load_source: read first RGB frame, depth, mask and intrinsics; use local identity w2c"]
    observation["SourceObservation: RGB, camera-Z depth, K, w2c, geometry_mask"]
    prepare["prepare_source: validate; capture valid depth; crop/resize all grids; adjust K"]
    prepared["PreparedSource: observation mapping + preparation metadata"]
    prompts["SegmentPrompts: positive local_id, XYXY box and/or labeled clicks"]
    wrapper["SAM21Segmenter.from_checkpoint / segment: load model, normalize RGB and validate prompts"]
    sam[["BLACK BOX: SAM set_image + predict"]]
    prediction["SegmentPrediction: nonempty InstanceProposal masks with quality scores"]
    segmented["segment_source / _segmented_observation: mark overlap, boundaries, unassigned and confirmed environment"]
    registry["label_source: fingerprint source and segmentation; allocate/bind canonical IDs"]
    raster["_rasterize: intersect masks with geometry validity; exclude uncertain pixels"]
    labeled["LabeledSource: registry, int64 label maps, segmentation snapshot and source fingerprint"]
    persist["save_prepared_source + save_labeled_source + optional label_overlays"]
    load --> observation --> prepare --> prepared --> wrapper --> sam --> prediction --> segmented --> registry --> raster --> labeled --> persist
    prompts --> wrapper
    prepared --> segmented
    confirmed["Explicit confirmed environment mask"] --> segmented
    classDef blackBox fill:#222222,color:#ffffff,stroke:#000000
    class sam blackBox
```

`SourceObservation` and the segmentation/result classes are data containers.
`prepare_source()` optionally converts ray distance to camera-Z using unchanged
`unproject_points()`. RGB resize is bilinear; depth, validity, and categorical
labels use nearest sampling. Boxes/clicks must be expressed on the **prepared**
image grid, after any crop/resize. Each prompt group describes one object.

`SAM21Segmenter.segment()` uses the external predictor as a black box. When
multimask output is requested, it retains the highest-quality nonempty mask for
each group. Automatic proposal discovery is an alternative when prompts are
omitted; the box-driven path uses explicit groups. A SAM local ID or proposal
index is not a canonical object ID.

`ObjectRegistry.allocate()`, `bind_observation()`, and `resolve()` establish
source-scoped canonical identities. `label_source()` reuses explicit identity
links or existing bindings when supplied; otherwise it allocates IDs.
`_rasterize()` writes `-1` for unresolved, invalid, or ambiguous pixels; `0` for
confirmed valid environment; and positive canonical IDs for valid objects.
Mask overlaps remain unresolved rather than arbitrarily assigning an object.

In the MVP preparation driver, the caller confirms the environment with an
all-true mask. Environment removal boxes are segmented into positive IDs;
selecting only environment ID `0` later excludes those objects. Excluded regions
leave holes; preparation does not reconstruct the background behind them.

Persisted outputs are the exact prepared RGB/depth/K/w2c/validity NPZ and the
label artifact pair (`*_labels.json` and `*_labels.npz`). Optional overlays let
the author inspect canonical labels before proceeding.

## 3. Author the scene, then build static source caches

```mermaid
flowchart TD
    author["Choose canonical selected_ids, environment, placements, target cameras and output H/W"]
    placement["Optional MVP object_placement: bottom pivot + destination depth + requested pixel height -> scale/translation"]
    spec["SceneSource entries -> SceneSpec.__post_init__ / validate / save"]
    cli["gen3c_composite.demo: SceneSpec.load or Python SceneSpec; validate CLI and prompts"]
    build["build_composite_scene: validate SceneSpec; create Cache3D_Composite"]
    sources["For each source: load_prepared_source + load_labeled_source; verify fingerprint and ownership"]
    kwargs["PreparedSource.cache_kwargs: pack RGB, depth, validity, labels and registry"]
    points[["BLACK BOX: unproject_points - depth + K + w2c + validity -> source-world points"]]
    child["Cache3D_Base.__init__: store CPU RGB/points, validate integer labels, bind registry and revisions"]
    attach["Cache3D_Composite.add_source: validate static child; attach source camera; designate environment"]
    select["set_selection: resolve canonical IDs; exclude unselected and unknown pixels"]
    transform["set_transform: source similarity and optional object similarities; invalidate sample tables"]
    ready["Static Cache3D_Composite + source provenance"]
    author --> spec
    author --> placement --> spec
    spec --> cli --> build --> sources --> kwargs --> points --> child --> attach --> select --> transform --> ready
    transform -->|"More sources"| sources
    classDef blackBox fill:#222222,color:#ffffff,stroke:#000000
    class points blackBox
```

[`SceneSpec`](../cosmos_predict1/diffusion/inference/composite_scene.py)
stores artifact paths/fingerprints, selections, transforms, target cameras and
output size. `load()` resolves paths relative to the JSON; `save()` records
resolved paths. `validate()` checks unique source IDs, required environment ID
`0`, transforms, camera shapes, and OpenCV camera validity. The environment's
source transform stays identity. Loading never reruns depth estimation,
segmentation, or label resizing.

`build_composite_scene()` explicitly supplies unprojected points to the changed
`Cache3D_Base.__init__()`, avoiding the legacy depth clamp. The unchanged
`_compute_input_points()` is therefore bypassed on this path. The cache attaches
RGB, valid geometry, canonical labels, registry, fingerprint and segmentation
revision. Its original `render_cache()` is also bypassed: composition uses
`Cache3D_Composite` rendering.

Placement applies **destination point = T_source @ T_object @ source point**.
Similarities allow orientation-preserving uniform scale, rotation and
translation. Placement changes the gathered samples, not the child geometry.
The MVP helper computes a placement from destination base pixel, measured
environment depth and desired projected object height; direct callers may
provide matrices instead. `adjust_object_depth()` optionally scales object
position and size about the first camera to change depth while preserving its
initial projection.

## 4. Inside Cache3D_Composite rendering

```mermaid
flowchart TD
    entry["render_conditioning(target camera chunk, include_depth)"]
    cameras["_target_cameras + conditioning_source_ids: validate cameras; assign sorted selected-source slots"]
    frames["_render_frames -> _joint_render_table -> _render_tables"]
    gather["_iter_render_inputs: check attachments; gather valid selected labels in source/object/pixel order"]
    support["_source_pixel_tangents: infer source-pixel cells using compatible neighbors or camera fallback"]
    place["Apply T_source @ T_object to points and linear transform to tangents; produce CompositeSamples"]
    tables["_SampleTable per source; join points, colors, tangents and ownership metadata"]
    resolve["_resolve_visibility -> adaptive_surface_winners"]
    cells["_adaptive_surface_candidates / _camera_pixel_cells: project cells, clip and stream pixel candidates"]
    depths["_pixel_cell_rays / _pixel_cell_intersection: ray-plane camera-Z"]
    winners["_surface_winners: minimum depth, then smallest sample rank within tolerance"]
    route["_render_frames: route each joint winner into its owning source slot"]
    result["RGB, coverage and optional camera-Z depth tensors"]
    entry --> cameras --> frames --> gather --> support --> place --> tables --> resolve --> cells --> depths --> winners --> route --> result
    reused["Cached sample tables if selection, placement, labels and RGB range are unchanged"] -.-> tables
```

The functions under
[`forward_warp_utils_pytorch.py`](../cosmos_predict1/diffusion/inference/forward_warp_utils_pytorch.py)
from `_point_pixel_coordinates()` through `adaptive_surface_winners()` are new
on this branch. They are expanded here; `unproject_points()` and
`project_points()` remain black boxes. The adaptive candidate path calls
unchanged `project_points()` to project cell centers/corners.

Pixel-cell tangents avoid bridging invalid pixels, different labels or large
depth discontinuities. Missing axes use a source-camera frontoparallel estimate.
Degenerate cells and near-plane-crossing cells fall back to point projection.
Candidates are streamed in bounded batches. Visibility compares **all selected
sources together**, so imported objects and the environment can occlude each
other. Ties resolve by deterministic source/object/original-pixel order within
the depth tolerance.

For `T` chunk frames and `N` selected source caches, outputs are:

| Output | Shape | Values / meaning |
| --- | --- | --- |
| `rgb` | `[1,T,N,3,H,W]` | Float32 `[-1,1]`; uncovered or hidden pixels are `-1` |
| `coverage` | `[1,T,N,1,H,W]` | Float32 binary visibility; hidden pixels are `0` |
| Optional `depth` | `[1,T,N,H,W]` | Winning pixel-cell camera-Z; holes are `0` |

`pixel_cell_winner_depth()` computes saved review depth at each winner's target
pixel. Source slots follow sorted source IDs. Several selected objects from one
source share a slot. Fully hidden or geometrically empty selected sources retain
their slot; sources without a selection are omitted. The wrapper rejects more
slots than the checkpoint's `frame_buffer_max`.

Related preview methods share the renderer: `render_cache()` calls `_render()`
for combined RGB or depth; `render_diagnostics()` additionally returns categorical
winner ownership and nearest depth. Generation uses `render_conditioning()`.
`combine_visible_buffers()` recombines the already resolved source buffers for
a joint preview or seed, without another projection/depth test.

`_source()` checks current attachments/label state; `refresh_source()` resolves
revised canonical IDs and invalidates tables. If merged objects have conflicting
placements, rendering requires an explicit `set_transform()` correction.
Label replacement and registry merging are editing paths, not normal generation
stages. Generated frames never update source caches.

## 5. Image-space seed, diffusion chunks, and final outputs

```mermaid
flowchart TD
    setup["demo: validate scene, prompt, frame count and seed options"]
    init[["BLACK BOX: Gen3cPipeline initialization / model loading"]]
    savegen["_save_generations: build static composite once; iterate prompts"]
    generate["Gen3cCompositePipeline.generate: validate model size, slots and chunk contract; save initial manifest"]
    imageSpace["build_image_space_seed: sample environment RGB; place textured object planes; resolve depth/alpha"]
    render["Render current camera chunk; save_render RGB/coverage/depth"]
    first{"First chunk?"}
    initialSeed["Use image-space SeedResult RGB; warp coverage is diagnostic only"]
    nextSeed["Previous generated final frame -> normalize to float32 seed"]
    model[["BLACK BOX: Gen3cPipeline.generate(prompt, seed, warp RGB, warp masks)"]]
    allowed{"Output returned?"}
    blocked["Record blocked_chunk; return None; CLI skips video save"]
    check["Require uint8 chunk_size x H x W x 3; retain last frame"]
    append["Append full first chunk; append segment[1:] for continuations"]
    more{"More trajectory chunks?"}
    join["Concatenate segments once -> video, final_prompt"]
    encode[["BLACK BOX: save_video"]]
    done["Write prompt.txt; update generation.json to saved with final video path"]
    setup --> init --> savegen --> generate --> imageSpace --> render
    render --> first
    first -->|"Yes"| initialSeed --> model
    first -->|"No"| nextSeed --> model
    render -->|"RGB + coverage"| model
    model --> allowed
    allowed -->|"No"| blocked
    allowed -->|"Yes"| check --> append --> more
    more -->|"Yes: advance chunk_size minus 1"| render
    more -->|"No"| join --> encode --> done
    classDef blackBox fill:#222222,color:#ffffff,stroke:#000000
    class init,model,encode blackBox
```

The unchanged `Gen3cPipeline.generate()` is a single black box here. Its contract
accepts the prompt, optional negative prompt, RGB seed `[1,3,1,H,W]`, and the
per-source warp RGB/coverage tensors. It returns `(uint8_video, final_prompt)`
or `None` on guardrail rejection. Its pretrained encoders, conditioning,
denoising, decoding and guardrails are not expanded. Review depth is not passed
to diffusion. The seed has no separate spatial coverage mask.

The first seed is produced directly by `build_image_space_seed()` using the
first scene camera: source environment RGB is sampled directly and
segmented object RGB is warped on transformed median-depth planes, with
per-pixel depth ordering and premultiplied alpha. `_maps()`, `_sample()`, and
`_result()` implement image mapping and the tensor contract.

There are no rendered/provided seed branches, seed-selection flags, or minimum
rendered-coverage threshold. The first camera must match the environment source
pose and have a covered field of view; moved environment objects are unsupported, while deselected objects are filled
with CPU LaMa.
Uncertain occlusions are rejected by default.
Unsupported scenes fail before diffusion rather than falling back to black holes.

Warp conditioning remains independent. The manifest records `seed_coverage=1.0`
for complete source RGB and `initial_warp_coverage` for diagnostic geometry
coverage; neither asserts trustworthy 3D geometry. Continuation chunks use the
previous generated final frame. `preview_composite_seed --compare-rendered` can
save a `cache_render_reference/` diagnostic, which is never used for generation.

Removed environment objects are inpainted on the source RGB grid, before the
image warp. The TorchScript adapter uses CPU PyTorch, preserves unmasked pixels,
and caches the repair by input/model hashes. Synthesized masks follow
interpolation and alpha compositing; synthetic RGB never enters the point cloud.
See [LaMa setup](LAMA_SEED_SETUP.md) for commands and diagnostics.
