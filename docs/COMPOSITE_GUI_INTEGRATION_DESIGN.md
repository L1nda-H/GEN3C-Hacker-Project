# Composite scene integration with the Gen3C GUI

Status: proposed implementation design. No runtime changes are included in this document.

## Goal and initial scope

Allow a user to load a prepared composite scene, inspect its placed objects,
edit a camera trajectory in the existing GUI, and generate a video with the
existing pretrained Gen3C model. Preserve the scene's source geometry, labels,
selections, and transforms throughout generation.

The first version supports loading an existing `SceneSpec` JSON and editing
cameras. Creating labels, selecting objects, and manipulating object placements
inside the GUI are later extensions. Sources remain static, with one prepared
image per source, as required by the current composite implementation.

## Existing implementation

The composite generation path already exists:

- `cosmos_predict1/diffusion/inference/cache3d_composite.py` implements
  `Cache3D_Composite`, selection, placement, and rendering.
- `composite_scene.py` persists `SceneSpec`, loads exact prepared inputs and
  labels, and builds the composite scene.
- `gen3c_composite_pipeline.py` wraps an initialized `Gen3cPipeline`. It renders
  conditioning, constructs the first seed, continues generation in chunks, and
  writes review artifacts.
- `gen3c_composite.py` provides a CLI with a `demo()` entry point and MP4 saving.

See [COMPOSITE_GEN3C_USAGE.md](COMPOSITE_GEN3C_USAGE.md) for artifact preparation
and existing CLI usage. A new standalone demo is not a prerequisite.

The current GUI follows a separate path:

1. `gui/api/client.py` loads seeding data and sends requests.
2. `gui/api/server.py` exposes the request endpoints.
3. `gui/api/server_cosmos_base.py` adapts seeding and inference requests.
4. `gui/api/server_cosmos.py` loads `Gen3cPersistentModel`; `multi_gpu.py`
   provides its worker proxy when multiple GPUs are used.
5. `gen3c_persistent.py` retains model weights, renders its cache, invokes
   diffusion, and saves video.

Testbed's C++/CUDA renderer provides interactive viewport reprojection. Its
framebuffer is separate from the Python conditioning buffers used by diffusion.

## Rendering contracts to preserve

Composite rendering exposes two distinct operations:

| Operation | Output | Purpose |
| --- | --- | --- |
| `render_cache()` | RGB `[1,T,1,3,H,W]`, coverage `[1,T,1,1,H,W]` | Joint visibility for the initial composed seed and previews |
| `render_conditioning()` | RGB `[1,T,N,3,H,W]`, coverage `[1,T,N,1,H,W]` | Independent source buffers for diffusion |
| `render_diagnostics()` | Depth, coverage, nearest depth, and winner metadata | Preview depth and inspection |

RGB values use `[-1,1]`. Cameras use OpenCV world-to-camera matrices and pixel
intrinsics. Source buffer ordering comes from `conditioning_source_ids` and
must remain stable for a generation job. Do not substitute joint visibility
rendering for per-source conditioning.

The environment defines destination world coordinates. Placement applies
`T_source @ T_object @ p_source`. Retain exact prepared inputs and label bindings;
do not rerun MoGe or resize source label maps while loading a scene.

## Proposed architecture

Add an explicit composite mode alongside legacy image/video seeding. Reuse the
already loaded diffusion pipeline through `Gen3cCompositePipeline`; keep the
legacy cache path intact.

```text
SceneSpec + prepared inputs + labels
                 |
          server scene session
          /                  \
 joint RGB/depth previews    GUI-authored target cameras
          |                          |
    Testbed viewport          composite generation adapter
                                     |
                          render_conditioning() per source
                                     |
                          Gen3cPipeline.generate()
                                     |
                          MP4 + review artifacts + GUI result
```

Do not assign a composite directly to the existing `self.cache` and run the
legacy inference loop. That loop expects `input_labels`, `input_depth`,
`update_cache()`, and `render_cache(selected_ids=...)`. Composite sources instead
have their own selection state and must retain their original geometry.

### Scene loading and session state

Introduce a dedicated scene-loading request rather than extending
`SeedingRequest`, whose required arrays describe ordinary image/video seeds.
Proposed names and fields:

- `CompositeSceneLoadRequest`: request ID and server-local `scene_spec_path`.
- `CompositeSceneLoadResult`: scene revision, source/selection summary,
  conditioning buffer count, target resolution, initial camera trajectory,
  and preview RGB/depth/camera data.
- Composite generation requests reference the loaded scene revision, include
  an explicit prompt, and carry the GUI-authored cameras.

For the initial version, artifacts must already exist on the inference server.
Label the path as server-local in the GUI. Client uploads and artifact packaging
are separate work; a client path is not assumed to exist on a remote server.

Load and validate a candidate scene before replacing the current session.
Serialize scene loading, clearing, and inference with the model execution lock.
Reject stale revision requests. Successful scene replacement resets legacy
camera history and generated results associated with the old scene. Failure
leaves the previous session available. Existing ordinary seeding switches back
to legacy mode and clears composite state.

Store the loaded `SceneSpec` and a validated composite for previews. Initially,
generation may use the wrapper's existing scene construction. If retaining a
second set of child tensors is too costly, release the preview composite before
generation and rebuild it afterward. Refactoring the wrapper to accept a
prebuilt composite is an optimization, not a reason to duplicate its algorithm.

### Generation adapter

Add an explicit composite inference branch in the persistent model/server
adapter. It should:

1. Check the requested scene revision and convert GUI cameras using the existing
   camera conversion helpers. Construct a request-specific copy of `SceneSpec`
   with those cameras; do not mutate the loaded scene's trajectory.
2. Validate model resolution, camera matrices, prompt, trajectory length,
   conditioning buffer capacity, and initial seed coverage.
3. Call `Gen3cCompositePipeline(self.pipeline).generate(...)` using a unique
   review directory and the existing loaded model/device.
4. Save returned frames with `save_video()` and adapt the result to the GUI's
   established inference response and video download mechanism.
5. Surface guardrail rejection and validation errors as failed requests;
   do not report success or publish an old output path.

The wrapper currently requires a nonempty explicit prompt and complete chunks
with one-frame overlap: `(T - 1) % (chunk_size - 1) == 0`. It uses the composed
first-camera RGB as the first image seed, checks coverage (default threshold
0.5), and uses the preceding generated frame for later chunks. Source caches
are never updated with generated content.

Treat each GUI Generate action as one independent composite job containing the
whole trajectory. Bypass legacy cross-request pose-history stitching for this
mode. Internal chunking belongs to `Gen3cCompositePipeline`. The GUI should
report the effective supported duration/frame count if padding the last camera
is needed; trim output to the requested length after generation and record both
counts in the result manifest. Reject trajectories shorter than one chunk in
the first version with an actionable duration message.

Do not enable automatic viewpoint inference or continuation across separate
composite jobs until their seed/history semantics are implemented explicitly.

### Viewport preview

The first version uses server-rendered RGB/depth reference views of the composed
scene. Render these with joint visibility at selected trajectory cameras,
convert RGB/depth into Testbed's established source-view format, and let Testbed
reproject them while the user moves the camera. Transfer coverage as image alpha
where supported and invalidate uncovered depth samples before reprojection.
Check the existing source-view conversion code for depth and camera conventions;
keep camera-Z depth consistent with that conversion.

This preserves selections and transforms without implementing the composite
rasterizer in CUDA. The approximation has limits: hidden geometry cannot be
recovered from a few reference views. Provide a refresh action that renders a
new joint view at the current camera. A later server preview request can be
debounced and tagged with the scene/camera revision so late responses do not
replace newer views. Serialize preview GPU work with generation initially.

Do not feed unfiltered original source images to the viewport: they show
unselected objects and original placements. Do not treat reprojected Testbed
frames as generation conditioning. Disable automatic insertion of generated
keyframes into the composite viewport initially, so users can continue to
inspect the authored source composition.

### Optional results and multi-GPU execution

The existing wrapper does not return the persistent model's estimated-depth
result. Composite mode should initially advertise that generated-depth/keyframe
return is unavailable and disable dependent GUI controls. Preview depth is a
separate result from generated-frame depth.

Reuse review renders for optional conditioning visualization, or add a wrapper
result/callback that exposes buffers. Do not silently accept the GUI's
show-cache-renderings option without producing the requested visualization.
Disable that option in the first increment if it is not yet implemented.

Extend `multi_gpu.py` task dispatch and proxy methods for scene loading and
composite inference. All ranks must load the same scene revision and execute
the same diffusion sequence; only the designated writer saves previews,
manifests, and MP4s. Coordinate validation failures before distributed model
collectives so one rank cannot exit while others enter inference. Initial
single-GPU support must explicitly reject composite mode with multiple GPUs
until this protocol is implemented.

## File-level implementation plan

| File | Planned change |
| --- | --- |
| `gui/api/api_types.py`, `api_serialization.py` | Scene load/result types, scene revision and capability metadata; round-trip encoding |
| `gui/api/server.py` | Composite scene loading endpoint using existing request/error handling |
| `gui/api/server_cosmos_base.py` | Route composite load/generation, execution lock, camera conversion, independent-job semantics |
| `cosmos_predict1/diffusion/inference/gen3c_persistent.py` | Composite session lifecycle and adapter around the existing wrapper |
| `gui/api/client.py` | Load scene, install preview views and trajectory, send revision-bound generation requests |
| `gui/src/testbed.cu`, GUI bindings/headers as needed | Scene mode controls, preview refresh, capability-dependent controls |
| `gui/api/multi_gpu.py` | Worker/proxy support and coordinated scene revisions |
| `gen3c_composite_pipeline.py` | Only targeted hooks needed for progress, reuse, or visualization; retain conditioning semantics |

## Delivery and verification

1. **Backend adapter:** load a scene and generate through a headless server
   request on one GPU. Use fake diffusion to verify per-source tensors, initial
   seed selection, chunk overlap, unchanged source geometry, and saved results.
2. **GUI loading and cameras:** load a prepared scene, display joint previews,
   edit cameras, and generate one independent job. Verify a moved selected
   object appears at the same destination in preview and conditioning.
3. **Lifecycle and failure handling:** verify invalid loads preserve the old
   scene, stale revisions fail, scene switches clear history, guardrail failure
   produces no successful result, and unsupported controls are disabled.
4. **Multi-GPU and optional visualization:** add only after the single-GPU
   request path works; verify identical source order and scene revision on all
   ranks and writer-only artifacts.

Tests should cover API serialization, camera/intrinsics conversion, frame-count
padding/trimming, buffer-capacity rejection, seed coverage failure, and legacy
image/video behavior. Reuse existing composite fixtures and fake diffusion
tests rather than duplicating renderer unit tests. Run one real-model smoke
test with an existing scene when a checkpoint/GPU is available; compare the
review manifest and conditioning against the CLI for the same cameras/settings.
Visual fidelity and viewpoint coverage require inspection beyond mocked tests.

## Acceptance criteria

- An existing scene JSON can be loaded without rebuilding depth or labels.
- Selected objects and transforms agree between preview and review artifacts.
- GUI-authored cameras reach the composite pipeline in the correct convention.
- Diffusion receives independent per-source buffers in stable order.
- Long jobs preserve fixed source geometry and correct one-frame chunk overlap.
- The generated MP4 is returned through the normal GUI result flow.
- Legacy image/video seeding remains usable after switching modes.
- Unsupported capabilities are visible before a generation request starts.
