# GEN3C extension plan

## Project overview

- Extend GEN3C to generate videos combining selected objects from different source scenes.
- MVP: single-view images and single-camera clips. Temporarily defer geometric multiview association (step 3 of the detailed plan); retain its requirements for later support. Still images bypass video propagation, while clips use SAM 2.1 tracking before labeling.

## Inputs and outputs

- Accept single images, multiview images, and videos as source scenes.
- Accept object selections, placement transforms, and a target camera trajectory.
- Produce a synthesized video of the composed scene.

## Input preparation and object identification

- **Input alignment complete:** `prepare_source()` preserves frame/view keys, validates OpenCV inputs, and packs inputs for existing caches.
- Crop/resize updates intrinsics and uses integer-safe half-pixel nearest sampling for depth/masks/labels; validity stays separate, and ray distance converts to camera-Z.
- **SAM 2.1 image wrapper implemented:** `SAM21Segmenter` uses the official image predictor and automatic mask generator; `requirements.txt` includes the pinned SAM repository as a pipeline dependency while preserving GEN3C dependency pins. SAM performs spatial preprocessing and proposal filtering. The official video predictor remains planned for video propagation.
- SAM and diffusion reuse one application-owned Hydra session through `model_config_session()`. The multiview CLI initializes it at startup; standalone loaders initialize only when needed. Compatible prior SAM imports are supported, and incompatible caller configuration sessions are preserved with a specific error.
- Target CUDA inference on appropriately provisioned GCP hardware, starting with SAM 2.1 large and bfloat16 autocast. No CPU inference fallback is planned.
- Separate appearance encoding and custom duplicate suppression have been removed; custom tracking is excluded from the plan. **25 local tests pass**, covering alignment and mocked SAM API contracts. Real-checkpoint overlay/accuracy validation on GCP and GUI integration remain pending.
- Preserve observation keys and geometry validity; retain ambiguity, unassigned regions, and GUI-confirmed environment masks. Use generic/user-assigned object names.
- Associate masks across cameras using depth and poses; reuse scoped SAM IDs within each video session. Assign canonical IDs before GUI object selection and cache point labeling.
- Save persistent object IDs, label maps, and source metadata.

## Composite cache

- Attach instance labels to each source's existing 3D cache.
- Select objects from different sources and designate one environment source.
- Wrap source caches in `Cache3D_Composite` while preserving source views and frames.
- Store object selections, placement transforms, and time mappings separately from source geometry.
- Follow the [Cache3D composite implementation plan](CACHE3D_COMPOSITE_IMPLEMENTATION_PLAN.md) for implementation details.

## Alignment and scale calibration

- Choose the environment source's coordinate system as the destination frame.
- Estimate relative source scales from known dimensions or user calibration.
- Apply scale, rotation, and translation to place selected objects in the destination frame.
- Check placement against environment geometry and target camera views.
- Align source timelines and repeat static objects across target frames.

## Composite rendering and fusion

- Render the composite along a shared target camera trajectory.
- Project selected geometry into per-source RGB, depth, and coverage buffers.
- Resolve visibility across sources using nearest-surface depth.
- Blend compatible observations of the same surface using confidence weights.
- Preserve uncovered regions for diffusion to complete.
- Produce RGB and coverage conditioning compatible with the existing GEN3C pipeline.

4. **Select, render, and fuse.** For each target frame: resolve source time → apply selected-ID and validity masks → transform selected points → obtain per-source RGB, depth, and coverage → resolve visibility across all sources → fuse compatible observations. Separate user object selection from fusion's choice of source observations. Start with nearest-surface visibility and confidence-weighted blending only among observations of the same surface; leave uncovered pixels for GEN3C. Never interpolate numeric instance IDs. Extend `forward_warp_utils_pytorch.py` only where categorical projection or shared visibility requires it. In `Cache3D_BufferSelector.render_cache()`, rank after selection and prevent its near-full-view shortcut from suppressing selected objects.

## Video diffusion integration

- Pass the composite conditioning through GEN3C video diffusion to synthesize the output video.
- Associate generated-frame labels before enabling generated-frame cache updates.
- Keep generated geometry in the composite coordinate system.

5. **Connect and verify.** Wire labeled source loading into `gen3c_multiview.demo()` and construct the composite there; retain the generation pipeline's RGB/coverage tensor contract. Feature-level fusion, if chosen later, belongs in the model conditioning path. Before enabling generated-frame cache updates, associate new labels and retain geometry in the correct coordinate frame. Check cross-view/frame ID consistency, selected-only rendering, cross-source occlusion, and static/dynamic time alignment. First milestone: one manually placed static object from one source inside another source's environment.

## Validation and first milestone

- Validate object identity, placement, occlusion, and temporal consistency.
- Start with one manually placed static object inside another source's environment.

## Further extensions

- Extend to multiple objects, multiview sources, and dynamic video sources.
- Add automatic placement and scale estimation.
- Explore improved fusion policies and feature-level conditioning.
