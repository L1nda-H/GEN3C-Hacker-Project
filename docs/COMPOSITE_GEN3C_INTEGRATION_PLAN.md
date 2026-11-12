# 3. Gen3C integration and MVP validation

Status: planned. Depends on [composite rendering and fusion](COMPOSITE_RENDERING_AND_FUSION_PLAN.md).

## Scope and contract

Generate a video containing one selected object from scene A inside scene B's confirmed environment, using static single-image sources and a camera trajectory in B's coordinates. Preserve existing single-source entry points.

## Work breakdown

1. **Define a shared scene model, with optional JSON serialization.** Define a Python `SceneSpec` with shared validation: version, source IDs, prepared-input references, label-artifact references, environment source, explicit selected IDs, source/object matrices, and target cameras. GUI callers create/update this model directly; optional JSON save/load supports reproducibility and CLI use. Resolve file-relative paths in the JSON loader. Load exact prepared RGB/depth/cameras/validity and validate label fingerprints; do not silently regenerate MoGE depth or resize persisted labels.
2. **Construct the composite.** Add shared construction helpers accepting `SceneSpec` and an optional composite mode to `gen3c_multiview.demo()`. Keep legacy flags working; reject conflicting scene/legacy inputs. Log resolved IDs and transforms for reproducibility. Reuse existing model/Hydra initialization.
3. **Connect conditioning.** Render trajectory chunks and pass fused RGB/coverage to `Gen3cPipeline.generate()`. Verify its configured buffer count accepts the singleton fused buffer; adapt at the entry-point boundary only if the actual model contract requires it. Avoid changing diffusion weights or training.
4. **Use a composed seed.** Render the initial target camera to construct the seed in the existing image-conditioning layout/range. Never pass an original unfiltered source image. Select a well-covered starting view and document deterministic hole handling; MVP uses the rendered RGB fill from stage 2. Coverage masks accompany warp conditioning; do not assume the separate seed-latent path honors them. Validate seed-hole artifacts on real inference before declaring the MVP complete.
5. **Handle continuation.** Keep static composite geometry fixed across chunks and generated-frame cache updates disabled. Use the preceding generated frame for normal autoregressive continuation. If persistent API support is included, add explicit composite seeding and cache capability checks; do not treat the wrapper as `Cache3D_Buffer` or infer behavior solely from `isinstance(Cache4D)`.
6. **Save review outputs.** Save scene configuration, initial composed seed, rendered RGB/depth/coverage previews, and final video. Record source/checkpoint revisions and generation settings.

## Verification and completion

- Verify directly constructed and JSON-loaded scene models share validation and produce equivalent composites; test JSON round trips and relative paths.
- Integration tests with mocked diffusion: exact conditioning layouts/ranges, composed seed, chunk boundaries, unchanged children, disabled updates, invalid artifacts/configuration, and legacy single-source compatibility.
- Run real CUDA SAM/GEN3C inference on representative inputs: inspect masks, placement scale, foreground/background occlusion, excluded-object leakage, holes, and temporal consistency. Contract tests do not establish visual quality.
- Complete when one object is correctly composed with the environment in pre-diffusion renders and remains recognizable and consistently placed in the generated moving-camera video. Record visual failures separately from geometric failures.

Deferred: GUI, persistent-server feature parity, automatic alignment, dynamic sources, and generated-frame labeling.
