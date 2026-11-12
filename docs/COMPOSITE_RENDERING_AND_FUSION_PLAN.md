# 2. Composite rendering and fusion

Status: planned. Depends on [composite cache and placement](finished/COMPOSITE_CACHE_AND_PLACEMENT_PLAN.md).

## Scope and contract

- Implement `Cache3D_Composite.render_cache(target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0)` using the composite's stored selections.
- RGB output: `[B,T,1,3,H,W]`; coverage: `[B,T,1,1,H,W]`. With `render_depth=True`, return camera-Z depth `[B,T,1,H,W]` plus identical coverage. The singleton dimension is the fused conditioning buffer.
- RGB uses GEN3C's `[-1,1]` range; coverage is binary for MVP. Uncovered RGB is `-1`, depth is `0`, coverage is `0`. Define output dtype/device consistently with existing callers.
- Static geometry repeats across frames; validate positive frame count and nonnegative offset without temporal bounds on source images.

## Work breakdown

1. **Project candidates.** Consume selected, transformed samples from stage 1. Require a positive integer target `(H,W)`, supplied through composite configuration or existing generation settings; intrinsics alone do not define output size. Reuse existing camera/projection helpers; reject nonfinite points, nonpositive camera-Z, and out-of-frame contributions. Permit different source resolutions. Specify pixel sampling and boundary rules; nearest-pixel projection with documented rounding is sufficient for MVP.
2. **Resolve joint visibility.** Build a shared nearest-depth rasterization path in `forward_warp_utils_pytorch.py` if needed. Resolve visibility across all source/object contributions before RGB accumulation. Existing depth-weighted splatting can mix foreground/background; independently warping then comparing blended depths is insufficient.
3. **Specify ties.** Document a nonnegative depth tolerance and compare candidates against the nearest camera-Z depth in destination units. Within tolerance, choose by sorted source ID, canonical object ID, then original pixel index, independent of registration/chunk order; never average different surfaces. Keep fusion behind an internal replaceable policy, initially nearest-surface only.
4. **Produce buffers.** Derive RGB, camera-Z depth, and coverage from the same winning contributions. Never interpolate instance IDs. Preserve holes; do not inpaint in the cache. Expose optional winner/depth diagnostics separately from the diffusion contract.
5. **Control memory.** Render in target-frame chunks; retain child geometry on CPU where practical. Preserve identical results across chunk sizes and source registration order.

## Verification and completion

- Synthetic tests: known projection/depth, identity parity, front/back object overlap, environment occlusion, equal-depth ties, invalid geometry, empty selections, holes, and unequal source sizes.
- Verify all three outputs agree on visibility; test reversed source registration, repeated static frames, moving cameras, and chunk equivalence.
- Inspect RGB/depth/coverage for one manually placed object in another scene's environment before diffusion.

Deferred: confidence blending, learned/feature fusion, hole filling, and dynamic time mapping.
