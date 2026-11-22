# 2. Composite rendering and fusion

Status: implemented and verified on synthetic CPU scenes. Depends on [composite cache and placement](COMPOSITE_CACHE_AND_PLACEMENT_PLAN.md). CUDA execution and representative real-source inspection remain pending; diffusion wiring is covered by the [integration plan](COMPOSITE_GEN3C_INTEGRATION_PLAN.md).

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

## Implemented API and numerical policy

```python
composite = Cache3D_Composite(
    image_size=(height, width),  # Required to render; independent of source sizes.
    depth_tolerance=1e-5,       # Absolute camera-Z tolerance in destination units.
    frame_chunk_size=2,
    point_chunk_size=65536,
)
# Register labeled children, select IDs, and set placement as in stage 1.
rgb, coverage = composite.render_cache(target_w2cs, target_intrinsics)
depth, same_coverage = composite.render_cache(target_w2cs, target_intrinsics, render_depth=True)
diagnostics = composite.render_diagnostics(target_w2cs, target_intrinsics)
# Get RGB, depth, coverage, and diagnostics with one visibility calculation:
all_buffers = composite.render_diagnostics(target_w2cs, target_intrinsics, include_rgb=True)
```

- Target cameras are finite floating tensors `[1,T,4,4]` and `[1,T,3,3]` on the same device. Poses must be right-handed rigid OpenCV world-to-camera matrices; intrinsics use the upper-triangular pinhole convention. `T` must be positive. Any nonnegative integer `start_frame_idx` is accepted for static geometry.
- Outputs are float32 on the target camera device, including inside diffusion autocast contexts. RGB/depth/coverage retain the layouts and hole values above. A nondefault shared source `rgb_range` is linearly mapped to `[-1,1]` for rendering.
- A point contributes only with finite camera coordinates, positive camera-Z, and finite projected coordinates. Nearest-pixel assignment uses `floor(uv + 0.5)` with half-open bounds `[-0.5,W-0.5)` and `[-0.5,H-0.5)`; exact halfway positions round toward the larger pixel index. Each point contributes to one pixel; sparse projections and disocclusions remain holes.
- `nearest_surface_winners()` in `forward_warp_utils_pytorch.py` computes the true minimum camera-Z jointly across all selected samples, then chooses the smallest categorical priority among samples within tolerance of that minimum. Comparing against the true minimum avoids chains of near-equal depths. No surface colors are averaged. The private `_resolve_visibility()` method isolates this policy for future replacement.
- RGB, coverage, and returned depth use one winner. Returned depth is that winner's actual camera-Z, which may exceed the true minimum within tolerance. Diagnostics separately expose `nearest_depth [1,T,1,H,W]` and `winners [1,T,H,W,3]`: `(source index, canonical instance ID, original source pixel index)`. `source_ids` maps indices to sorted source IDs; all winner fields are int64, with `-1` for holes.
- Child caches and the concatenated selected sample table remain on CPU. Per-object sample copies are released before frame rendering; depth-only requests omit RGB samples. The table is rebuilt per request rather than introducing another invalidation cache.
- Projection transfers point chunks for each target-frame chunk in two passes; GPU projection memory scales with `frame_chunk_size * point_chunk_size`, plus per-chunk visibility buffers and the returned video buffers. Chunk boundaries and registration order do not affect tie priority.
- Winner gathering also runs in batches of at most `point_chunk_size` covered pixels, bounding temporary indices, selected attributes, and CPU/GPU transfers.
- RGB-only requests allocate/populate only RGB and coverage; depth-only and default diagnostics requests omit RGB output. `render_diagnostics(..., include_rgb=True)` obtains all buffers in one render.
- Source preparation and rendering share camera convention validation with absolute pose tolerances. Geometry and camera projection use float32; very large coordinate magnitudes retain the precision limitation identified in the placement review.

## Verification and review artifacts

`python -m unittest discover -s tests -q` runs 93 tests: 92 pass and the CUDA parity test skips when no GPU is available. The 16 rendering tests cover projection and camera-Z, rigid camera rotation, identity parity with the existing renderer, cross-source and within-source occlusion, tolerance chains, categorical ties with large integer IDs, invalid geometry, rounding/bounds, empty selections, unequal source sizes, placement, moving cameras, exact chunk equivalence, registration order, RGB ranges, unchanged children, and autocast/default-dtype behavior. Optimization regressions additionally check combined-output parity, absence of unrequested full output buffers, bounded winner attribute gathers across frame boundaries, and shared individual/batched camera tolerances.

Run `python -m scripts.preview_composite_render --output-dir outputs/composite_preview` to regenerate `preview.png` and `buffers.npz`. The synthetic scene places a checkerboard prop from a second source in front of a blue environment and behind its green foreground post. Three moving-camera views save RGB, depth, coverage, exact winner IDs, and target cameras. The contact sheet has been inspected: the post occludes the prop, the prop occludes the background, and camera-motion disocclusion holes agree across all three buffers. This establishes synthetic geometry behavior; real-source visual quality and CUDA/diffusion inference remain separate validation work.
