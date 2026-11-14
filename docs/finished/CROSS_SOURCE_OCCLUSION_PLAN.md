# Cross-source occlusion for separate conditioning buffers

Status: implemented. Keep one conditioning buffer per selected source, but retain
only surfaces visible across the whole composite. Environment ID 0 participates
in depth testing normally. Existing back-projection and placement already put
geometry in destination coordinates; placement must also establish a consistent
destination scale and position for unrelated scenes. No additional depth
estimator is needed.

Keep both public methods: `render_cache()` returns one combined scene buffer;
`render_conditioning()` returns separate source buffers required by diffusion.
Both depth-test a joint table of selected, transformed points. Concatenating
this table preserves source/object ownership; it does not fuse geometry or
modify child caches. Objects from the same source share a conditioning buffer.

## Function changes

1. **`Cache3D_Composite._render_frames()` — route joint winners to source buffers.**
   Use one shared rendering path over the joint sample table, calling existing
   `_resolve_visibility()` once per frame chunk. A source-to-output-slot mapping
   determines where each winning point is written: all sources map to slot 0
   for joint seed/diagnostic renders; conditioning maps each selected source to
   its slot in `conditioning_source_ids`. Obtain ownership from sample metadata
   without allocating full diagnostic outputs, and explicitly translate its index in `sorted(self.sources)`; registered and
   selected source lists can differ. Reuse the same bounded winner gathers,
   depth calculation, and output writes for both output layouts. Preserve
   frame/point chunking and the exact nearest-depth and deterministic
   source/object/pixel tie rules, including tolerance cases. Retain the
   per-source `_sample_tables` and cached `_joint_table`; remove the per-source
   `table=` override once the independent conditioning loop is removed.

2. **`Cache3D_Composite.render_conditioning()` — use the shared joint renderer.**
   Replace the independent per-source render loop with one call to the shared
   renderer, supplying the source-to-output-slot mapping. No separate rendering
   mode or independent-visibility fallback is needed.
   Keep RGB `[1,T,N,3,H,W]`, coverage `[1,T,N,1,H,W]`, and optional camera-Z depth
   `[1,T,N,H,W]`. Nonwinning pixels use RGB -1, coverage 0, and depth 0. Keep
   stable slots for fully occluded or empty selected sources; omit unselected
   sources. `include_depth` controls returned depth only: visibility always
   computes depth internally, on every process. Hidden RGB must be cleared as
   well as coverage because the model encodes them separately.

3. **`Gen3cCompositePipeline.generate()` — update comments and review metadata.**
   Record cross-source visibility alongside the existing latent-concatenation
   strategy and depth tolerance. Keep `include_depth=is_writer`: only the file
   writer needs returned review depth, while every process resolves visibility.
   Derive the initial seed and its coverage from the first conditioning frame,
   removing the repeated `render_cache()` projection. Use one small shared
   recombination helper: sum RGB only where coverage is 1, union coverage, and
   fill uncovered RGB with -1. Never sum raw RGB containing -1 hole values.
   Preserve buffer capacity checks and release review tensors before
   diffusion. No diffusion API or encoder changes are required.

4. **`scripts/evaluate_real_cache_mvp.export_previews()` — describe visible buffers.**
   Record the visibility policy in preview metadata. Existing per-source panels
   should now show occlusion holes; retain the joint reference for comparison.
   Build that image with the same recombination helper instead of rendering
   each frame again with `render_cache()`. Keep the standalone joint renderer
   as an independent test oracle.
   `composite_review.save_render()` can keep its current output schema.

## Verification

- Update `tests/test_composite_rendering.py`: environment/object occlusion in
  both directions, partial overlap, multiple objects and sources, depth ties
  (including tolerance), invalid geometry, empty/fully hidden slots, and source
  ownership when registered and selected source lists differ.
- At every covered pixel, exactly one source buffer must win. Combining visible
  buffers must reproduce joint RGB, depth, and coverage, across camera motion
  and frame/point chunk sizes; holes must never occlude valid surfaces.
  Verify recombined seeds/previews match `render_cache()`, including RGB -1 holes.
- Update `tests/test_composite_pipeline.py` to verify masked buffers reach
  diffusion, writer/nonwriter visibility agrees, and tensor lifetime checks
  still pass. Nonwriters continue to omit returned review depth and files.
- Update `docs/COMPOSITE_GEN3C_USAGE.md` and
  `docs/REAL_CACHE_MVP_EVALUATION.md` to replace independent-visibility claims.

## Scope

Keep back-projection, placement transforms, joint seed/diagnostic behavior, and
model weights unchanged. Single-pixel projection can still expose background
through sampling gaps; adaptive splatting remains a separate patch. Validation
here concerns geometry and buffer contracts; generation-quality A/B testing is
outside scope.
