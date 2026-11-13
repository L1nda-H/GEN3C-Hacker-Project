# Cross-source occlusion for separate conditioning buffers

Status: planned. Keep one conditioning buffer per selected source, but retain
only surfaces visible across the whole composite. Environment ID 0 participates
in depth testing normally. Existing back-projection and placement already put
geometry in destination coordinates; no additional depth estimator is needed.

## Function changes

1. **`Cache3D_Composite._render_frames()` — route joint winners to source buffers.**
   Add an internal mode that uses the joint sample table and existing
   `_resolve_visibility()` once per frame chunk, then writes each winning point
   into its owning source's buffer. Obtain ownership from sample metadata and
   explicitly map its index in `sorted(self.sources)` to the buffer index in
   `conditioning_source_ids`; these lists can differ. Preserve bounded winner
   gathers and frame/point chunking. This shares the joint renderer's exact
   nearest-depth and deterministic source/object/pixel tie rules, including
   tolerance cases, rather than approximating them with independent depth maps.

2. **`Cache3D_Composite.render_conditioning()` — use joint visibility by default.**
   Replace the independent per-source render loop with the source-routing mode.
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
   Update the seed comment: the seed and separate buffers now share joint
   visibility. Preserve buffer capacity checks and release review tensors before
   diffusion. No diffusion API or encoder changes are required.

4. **`scripts/evaluate_real_cache_mvp.export_previews()` — describe visible buffers.**
   Record the visibility policy in preview metadata. Existing per-source panels
   should now show occlusion holes; retain the joint reference for comparison.
   `composite_review.save_render()` can keep its current output schema.

## Verification

- Update `tests/test_composite_rendering.py`: environment/object occlusion in
  both directions, partial overlap, multiple objects and sources, depth ties
  (including tolerance), invalid geometry, empty/fully hidden slots, and source
  ownership when registered and selected source lists differ.
- At every covered pixel, exactly one source buffer must win. Combining visible
  buffers must reproduce joint RGB, depth, and coverage, across camera motion
  and frame/point chunk sizes; holes must never occlude valid surfaces.
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
