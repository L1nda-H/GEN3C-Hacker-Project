# Adaptive splatting patch

Status: implemented with CPU coverage/visibility and integration tests. Real CUDA
performance and GEN3C visual validation remain pending. Extends the static
single-image composite renderer using existing
PyTorch operations only; no new libraries, installations, or dependency changes.
Build on the [completed integration implementation](finished/COMPOSITE_GEN3C_INTEGRATION_PLAN.md).
Real CUDA visual validation of that integration remains pending.

## Problem and scope

`nearest_surface_winners()` projects each source point into one target pixel.
Moving the camera closer spreads samples apart, leaving background visible
between foreground samples even when composite coverage is 100%. Disabling
object scaling does not prevent this. Existing fixed-support bilinear splatting
also cannot cover arbitrary magnification.

Give each sample the projected footprint of its source pixel. Preserve source
geometry, object selection, placement, nearest-surface visibility, and GEN3C's
RGB/depth/coverage layouts, ranges, devices, and hole values. Adaptive rendering
is the composite pipeline, with no selectable point-rendering mode. Extend the
shared joint visibility path:
previews and diffusion conditioning must use the same footprint coverage and
occlusion; conditioning still routes winners into source-specific latent buffers.
Additional object-buffer rendering or frame-compositing stages,
Gaussian alpha compositing, model changes, and new-view geometry generation are
outside this patch.

## Implementation

1. **Retain source-pixel support.** Extend `CompositeSamples` in
   `cache3d_composite.py` with two source-space pixel tangent vectors aligned with
   its selected points. Estimate them on the original point grid before flattening,
   using valid, depth-compatible neighbors with the same canonical label. Use
   one-sided differences at boundaries. Where neighbors are insufficient, use a
   conservative pixel-cell estimate from the prepared source camera and depth;
   pass that runtime metadata through `build_composite_scene()`.
   Do not change persisted artifacts. Transform tangents with the placement's
   linear part, while points also receive translation.
2. **Project footprints.** Extend `_resolve_visibility()` and the candidate
   generation in `forward_warp_utils_pytorch.py`. Represent each pixel cell by
   half-tangent extents around its point and project its corners into the target
   camera. Enumerate target pixel centers inside the projected quadrilateral,
   with a consistent half-open edge rule. Estimate support independently per
   target frame; never use one global splat radius. Test footprint/image overlap
   rather than rejecting a point solely because its center is offscreen.
3. **Keep hard visibility.** Reuse the two-pass minimum-depth and deterministic
   source/object/pixel tie policy. For a reliable planar footprint, obtain each
   candidate's camera-Z from the pixel ray's intersection with its local plane
   during both visibility passes. During gathering, recompute depth from the
   winning plane with the same intersection calculation; do not substitute its
   center depth or the global minimum when a tolerance tie selects another plane.
   This avoids storing all candidate depths. Gather RGB and categorical IDs
   from the winning source point; never average IDs or blend different objects.
4. **Bound work and uncertainty.** Batch expanded point–pixel candidates as well
   as frames and source points. Never allocate all points times the maximum
   footprint area. Use an internal candidate-batch budget with no footprint-size
   cutoff. Clip footprint bounds to the image and batch their
   pixel enumeration, including a single footprint larger than the budget.
   The budget bounds temporary candidate storage, not total work or coverage.
   Large valid footprints must never revert to points: that reintroduces grids
   on camera approach. Fall back to the existing point projection only for
   degenerate, unreliable, or near-plane-crossing footprints. Do not
   bridge label/depth discontinuities or enlarge footprints to fill genuine
   disocclusions. Preserve coverage from actual accepted contributions.

Source-space support may be reused while geometry and labels are unchanged;
invalidate derived support after either changes. Placement changes require
retransforming support. Keep fixed defaults inside the renderer and record them
in review metadata; diffusion APIs stay unchanged.

Internal defaults: 262144 expanded candidates per batch and a relative
source-neighbor camera-Z tolerance of 0.05. There are no CLI flags, constructor
options, or pipeline configuration plumbing for these implementation details.
Adaptive sources require their original rigid w2c and pinhole intrinsics at
runtime; the scene builder supplies these without changing Cache3D_Base.
Missing axis neighbors use a frontoparallel pixel cell at the sample's source
camera-Z. This is an approximation at thin structures and silhouettes, not new
geometry reconstruction. Independently estimated cells may leave seams on
curved/noisy depth; planar coverage alone does not establish real-image quality.

Implementation files:

- `cache3d_composite.py`: runtime source cameras, selected/transformed pixel
  tangents in cached sample tables, shared adaptive visibility, winning-plane
  depth gathering, and review settings. Existing refresh/selection/placement
  invalidation also invalidates support.
- `forward_warp_utils_pytorch.py`: streamed pixel-cell candidates, shared
  ray-plane intersection, and one two-pass visibility reducer shared by point
  and pixel-cell candidates. Point projection and degenerate-cell fallback also
  share bounds/rounding logic. Legacy bilinear warping is unchanged.
- `composite_scene.py`: source-camera handoff, without changing scene/prepared
  artifact formats.
- `gen3c_composite_pipeline.py`: renderer defaults in `generation.json`.
  Reuse the existing joint winners for
  source slots and seed construction; no additional rendering stage.
- `scripts/preview_composite_render.py`: uses the same adaptive renderer as generation.
- `tests/test_adaptive_splatting.py` and `tests/test_composite_pipeline.py`:
  CPU geometry, batching, invalidation, and conditioning integration checks;
  CUDA parity coverage skips when no GPU is available.

## Verification and acceptance

- Verify identity-view alignment and deterministic results across frame, point,
  and candidate batches. Degenerate-cell fallback retains point-projection semantics.
- With object scale fixed, approach a planar object at 1x, 2x, and 4x
  magnification. A solid 4x4 pixel patch should cover 16, 64, and 256 target
  pixels under aligned test cameras, without background showing inside it.
- Check fractional camera motion, oblique surfaces, silhouettes, thin strips,
  internal cutouts, invalid depth, depth edges, partly offscreen splats, and
  foreground/background occlusion. Compare RGB, winner IDs, depth, and coverage;
  total coverage alone cannot detect foreground perforation.
- Include a single valid splat larger than the candidate budget and verify full
  coverage, batch-size invariance and bounded candidate enumeration. Include
  curved/noisy depth and tolerance ties where the winning plane is not the
  global minimum; verify gathered depth agrees with visibility.
- On GCP CUDA, inspect real-image renders and measure render
  time and peak memory at the existing chunk size and output resolution. Inspect
  moving-camera stability and then one GEN3C generation. CPU probes establish
  feasibility only; accept the patch after gaps improve without material
  silhouette growth or incorrect occlusion, and memory respects the configured
  candidate budget. No library replacement is required for this milestone.
