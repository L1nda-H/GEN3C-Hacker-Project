# Adaptive splatting patch

Status: planned. Extend the static single-image composite renderer using existing
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
RGB/depth/coverage layouts, ranges, devices, and hole values. Keep the original
point renderer as a selectable baseline. Separate object-buffer rendering,
Gaussian alpha compositing, model changes, and new-view geometry generation are
outside this patch.

## Implementation

1. **Retain source-pixel support.** Extend `CompositeSamples` in
   `cache3d_composite.py` with two source-space pixel tangent vectors aligned with
   its selected points. Estimate them on the original point grid before flattening,
   using valid, depth-compatible neighbors with the same canonical label. Use
   one-sided differences at boundaries. Where neighbors are insufficient, use a
   conservative pixel-cell estimate from the prepared source camera and depth;
   pass that runtime metadata through `build_composite_scene()` if needed.
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
   candidate's camera-Z from the pixel ray's intersection with its local plane.
   Carry the winning candidate depth through RGB/depth gathering so output depth
   matches visibility, including tolerance ties. Gather RGB and categorical IDs
   from the winning source point; never average IDs or blend different objects.
4. **Bound work and uncertainty.** Batch expanded point–pixel candidates as well
   as frames and source points. Never allocate all points times the maximum
   footprint area. Add explicit runtime controls for the candidate budget and
   maximum footprint extent. Fall back to the existing point projection for
   degenerate, unreliable, near-plane-crossing, or over-limit footprints. Do not
   bridge label/depth discontinuities or enlarge footprints to fill genuine
   disocclusions. Preserve coverage from actual accepted contributions.

Source-space support may be reused while geometry and labels are unchanged;
invalidate derived support after either changes. Placement changes require
retransforming support. Keep configuration at the composite renderer boundary
and record the chosen settings in review metadata; diffusion APIs stay unchanged.

## Verification and acceptance

- Preserve baseline outputs when adaptive mode is disabled; verify identity-view
  alignment and deterministic results across frame, point, and candidate batches.
- With object scale fixed, approach a planar object at 1x, 2x, and 4x
  magnification. A solid 4x4 pixel patch should cover 16, 64, and 256 target
  pixels under aligned test cameras, without background showing inside it.
- Check fractional camera motion, oblique surfaces, silhouettes, thin strips,
  internal cutouts, invalid depth, depth edges, partly offscreen splats, and
  foreground/background occlusion. Compare RGB, winner IDs, depth, and coverage;
  total coverage alone cannot detect foreground perforation.
- On GCP CUDA, compare baseline/adaptive real-image renders and measure render
  time and peak memory at the existing chunk size and output resolution. Inspect
  moving-camera stability and then one GEN3C generation. CPU probes establish
  feasibility only; accept the patch after gaps improve without material
  silhouette growth or incorrect occlusion, and memory respects the configured
  candidate budget. No library replacement is required for this milestone.
