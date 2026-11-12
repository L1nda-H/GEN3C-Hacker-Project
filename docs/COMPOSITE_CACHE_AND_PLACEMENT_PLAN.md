# 1. Composite cache and placement

Status: implemented. First stage of the **static single-image MVP**; existing segmentation, labeling, and child caches are prerequisites. `cache3d_composite.py` provides registration, selection, manual placement, and selected sample iteration. Rendering/fusion remains the [next stage](COMPOSITE_RENDERING_AND_FUSION_PLAN.md).

## Scope and contract

- Add `cosmos_predict1/diffusion/inference/cache3d_composite.py` with `Cache3D_Composite`; compose child caches rather than inherit buffer behavior.
- MVP children have `B=F=N=V=1`, registered labels, and compatible RGB ranges. Reject unsupported layouts explicitly. Preserve the broader multiview/video roadmap separately.
- Exactly one source supplies environment ID `0`. Other sources contribute selected positive IDs only; reject `-1`. Empty object selections are valid.
- Identify objects by `(source_id, instance_id)`; resolve registry aliases. Never compare bare IDs across sources.

## Work breakdown

1. **Define state and registration.** Store `sources`, `environment_source_id`, `selections`, `source_transforms`, and `object_transforms`. Implement `add_source(source_id, cache, is_environment=False)`; validate unique IDs, registry ownership, and cache consistency. Require an environment before rendering.
2. **Define selection.** Implement `set_selection(source_id, instance_ids)` as replacement, retaining geometry. Environment ID `0` is always included from the designated source; positive objects there remain explicitly selectable. Revalidate IDs after label corrections; fail clearly if source attachment becomes invalid.
3. **Define placement.** Implement `set_transform(source_id, matrix, instance_id=None)`. Accept finite orientation-preserving similarity transforms: positive uniform scale, rotation, translation; reject shear and singular matrices. Keep the environment source transform at identity and reject object transforms targeting environment ID `0`; positive IDs from that source remain independently placeable.
4. **Fix coordinates and pivots.** Use environment world coordinates and OpenCV world-to-camera cameras. For column vectors, `p_destination = T_source @ T_object @ p_source`. Object transforms act in source coordinates; default identity. Express a chosen source-space pivot explicitly as `Translate(pivot) @ RotateScale @ Translate(-pivot)`. Manual scale calibration supplies the matrix; automatic estimation is deferred.
5. **Expose render inputs.** Add an internal iterator yielding selected valid RGB samples and transformed world points per source/object. Selection intersects `input_mask`; transforms never alter child points, RGB, labels, or validity. Repeat static geometry at every requested target time; add no timeline API yet.

## Verification and completion

- Test independent ID namespaces, environment rules, alias resolution, invalid selections/layouts/transforms, transform order/pivot, and unchanged children after edits.
- Test known scale/rotation/translation on synthetic points and identity placement parity.

Deferred: automatic placement, GUI, multiview association, dynamic sources, and generated-frame updates.

## Implemented API

```python
from cosmos_predict1.diffusion.inference.cache3d_composite import Cache3D_Composite

composite = Cache3D_Composite()  # Shared RGB range defaults to GEN3C's [-1, 1].
composite.add_source("room", environment_cache, is_environment=True)
composite.add_source("props", object_cache)  # IDs must match each child registry's source_id.
composite.set_selection("props", [cup_id])
composite.set_transform("props", source_to_environment)
composite.set_transform("props", object_in_source, instance_id=cup_id)

for samples in composite._iter_render_inputs():
    # source_id, canonical instance_id, destination-world points [K,3],
    # RGB [K,3], and original row-major source pixel_indices [K].
    consume(samples)
```

Positive objects start unselected. Environment ID `0` is always included from the designated source, including after an empty replacement selection. Other sources' environment and all unresolved labels are excluded. Sources may have different image sizes. Samples are ordered by source ID, resolved object ID, and original pixel index; empty objects yield no samples. Validity intersects the child mask and finite world points. Render samples use float32 points; placement matrices are validated/stored in float64. RGB retains the child's dtype. Returned sample tensors own their storage, and static samples can be reused for every target frame.

Label corrections through `child.replace_labels()` trigger one full validation and canonical-ID refresh, preserving placement and selection. If two placed IDs merge with different object transforms, explicitly call `set_transform()` for the surviving object (or its alias) to choose its placement. Such conflicts block only selected objects. Stale attachments or invalid IDs fail before any samples are yielded.

## Review fixes

- **Attachment tracking:** child `geometry_revision` increments on buffer updates. The composite checks child identity, geometry revision, fingerprint, and observation keys; updates require rebuilding even when fingerprint/keys were initially absent.
- **Validation cost:** full RGB/label scans run at registration and correction refresh. Placement, selection, and unchanged render requests check only attachment/label metadata. `refresh_source(source_id)` explicitly validates in-place edits; arbitrary tensor mutation is not automatically tracked.
- **Alias handling:** refresh canonicalizes stored selections/transforms once per changed label snapshot. Placement setters use canonical keys directly. Conflicting merges remain explicit, while unselected conflicts do not suppress unrelated objects.
- **Precision:** selected geometry is transformed into float32 render samples, reducing point memory and avoiding per-sample float64 computation; child precision and float64 placement validation remain unchanged.

Verification: `python -m unittest discover -s tests -q` passes all 76 CPU tests, including regressions for unfingerprinted buffer invalidation, skipped repeated validation, correction refresh, unselected merge conflicts, and float32 samples. CUDA rendering and diffusion are outside this stage.
