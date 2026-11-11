# 1. Composite cache and placement

Status: planned. First stage of the **static single-image MVP**; existing segmentation, labeling, and child caches are prerequisites.

## Scope and contract

- Add `cosmos_predict1/diffusion/inference/cache3d_composite.py` with `Cache3D_Composite`; compose child caches rather than inherit buffer behavior.
- MVP children have `B=F=N=V=1`, registered labels, and compatible RGB ranges. Reject unsupported layouts explicitly. Preserve the broader multiview/video roadmap separately.
- Exactly one source supplies environment ID `0`. Other sources contribute selected positive IDs only; reject `-1`. Empty object selections are valid.
- Identify objects by `(source_id, instance_id)`; resolve registry aliases. Never compare bare IDs across sources.

## Work breakdown

1. **Define state and registration.** Store `sources`, `environment_source_id`, `selections`, `source_transforms`, and `object_transforms`. Implement `add_source(source_id, cache, is_environment=False)`; validate unique IDs, registry ownership, and cache consistency. Require an environment before rendering.
2. **Define selection.** Implement `set_selection(source_id, instance_ids)` as replacement, retaining geometry. Environment ID `0` is always included from the designated source; positive objects there remain explicitly selectable. Revalidate IDs after label corrections; fail clearly if source attachment becomes invalid.
3. **Define placement.** Implement `set_transform(source_id, matrix, instance_id=None)`. Accept finite orientation-preserving similarity transforms: positive uniform scale, rotation, translation; reject shear and singular matrices. The environment source transform is identity.
4. **Fix coordinates and pivots.** Use environment world coordinates and OpenCV world-to-camera cameras. For column vectors, `p_destination = T_source @ T_object @ p_source`. Object transforms act in source coordinates; default identity. Express a chosen source-space pivot explicitly as `Translate(pivot) @ RotateScale @ Translate(-pivot)`. Manual scale calibration supplies the matrix; automatic estimation is deferred.
5. **Expose render inputs.** Add an internal iterator yielding selected valid RGB samples and transformed world points per source/object. Selection intersects `input_mask`; transforms never alter child points, RGB, labels, or validity. Repeat static geometry at every requested target time; add no timeline API yet.

## Verification and completion

- Test independent ID namespaces, environment rules, alias resolution, invalid selections/layouts/transforms, transform order/pivot, and unchanged children after edits.
- Test known scale/rotation/translation on synthetic points and identity placement parity.
- Complete when one object from source A and environment from B yield correctly placed samples for [stage 2](COMPOSITE_RENDERING_AND_FUSION_PLAN.md).

Deferred: automatic placement, GUI, multiview association, dynamic sources, and generated-frame updates.
