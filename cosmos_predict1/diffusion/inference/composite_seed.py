"""CPU image-space seed construction, independent of adaptive warp rendering."""

from dataclasses import dataclass
from collections.abc import Mapping
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class SeedResult:
    rgb: torch.Tensor  # [1,3,1,H,W], float32 [-1,1], CPU
    diagnostics: dict
    metadata: dict


def _result(rgb, diagnostics, metadata):
    seed = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None, :, None]
    return SeedResult(seed, diagnostics, metadata)


def load_provided_seed(path, image_size):
    """An explicit, complete RGB override; never resize or infer an alpha matte."""
    from cosmos_predict1.diffusion.inference.instance_segmentation import file_sha256
    path = Path(path)
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != tuple(reversed(image_size)):
            raise ValueError('Provided seed must be RGB with exactly the scene output dimensions')
        rgb = np.asarray(image).astype(np.float32) / 127.5 - 1
    return _result(rgb, {}, dict(strategy='provided', path=str(path.resolve()),
                                sha256=file_sha256(path), completeness=1., geometry_confidence=None))


def prepare_initial_seed(scene, composite, *, strategy='rendered', provided_seed=None,
                         plane_depths=None, occlusion_policy='reject'):
    if strategy not in ('rendered', 'image_space', 'provided'):
        raise ValueError('Unknown seed strategy')
    if (provided_seed is not None) != (strategy == 'provided'):
        raise ValueError('A seed image is required only for provided seed strategy')
    if plane_depths and strategy != 'image_space':
        raise ValueError('Plane-depth overrides require image_space seed strategy')
    if strategy == 'provided':
        return load_provided_seed(provided_seed, scene.image_size)
    if strategy == 'image_space':
        poses, intrinsics = scene.cameras()
        return build_image_space_seed(scene, composite, poses[0, 0], intrinsics[0, 0],
                                      plane_depths=plane_depths, occlusion_policy=occlusion_policy)
    return None


def _array(tensor):
    return tensor.detach().cpu().double().numpy()


def _maps(homography, size):
    h, w = size
    y, x = np.mgrid[:h, :w]
    pixels = np.stack((x, y, np.ones_like(x)), -1).astype(np.float64)
    source = pixels @ np.linalg.inv(homography).T
    denominator = source[..., 2]
    if not np.isfinite(source).all() or np.any(np.abs(denominator) < 1e-12):
        raise ValueError('Seed projection is singular in the target view')
    return (source[..., :2] / denominator[..., None]).astype(np.float32), pixels


def _sample(image, xy, *, nearest=False, border=cv2.BORDER_CONSTANT):
    return cv2.remap(image.astype(np.float32), xy[..., 0], xy[..., 1],
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=border)


@torch.no_grad()
def build_image_space_seed(scene, composite, first_w2c, first_intrinsics, *,
                           plane_depths=None, occlusion_policy='reject', boundary_threshold=.05):
    """Paste calibrated textured planes over a same-camera environment RGB image.

    plane_depths maps 'source_id:canonical_id' to source-camera Z. Unknown overlap
    depth is rejected by default; 'allow' draws over it and records uncertainty.
    The builder never consumes adaptive RGB, coverage, or visibility winners.
    """
    if occlusion_policy not in ('reject', 'allow'):
        raise ValueError('Seed occlusion policy must be reject or allow')
    if not math.isfinite(boundary_threshold) or boundary_threshold <= 0:
        raise ValueError('Seed boundary threshold must be finite and positive')
    if plane_depths is not None and not isinstance(plane_depths, Mapping):
        raise ValueError('Seed plane depths must be a mapping of source_id:instance_id to depth')
    plane_depths = dict(plane_depths or {})
    if any(not isinstance(key, str) or isinstance(value, bool)
           or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for key, value in plane_depths.items()):
        raise ValueError('Invalid seed plane depth: keys must be strings and depths finite positive numbers')
    used_overrides = set()
    pose, k = _array(first_w2c), _array(first_intrinsics)
    env_id = composite.environment_source_id
    env = composite._source(env_id)
    env_pose, env_k = map(_array, composite._source_cameras[env_id])
    if not np.allclose(pose, env_pose, atol=1e-6, rtol=0):
        raise ValueError('Image-space seed requires the first camera to match the environment source pose; '
                         'choose rendered or provided seed for a different starting view')
    labels = env.input_labels.reshape(env.input_image.shape[-2:]).numpy()
    valid = env.input_mask.reshape(labels.shape).bool().numpy()
    if np.any(valid & ~np.isin(labels, list(composite.selections[env_id]))):
        raise ValueError('Image-space seed cannot restore removed or unresolved environment content')
    for (sid, _), transform in composite.object_transforms.items():
        if sid == env_id and not torch.allclose(transform, torch.eye(4, dtype=transform.dtype)):
            raise ValueError('Image-space seed does not support moved environment objects')
    xy, pixels = _maps(k @ np.linalg.inv(env_k), scene.image_size)
    eh, ew = labels.shape
    if np.any((xy[..., 0] < -.5001) | (xy[..., 0] > ew - .4999)
              | (xy[..., 1] < -.5001) | (xy[..., 1] > eh - .4999)):
        raise ValueError('Image-space seed target field of view extends beyond environment RGB')
    background = _array(env.input_image[0, 0, 0, 0]).transpose(1, 2, 0).astype(np.float32)
    canvas = _sample(background, xy, border=cv2.BORDER_REPLICATE)
    points = _array(env.input_points).reshape(eh, ew, 3)
    env_z = points @ env_pose[2, :3] + env_pose[2, 3]
    z = _sample(env_z, xy, nearest=True, border=cv2.BORDER_REPLICATE)
    depth_valid = _sample(valid, xy, nearest=True, border=cv2.BORDER_REPLICATE) > .5
    safe = np.where(valid, env_z, 0).astype(np.float32)
    high = cv2.dilate(safe, np.ones((3, 3), np.uint8))
    low = cv2.erode(safe, np.ones((3, 3), np.uint8))
    uncertain_source = ~valid | ((high - low) > boundary_threshold * np.maximum(low, 1e-8))
    uncertain_depth = _sample(uncertain_source, xy, nearest=True, border=cv2.BORDER_REPLICATE) > .5
    rays = pixels @ np.linalg.inv(k).T
    layers, records = [], []
    source_ids = sorted(composite.sources)
    for sid in source_ids:
        if sid == env_id:
            continue
        child = composite._source(sid)
        sh, sw = child.input_image.shape[-2:]
        source_pose, source_k = map(_array, composite._source_cameras[sid])
        source_rgb = _array(child.input_image[0, 0, 0, 0]).transpose(1, 2, 0).astype(np.float32)
        source_labels = child.input_labels.reshape(sh, sw).numpy()
        source_valid = child.input_mask.reshape(sh, sw).bool().numpy()
        source_points = _array(child.input_points).reshape(sh, sw, 3)
        source_z = source_points @ source_pose[2, :3] + source_pose[2, 3]
        for instance in sorted(composite.selections[sid]):
            mask = (source_labels == instance) & source_valid
            if not mask.any():
                continue
            key = f'{sid}:{instance}'
            depth = float(plane_depths.get(key, np.median(source_z[mask])))
            if key in plane_depths:
                used_overrides.add(key)
            if not math.isfinite(depth) or depth <= 0:
                raise ValueError(f'Invalid seed plane depth for {key}')
            transform = composite.source_transforms[sid] @ composite.object_transforms.get(
                (sid, instance), composite._identity())
            relative = pose @ _array(transform) @ np.linalg.inv(source_pose)
            # Source camera coordinates are depth * inv(K) @ [u,v,1].
            basis = relative[:3, :3] + relative[:3, 3:4] @ np.array([[0., 0., 1.]]) / depth
            homography = k @ basis @ np.linalg.inv(source_k)
            ys, xs = np.nonzero(mask)
            corners = np.array([[xs.min()-.5, ys.min()-.5, 1], [xs.max()+.5, ys.min()-.5, 1],
                                [xs.max()+.5, ys.max()+.5, 1], [xs.min()-.5, ys.max()+.5, 1]])
            target_corners = (corners @ np.linalg.inv(source_k).T * depth) @ relative[:3, :3].T + relative[:3, 3]
            if (not np.isfinite(homography).all() or np.linalg.cond(basis) > 1e10
                    or np.any(target_corners[:, 2] <= 1e-6)):
                raise ValueError(f'Degenerate or near-plane-crossing seed proxy for {key}')
            object_xy, _ = _maps(homography, scene.image_size)
            alpha = _sample(mask, object_xy)
            premultiplied = _sample(source_rgb * mask[..., None], object_xy)
            # Transform the source plane normal using inverse transpose (similarity includes scale).
            normal = np.linalg.inv(relative[:3, :3]).T[:, 2]
            center = relative[:3, :3] @ np.array([0., 0., depth]) + relative[:3, 3]
            denominator = rays @ normal
            proxy_z = np.divide(normal @ center, denominator, out=np.full_like(denominator, np.inf),
                                where=np.abs(denominator) > 1e-10)
            support = (alpha > 1e-6) & np.isfinite(proxy_z) & (proxy_z > 1e-6)
            uncertainty = support & uncertain_depth
            if uncertainty.any() and occlusion_policy == 'reject':
                raise ValueError(f'Uncertain environment occlusion for {key} at {int(uncertainty.sum())} pixels; '
                                 'inspect depth or explicitly choose allow/provided seed')
            closer = (proxy_z < z - composite.depth_tolerance)
            tied = np.abs(proxy_z - z) <= composite.depth_tolerance
            visible = support & (~depth_valid | closer | (tied & (sid < env_id)))
            alpha = np.where(visible, alpha, 0)
            premultiplied = np.where(visible[..., None], premultiplied, 0)
            layers.append((proxy_z, alpha, premultiplied, source_ids.index(sid), instance, uncertainty))
            projected = (target_corners @ k.T)
            projected = projected[:, :2] / projected[:, 2:]
            records.append(dict(source_id=sid, instance_id=instance, source_plane_depth=depth,
                                depth_method='override' if key in plane_depths else 'median_valid_depth',
                                projected_bounds=[*projected.min(0).tolist(), *projected.max(0).tolist()],
                                visible_pixels=int(visible.sum())))
    if set(plane_depths) != used_overrides:
        raise ValueError(f'Unused seed plane-depth overrides: {sorted(set(plane_depths) - used_overrides)}')
    ownership = np.full((*scene.image_size, 2), [source_ids.index(env_id), 0], dtype=np.int64)
    label_x = np.clip(np.floor(xy[..., 0]+.5).astype(np.int64), 0, ew-1)
    label_y = np.clip(np.floor(xy[..., 1]+.5).astype(np.int64), 0, eh-1)
    ownership[..., 1] = labels[label_y, label_x]
    uncertainty = np.zeros(scene.image_size, bool)
    proxy_depth = np.zeros(scene.image_size, np.float32)
    if layers:
        depths = np.stack([layer[0] for layer in layers])
        order = np.argsort(depths, axis=0, kind='stable')
        # Per-pixel far-to-near compositing; stable reversal gives smaller IDs tie priority.
        for level in range(len(layers)-1, -1, -1):
            for index, (layer_z, alpha, color, source_index, instance, unknown) in enumerate(layers):
                selected = order[level] == index
                a = np.where(selected, alpha, 0)
                canvas = color * selected[..., None] + canvas * (1-a[..., None])
                covered = a > 1e-6
                ownership[covered] = [source_index, instance]
                proxy_depth[covered] = layer_z[covered]
                uncertainty |= unknown
    diagnostics = dict(ownership=ownership, proxy_depth=proxy_depth, uncertainty=uncertainty,
                       observed=np.ones(scene.image_size, bool), synthesized=np.zeros(scene.image_size, bool),
                       environment_depth_valid=depth_valid)
    return _result(np.clip(canvas, -1, 1), diagnostics,
                   dict(strategy='image_space', completeness=1., geometry_confidence=None,
                        environment_pose_matches=True, environment_fov_covered=True,
                        source_ids=source_ids,
                        source_fingerprints={sid: composite.sources[sid].source_fingerprint for sid in source_ids},
                        planes=records, occlusion_policy=occlusion_policy,
                        boundary_threshold=boundary_threshold, uncertain_pixels=int(uncertainty.sum()),
                        observed_definition='RGB sampled from source images; not measured target-view geometry'))
