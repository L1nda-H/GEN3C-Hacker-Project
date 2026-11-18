"""Build a translating camera path looking toward the original scene center."""

import math

import torch

from cosmos_predict1.diffusion.inference.composite_scene import load_prepared_source
from cosmos_predict1.diffusion.inference.instance_labeling import load_labeled_source


def dolly_path(scene, num_frames, forward=0.15, sideways=0.20):
    """Move forward/right, looking at a fixed center point in source-camera space.

    Forward travel is a fraction of nearby selected geometry depth. Sideways
    travel is a fraction of the view width at that depth, adapting to focal length.
    """
    if not math.isfinite(forward) or not 0 <= forward <= 0.5:
        raise ValueError('CAMERA_DOLLY_FORWARD must be between 0 and 0.5')
    if not math.isfinite(sideways) or not -0.5 <= sideways <= 0.5:
        raise ValueError('CAMERA_DOLLY_SIDEWAYS must be between -0.5 and 0.5')
    near_depths = []
    for source in scene.sources:
        prepared = load_prepared_source(source.prepared_path, source.fingerprint)
        obs = next(iter(prepared.observations.values()))
        labels = next(iter(load_labeled_source(source.labels_path, prepared).labels.values()))
        selected = obs.geometry_mask & torch.isin(labels, torch.tensor(source.selected_ids))
        ys, xs = torch.where(selected)
        if not len(xs):
            continue
        z = obs.depth[ys, xs]
        rays = torch.linalg.solve(obs.intrinsics, torch.stack((xs.float(), ys.float(), torch.ones_like(z))))
        points = rays.T * z[:, None]
        transform = torch.as_tensor(source.source_transform, dtype=points.dtype)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        positive = points[:, 2][points[:, 2] > 0]
        if positive.numel():
            near_depths.append(float(torch.quantile(positive, 0.05)))
        if source.source_id == scene.environment_source:
            h, w = obs.depth.shape
            central = selected[h // 3:2 * h // 3, w // 3:2 * w // 3]
            depths = obs.depth[h // 3:2 * h // 3, w // 3:2 * w // 3][central]
            focus_depth = float(depths.median() if depths.numel() else z.median())
    if not near_depths:
        raise ValueError('No selected geometry for camera path')
    travel_depth = min(near_depths)
    _, intrinsics = scene.cameras()
    centers = torch.zeros(num_frames, 3)
    progress = torch.linspace(0, 1, num_frames)
    centers[:, 0] = progress * sideways * scene.image_size[1] / intrinsics[0, 0, 0, 0] * travel_depth
    centers[:, 2] = progress * forward * travel_depth
    yaw = torch.atan2(-centers[:, 0], focus_depth - centers[:, 2])
    poses = torch.eye(4).repeat(1, num_frames, 1, 1)
    poses[0, :, 0, 0] = poses[0, :, 2, 2] = yaw.cos()
    poses[0, :, 0, 2] = -yaw.sin()
    poses[0, :, 2, 0] = yaw.sin()
    poses[0, :, :3, 3] = -(poses[0, :, :3, :3] @ centers[..., None]).squeeze(-1)
    return poses
