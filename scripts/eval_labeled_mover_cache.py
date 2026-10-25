"""Score a labeled 3D cache on a synthetic moving box.

Rasterizes a back wall, a floor, and one box translating sideways, then
renders one held-out camera. Prints precision and recall of the motion
label, and checks that attaching the label does not drop points.

Run from the repository root:

    python -m scripts.eval_labeled_mover_cache
"""

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Labeled
from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import forward_warp

OUTPUT_DIR = ROOT / "outputs" / "labeled_mover_cache"
HEIGHT = 128
WIDTH = 128
FOCAL = 80.0
REL_THRESH = 0.05
NOISE_SCALE = 0.02
FRAME_INDEX = 1

WALL_COLOR = (0.55, 0.62, 0.75)
FLOOR_COLOR = (0.45, 0.38, 0.28)
BOX_COLOR = (0.85, 0.22, 0.15)
FLOOR_Y = 1.3
WALL_Z = 6.0


def intrinsics(frames):
    cameras = torch.zeros(frames, 3, 3)
    cameras[:, 0, 0] = FOCAL
    cameras[:, 1, 1] = FOCAL
    cameras[:, 0, 2] = (WIDTH - 1) / 2
    cameras[:, 1, 2] = (HEIGHT - 1) / 2
    cameras[:, 2, 2] = 1
    return cameras


def w2c_from_centers(centers):
    poses = torch.eye(4).view(1, 4, 4).repeat(centers.shape[0], 1, 1)
    poses[:, :3, 3] = -centers
    return poses


def _box_t(origin, dirs, box_min, box_max):
    near_t = torch.full(dirs.shape[:2], -1e9)
    far_t = torch.full(dirs.shape[:2], 1e9)
    for axis in range(3):
        direction = dirs[..., axis]
        parallel = direction.abs() < 1e-8
        outside = parallel & ((origin[axis] < box_min[axis]) | (origin[axis] > box_max[axis]))
        denom = torch.where(parallel, torch.ones_like(direction), direction)
        entry = (box_min[axis] - origin[axis]) / denom
        exit_ = (box_max[axis] - origin[axis]) / denom
        axis_near = torch.minimum(entry, exit_)
        axis_far = torch.maximum(entry, exit_)
        axis_near = torch.where(parallel, torch.full_like(axis_near, -1e9), axis_near)
        axis_far = torch.where(parallel, torch.full_like(axis_far, 1e9), axis_far)
        near_t = torch.maximum(near_t, axis_near)
        far_t = torch.minimum(far_t, axis_far)
        far_t = torch.where(outside, torch.full_like(far_t, -1.0), far_t)
    hit = (far_t >= near_t) & (near_t > 1e-4)
    return torch.where(hit, near_t, torch.full_like(near_t, 1e9))


def render_frame(center, box_min, box_max):
    """Z-depth, RGB in [-1, 1], and the box mask for one camera."""
    ys, xs = torch.meshgrid(
        torch.arange(HEIGHT, dtype=torch.float32),
        torch.arange(WIDTH, dtype=torch.float32),
        indexing="ij",
    )
    cx = (WIDTH - 1) / 2
    cy = (HEIGHT - 1) / 2
    dirs = torch.stack([(xs - cx) / FOCAL, (ys - cy) / FOCAL, torch.ones_like(xs)], dim=-1)
    wall_t = torch.full((HEIGHT, WIDTH), WALL_Z - float(center[2]))
    wall_t = torch.where(wall_t > 1e-4, wall_t, torch.full_like(wall_t, 1e9))
    floor_t = (FLOOR_Y - float(center[1])) / dirs[..., 1]
    floor_t = torch.where((dirs[..., 1] > 1e-6) & (floor_t > 1e-4), floor_t, torch.full_like(floor_t, 1e9))
    box_t = _box_t(center, dirs, box_min, box_max)
    depth = torch.minimum(torch.minimum(wall_t, floor_t), box_t)
    box = box_t <= depth
    floor = (~box) & (floor_t <= depth)
    color = torch.tensor(WALL_COLOR, dtype=torch.float32).view(3, 1, 1).expand(3, HEIGHT, WIDTH).clone()
    color = torch.where(floor.unsqueeze(0), torch.tensor(FLOOR_COLOR).view(3, 1, 1), color)
    color = torch.where(box.unsqueeze(0), torch.tensor(BOX_COLOR).view(3, 1, 1), color)
    return depth.unsqueeze(0), color * 2 - 1, box


def make_scene(centers, box_centers):
    depth = []
    image = []
    box = []
    for center, box_x in zip(centers, box_centers):
        box_min = torch.tensor([box_x - 0.3, 0.7, 3.0])
        box_max = torch.tensor([box_x + 0.3, FLOOR_Y, 3.45])
        frame_depth, frame_image, frame_box = render_frame(center, box_min, box_max)
        depth.append(frame_depth)
        image.append(frame_image)
        box.append(frame_box)
    return torch.stack(depth), torch.stack(image), torch.stack(box)


def add_depth_noise(depth, scale, generator):
    valid = depth > 0
    span = torch.quantile(depth[valid], 0.95) - torch.quantile(depth[valid], 0.05)
    sigma = scale * span
    noise = torch.randn(depth.shape, generator=generator, dtype=depth.dtype)
    noisy = depth.clone()
    noisy[valid] = (depth[valid] + noise[valid] * sigma).clamp(min=1e-3)
    return noisy, float(sigma)


def precision_recall(prediction, ground_truth):
    prediction = prediction.bool()
    ground_truth = ground_truth.bool()
    true_positive = int((prediction & ground_truth).sum())
    predicted = int(prediction.sum())
    actual = int(ground_truth.sum())
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    return precision, recall, true_positive, predicted, actual


def psnr(rendered, reference, both):
    if int(both.sum()) == 0:
        return float("nan")
    mse = (rendered[:, both] - reference[:, both]).square().mean()
    if float(mse) == 0:
        return float("inf")
    return float(10 * torch.log10(torch.tensor(4.0) / mse))


def save_views(prefix, rgb, coverage, label):
    rgb_u8 = ((rgb.detach().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    Image.fromarray(rgb_u8.permute(1, 2, 0).cpu().numpy()).save(OUTPUT_DIR / f"{prefix}rgb.png")
    coverage_u8 = (coverage.detach().clamp(0, 1) * 255).round().to(torch.uint8)
    Image.fromarray(coverage_u8.cpu().numpy()).save(OUTPUT_DIR / f"{prefix}coverage.png")
    shown = (rgb.detach().clamp(-1, 1) + 1) * 0.5
    weight = label.detach().clamp(0, 1)
    tint = torch.tensor([0.15, 0.95, 0.85], dtype=shown.dtype).view(3, 1, 1)
    overlay = shown * (1 - 0.75 * weight) + tint * (0.75 * weight)
    overlay_u8 = (overlay.clamp(0, 1) * 255).round().to(torch.uint8)
    Image.fromarray(overlay_u8.permute(1, 2, 0).cpu().numpy()).save(OUTPUT_DIR / f"{prefix}label_overlay.png")


def reference_warp(cache, frame_idx, target_w2c, target_intrinsics):
    image = cache.input_image[0, frame_idx, 0, 0]
    points = cache.input_points[0, frame_idx, 0, 0]
    frames = cache.input_image.shape[1]
    depth = cache.input_depth.reshape(frames, 1, HEIGHT, WIDTH)[frame_idx]
    valid = (torch.isfinite(depth) & (depth > 0)).to(dtype=image.dtype)
    rendered, covered, _, _ = forward_warp(
        image.unsqueeze(0),
        mask1=valid.unsqueeze(0),
        depth1=None,
        transformation1=None,
        transformation2=target_w2c.unsqueeze(0),
        intrinsic1=target_intrinsics.unsqueeze(0),
        intrinsic2=target_intrinsics.unsqueeze(0),
        is_image=True,
        world_points1=points.unsqueeze(0),
    )
    return rendered[0], covered[0, 0]


def evaluate(depth, image, box, w2c, cameras, target_w2c, target_k, prefix):
    cache = Cache3D_Labeled(
        input_image=image,
        input_depth=depth,
        input_w2c=w2c,
        input_intrinsics=cameras,
        input_format=["F", "C", "H", "W"],
        device="cpu",
        weight_dtype=torch.float32,
        rel_thresh=REL_THRESH,
    )
    prediction = cache.motion_label[0, :, 0, 0, 0] > 0.5
    precision, recall, true_positive, predicted, actual = precision_recall(prediction, box)
    stored_points = int((cache.input_points.abs().sum(-1) > 0).sum())
    valid_points = int((torch.isfinite(cache.input_depth) & (cache.input_depth > 0)).sum())

    rendered, coverage, splatted = cache.render_cache(
        target_w2c.view(1, 1, 4, 4),
        target_k.view(1, 1, 3, 3),
        start_frame_idx=FRAME_INDEX,
    )
    reference, reference_coverage = reference_warp(cache, FRAME_INDEX, target_w2c, target_k)
    labeled_coverage = int(coverage[0, 0, 0, 0].sum())
    full_coverage = int(reference_coverage.sum())
    both = (coverage[0, 0, 0, 0] > 0) & (reference_coverage > 0)
    score = psnr(rendered[0, 0, 0], reference, both)
    save_views(prefix, rendered[0, 0, 0], coverage[0, 0, 0, 0], splatted[0, 0, 0, 0])
    return {
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "predicted": predicted,
        "actual": actual,
        "stored_points": stored_points,
        "valid_points": valid_points,
        "labeled_coverage": labeled_coverage,
        "full_coverage": full_coverage,
        "psnr": score,
    }


def print_report(title, metrics):
    print(title)
    print(
        f"  precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}"
        f"  (tp={metrics['true_positive']} pred={metrics['predicted']} gt={metrics['actual']})"
    )
    print(
        f"  points labeled={metrics['stored_points']}  all-valid={metrics['valid_points']}"
        f"  match={metrics['stored_points'] == metrics['valid_points']}"
    )
    print(
        f"  coverage labeled={metrics['labeled_coverage']}  all-valid={metrics['full_coverage']}"
        f"  match={metrics['labeled_coverage'] == metrics['full_coverage']}"
    )
    print(f"  psnr {metrics['psnr']:.2f} dB")


def main():
    torch.manual_seed(0)
    centers = torch.tensor(
        [
            [-0.20, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ]
    )
    box_centers = [-1.5, -0.5, 0.5, 1.5]
    depth, image, box = make_scene(centers, box_centers)
    w2c = w2c_from_centers(centers)
    cameras = intrinsics(centers.shape[0])
    target_center = torch.tensor([0.08, 0.10, 0.0])
    target_w2c = w2c_from_centers(target_center.view(1, 3))[0]
    target_k = intrinsics(1)[0]
    if int(box.sum()) == 0 or not torch.isfinite(depth).all():
        raise RuntimeError("The synthetic scene did not rasterize a finite box.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = evaluate(depth, image, box, w2c, cameras, target_w2c, target_k, prefix="")
    print_report("clean depth", clean)
    if clean["stored_points"] != clean["valid_points"]:
        raise RuntimeError("The motion label changed the point-cloud size.")
    if clean["labeled_coverage"] != clean["full_coverage"]:
        raise RuntimeError("Labeled coverage dropped against the all-valid warp.")

    noisy, sigma = add_depth_noise(depth, NOISE_SCALE, torch.Generator().manual_seed(0))
    noise = evaluate(noisy, image, box, w2c, cameras, target_w2c, target_k, prefix="noise_")
    print_report(f"depth noise s={NOISE_SCALE:g}  sigma={sigma:.5f}", noise)
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
