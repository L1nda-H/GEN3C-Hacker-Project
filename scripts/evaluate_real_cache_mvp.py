"""Preview real RGB-D inputs, then segment, compose, and render on CPU.

Run from the repository root:
    python -m scripts.evaluate_real_cache_mvp
    python -m scripts.evaluate_real_cache_mvp --box X1 Y1 X2 Y2

See docs/REAL_CACHE_MVP_EVALUATION.md for setup and downloads.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from cosmos_predict1.diffusion.inference.composite_review import save_render, save_seed
from cosmos_predict1.diffusion.inference.composite_scene import (
    SceneSource,
    SceneSpec,
    build_composite_scene,
    save_prepared_source,
)
from cosmos_predict1.diffusion.inference.instance_labeling import (
    label_overlays,
    label_source,
    save_labeled_source,
)
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    SAM21Segmenter,
    SegmentPrompts,
    SourceObservation,
    prepare_source,
    segment_source,
)


KEY = (0, 0)
IMAGE_SIZE = (240, 320)


def load_source(batch):
    folder = Path("datasets/gen3c_examples") / batch
    capture = cv2.VideoCapture(str(folder / "rgb.mp4"))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {folder / 'rgb.mp4'}")
    rgb = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    rgb = rgb.permute(2, 0, 1).float() / 127.5 - 1
    with np.load(folder / "depth.npz") as data:
        depth = torch.from_numpy(data["depth"][0].copy()).float()
    with np.load(folder / "mask.npz") as data:
        valid = torch.from_numpy(data["mask"][0].copy()).bool()
    with np.load(folder / "camera.npz") as data:
        intrinsics = torch.from_numpy(data["intrinsics"][0].copy()).float()
    # Independent single-image sources use their camera as their local origin.
    observation = SourceObservation(
        rgb, depth, intrinsics, torch.eye(4), geometry_mask=valid,
    )
    return prepare_source({KEY: observation}, output_size=IMAGE_SIZE)


def rgb_image(rgb):
    pixels = ((rgb.permute(1, 2, 0) + 1) * 127.5).round().byte().numpy()
    return Image.fromarray(pixels)


def object_placement(points, source_rows, environment, base_pixel, height):
    """Anchor the candle's base on the foreground rock and size it in pixels."""
    u, v = base_pixel
    k = environment.intrinsics
    # Use the selected object's bottom surface as the pivot, rather than its center.
    anchor = points[source_rows >= source_rows.max() - 2].median(dim=0).values
    # Keep the base just in front of the visible support surface.
    target_z = float(environment.depth[v, u]) * 0.98
    target = torch.linalg.solve(k, torch.tensor([float(u), float(v), 1.])) * target_z
    relative = points - anchor

    # Account for both cameras' focal lengths and the object's depth variation.
    # The upper bound keeps every selected point in front of the camera.
    lower = 0.
    upper = 0.95 * target_z / max(float(-relative[:, 2].min()), 1e-6)
    for _ in range(32):
        scale = (lower + upper) / 2
        projected = (relative * scale + target) @ k.T
        rows = projected[:, 1] / projected[:, 2]
        if float(rows.max() - rows.min()) < height:
            lower = scale
        else:
            upper = scale
    scale = (lower + upper) / 2
    placement = torch.eye(4)
    placement[:3, :3] *= scale
    placement[:3, 3] = target - scale * anchor
    return placement, target_z


@torch.inference_mode()
def export_previews(cache, poses, intrinsics, root):
    """Preview every conditioning buffer; joint RGB is a geometric reference only."""
    root.mkdir(parents=True, exist_ok=True)
    source_ids = cache.conditioning_source_ids
    frames = []
    buffer_frames = [[] for _ in source_ids]
    height, width = cache.image_size
    for i in range(poses.shape[1]):
        rendered = cache.render_conditioning(
            poses[:, i:i + 1], intrinsics[:, i:i + 1], start_frame_idx=i, include_depth=True,
        )
        save_render(root, i, rendered)
        # This joint image is not sent as a conditioning buffer to diffusion.
        joint, _ = cache.render_cache(poses[:, i:i + 1], intrinsics[:, i:i + 1], start_frame_idx=i)
        joint_image = rgb_image(joint[0, 0, 0])
        joint_image.save(root / f"joint_reference_{i:06d}.png")
        if i == 0:
            save_seed(root, joint[:, 0, 0].unsqueeze(2))
        sheet = Image.new("RGB", ((len(source_ids) + 1) * width, height + 24), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, source_id in enumerate(source_ids):
            image = rgb_image(rendered["rgb"][0, 0, index])
            buffer_frames[index].append(image)
            draw.text((index * width + 4, 4), f"Buffer {index}: {source_id}", fill="white")
            sheet.paste(image, (index * width, 24))
        draw.text((len(source_ids) * width + 4, 4), "Joint reference (seed at frame 0)", fill="white")
        sheet.paste(joint_image, (len(source_ids) * width, 24))
        frames.append(sheet)
    frames[0].save(root / "preview.png")
    frames[0].save(root / "preview.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    for index, images in enumerate(buffer_frames):
        images[0].save(root / f"buffer_{index:03d}.gif", save_all=True,
                       append_images=images[1:], duration=100, loop=0)
    metadata = dict(conditioning_source_ids=source_ids, conditioning_strategy="per_source_latent_concat",
                    num_frames=len(frames), image_size=cache.image_size,
                    joint_reference="Geometric preview only; frame 0 supplies the image-conditioning seed")
    (root / "preview.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved per-source buffers, seed, and preview.gif to {root}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    inputs.add_argument("--scene-spec", type=Path, help="Render an existing scene JSON without rerunning SAM")
    parser.add_argument("--target-base", nargs=2, type=int, default=(275, 225), metavar=("U", "V"),
                        help="Destination base pixel; default places the candle on the foreground rock")
    parser.add_argument("--object-height", type=float, default=80.,
                        help="Rendered object height in pixels (default: 80)")
    parser.add_argument("--trajectory", choices=("lateral", "forward"), default="lateral",
                        help="Move sideways or forward toward the creature (default: lateral)")
    parser.add_argument("--camera-travel", type=float, default=0.006,
                        help="Camera travel as a fraction of placement depth (default: 0.006)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_cache_mvp"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.scene_spec is not None:
        scene = SceneSpec.load(args.scene_spec)
        root.mkdir(parents=True, exist_ok=True)
        scene.save(root / "scene.json")
        export_previews(build_composite_scene(scene, device="cpu"), *scene.cameras(), root)
        return
    if not (0 <= args.target_base[0] < IMAGE_SIZE[1] and 0 <= args.target_base[1] < IMAGE_SIZE[0]):
        parser.error("--target-base must be inside the 320 x 240 destination image")
    if args.object_height <= 0 or args.camera_travel < 0:
        parser.error("--object-height must be positive and --camera-travel nonnegative")
    root.mkdir(parents=True, exist_ok=True)

    sources = {
        "environment": load_source("batch_0004"),
        "object": load_source("batch_0008"),
    }
    for name, prepared in sources.items():
        rgb_image(prepared.observations[KEY].rgb).save(root / f"{name}_input.png")
    if args.box is None:
        print(f"Inspect {root / 'object_input.png'}; rerun with --box X1 Y1 X2 Y2")
        return

    sam = SAM21Segmenter.from_checkpoint(
        "checkpoints/sam2.1_hiera_tiny.pt", model_size="tiny", device="cpu",
    )
    labeled = {}
    for name, prepared in sources.items():
        # The destination image is explicitly treated as the fixed environment.
        if name == "environment":
            options = dict(
                prompts={KEY: []},
                environment_masks={KEY: torch.ones(IMAGE_SIZE, dtype=torch.bool)},
            )
        else:
            options = dict(prompts={KEY: [
                SegmentPrompts(1, box=torch.tensor(args.box), name="Selected object"),
            ]})
        segmented = segment_source(prepared, sam, rgb_range=(-1, 1), **options)
        labeled[name] = label_source(segmented, source_id=name)
        save_prepared_source(root / f"{name}.npz", prepared)
        save_labeled_source(root / f"{name}_labels", labeled[name])
        overlay = label_overlays(labeled[name], rgb_range=(-1, 1))[KEY]
        Image.fromarray(overlay.permute(1, 2, 0).numpy()).save(root / f"{name}_labels.png")

    object_ids = list(labeled["object"].registry.names)
    if not object_ids:
        raise ValueError("No object mask returned; choose another box")
    object_id = object_ids[0]
    obs = sources["object"].observations[KEY]
    ys, xs = torch.where(labeled["object"].labels[KEY] == object_id)
    if not len(xs):
        raise ValueError("No valid object pixels; choose another box")
    z = obs.depth[ys, xs]
    pixels = torch.stack((xs.float(), ys.float(), torch.ones_like(z)))
    points = (torch.linalg.inv(obs.intrinsics) @ pixels).T * z[:, None]
    environment = sources["environment"].observations[KEY]
    u, v = args.target_base
    if not environment.geometry_mask[v, u]:
        raise ValueError("Choose a --target-base pixel with valid environment depth")
    placement, target_z = object_placement(
        points, ys, environment, args.target_base, args.object_height,
    )

    poses = torch.eye(4).repeat(1, 24, 1, 1)
    # OpenCV world-to-camera translation is negative as the camera moves forward.
    travel_axis = 2 if args.trajectory == "forward" else 0
    poses[0, :, travel_axis, 3] = torch.linspace(0, -args.camera_travel * target_z, 24)
    intrinsics = environment.intrinsics.repeat(1, 24, 1, 1)
    scene = SceneSpec(
        sources=[
            SceneSource(
                name, root / f"{name}.npz", root / f"{name}_labels",
                result.source_fingerprint,
                [0] if name == "environment" else [object_id],
                source_transform=torch.eye(4) if name == "environment" else placement,
            )
            for name, result in labeled.items()
        ],
        environment_source="environment",
        target_w2cs=poses,
        target_intrinsics=intrinsics,
        image_size=IMAGE_SIZE,
    )
    scene.save(root / "scene.json")
    cache = build_composite_scene(scene, device="cpu")

    export_previews(cache, poses, intrinsics, root)


if __name__ == "__main__":
    main()
