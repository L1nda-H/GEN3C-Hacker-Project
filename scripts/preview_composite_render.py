"""Render a synthetic placed object, foreground occluder, and moving camera.

Run: python -m scripts.preview_composite_render --output-dir outputs/composite_preview
No checkpoints or diffusion required. NPZ retains exact conditioning buffers and
int64 winner diagnostics; PNG provides an RGB/depth/coverage contact sheet.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base
from cosmos_predict1.diffusion.inference.cache3d_composite import Cache3D_Composite
from cosmos_predict1.diffusion.inference.instance_labeling import ObjectRegistry


def make_scene():
    height, width = 64, 96
    intrinsics = torch.tensor([[80., 0, 48], [0, 80, 32], [0, 0, 1]])
    environment = torch.zeros(3, height, width)
    environment[0] = -.7
    environment[1] = -.4
    environment[2] = .5
    environment[:, ::8, :] += .15
    environment[:, :, ::8] += .15
    depth = torch.full((height, width), 6.)
    # A green post in the destination environment must occlude part of the prop.
    depth[18:56, 50:56] = 2.5
    environment[:, 18:56, 50:56] = torch.tensor([-.8, .7, -.6])[:, None, None]

    def cache(source_id, rgb, z, k, labels):
        registry = ObjectRegistry(source_id)
        if (labels > 0).any():
            registry.allocate('Checkerboard prop')
        return Cache3D_Base(
            input_image=rgb[None], input_depth=z[None, None], input_w2c=torch.eye(4)[None],
            input_intrinsics=k[None], input_labels=labels[None, None], object_registry=registry, device='cpu',
        )

    room = cache('room', environment, depth, intrinsics, torch.zeros_like(depth, dtype=torch.int64))
    prop_rgb = torch.zeros(3, 24, 24)
    y, x = torch.meshgrid(torch.arange(24), torch.arange(24), indexing='ij')
    prop_rgb[0] = .9
    prop_rgb[1] = torch.where((x // 4 + y // 4) % 2 == 0, .6, -.5)
    prop_rgb[2] = -.8
    prop_k = intrinsics.clone()
    prop_k[:2, 2] = torch.tensor([12., 12.])
    prop = cache('prop', prop_rgb, torch.full((24, 24), 3.), prop_k, torch.ones(24, 24, dtype=torch.int64))
    scene = Cache3D_Composite(image_size=(height, width), point_chunk_size=1024)
    scene.add_source('room', room, is_environment=True, source_w2c=torch.eye(4), source_intrinsics=intrinsics)
    scene.add_source('prop', prop, source_w2c=torch.eye(4), source_intrinsics=prop_k)
    scene.set_selection('prop', [1])
    placement = torch.eye(4)
    placement[:3, 3] = torch.tensor([.225, .075, 0.])
    scene.set_transform('prop', placement)
    poses = torch.eye(4).repeat(1, 3, 1, 1)
    poses[0, :, 0, 3] = torch.tensor([0., -.15, -.3])
    return scene, poses, intrinsics.repeat(1, 3, 1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/composite_preview'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene, poses, intrinsics = make_scene()
    diagnostics = scene.render_diagnostics(poses, intrinsics, include_rgb=True)
    rgb, coverage = diagnostics['rgb'], diagnostics['coverage']
    depth = diagnostics['depth']
    np.savez_compressed(
        args.output_dir / 'buffers.npz', rgb=rgb.numpy(), coverage=coverage.numpy(), depth=depth.numpy(),
        winners=diagnostics['winners'].numpy(), source_ids=np.array(diagnostics['source_ids']),
        nearest_depth=diagnostics['nearest_depth'].numpy(), w2cs=poses.numpy(), intrinsics=intrinsics.numpy(),
        candidate_batch_size=scene.render_settings['candidate_batch_size'],
    )
    h, w = scene.image_size
    sheet = Image.new('RGB', (3 * w, 3 * (h + 16)), '#202020')
    draw = ImageDraw.Draw(sheet)
    for frame in range(3):
        visible = coverage[0, frame, 0, 0].numpy()
        image = ((rgb[0, frame, 0].permute(1, 2, 0).numpy() + 1) * 127.5).round().astype('uint8')
        z = (depth[0, frame, 0].numpy() / 6 * 255).clip(0, 255).astype('uint8')
        panels = [image, np.repeat(z[..., None], 3, -1), np.repeat((visible * 255).astype('uint8')[..., None], 3, -1)]
        for col, (label, panel) in enumerate(zip(['RGB', 'Depth (0-6)', 'Coverage'], panels)):
            draw.text((col * w + 2, frame * (h + 16)), f'{label} / {frame}', fill='white')
            sheet.paste(Image.fromarray(panel), (col * w, frame * (h + 16) + 16))
    path = args.output_dir / 'preview.png'
    sheet.resize((sheet.width * 3, sheet.height * 3), Image.Resampling.NEAREST).save(path)
    print(f'Saved {path} and {args.output_dir / "buffers.npz"}')


if __name__ == '__main__':
    main()
