"""Preview the initial seed on CPU without loading diffusion or SAM weights."""

import argparse
import json
from pathlib import Path

from cosmos_predict1.diffusion.inference.cache3d_composite import combine_visible_buffers
from cosmos_predict1.diffusion.inference.composite_scene import SceneSpec, build_composite_scene
from cosmos_predict1.diffusion.inference.composite_seed import build_image_space_seed
from cosmos_predict1.diffusion.inference.composite_review import save_seed, save_seed_result, save_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-spec', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed-plane-depths', type=json.loads)
    parser.add_argument('--seed-removal-mask', help='Source-resolution grayscale mask of already removed geometry')
    parser.add_argument('--seed-occlusion-policy', choices=('reject', 'allow'), default='reject')
    parser.add_argument('--compare-rendered', action='store_true')
    args = parser.parse_args()
    scene = SceneSpec.load(args.scene_spec)
    composite = build_composite_scene(scene)
    poses, intrinsics = scene.cameras()
    result = build_image_space_seed(scene, composite, poses[0, 0], intrinsics[0, 0],
                                    plane_depths=args.seed_plane_depths,
                                    occlusion_policy=args.seed_occlusion_policy, removal_mask=args.seed_removal_mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene.save(args.output_dir / 'scene.json')
    save_seed_result(args.output_dir, result)
    if args.compare_rendered:
        buffers = composite.render_conditioning(poses[:, :1], intrinsics[:, :1])
        rgb, coverage = combine_visible_buffers(buffers['rgb'], buffers['coverage'])
        directory = args.output_dir / 'cache_render_reference'
        directory.mkdir(exist_ok=True)
        save_seed(directory, rgb[:, 0, 0].unsqueeze(2))
        save_manifest(directory, dict(purpose='cache_render_reference_only', coverage=float(coverage.mean())))
    print(f'Saved seed preview to {args.output_dir}')


if __name__ == '__main__':
    main()
