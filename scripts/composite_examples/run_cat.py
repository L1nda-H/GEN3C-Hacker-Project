#!/usr/bin/env python3
"""Run the cat composite preparation and GEN3C generation pipeline."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, message):
        self.stream.write(message)
        self.log.write(message)
        return len(message)

    def flush(self):
        self.stream.flush()
        self.log.flush()


@contextmanager
def log_output(path):
    with path.open('w') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='Prepare sources and CPU previews')
    mode.add_argument('--generate-only', action='store_true', help='Generate from an already prepared scene')
    parser.add_argument('--output-dir', type=Path, default=os.environ.get('OUTPUT_DIR'),
                        help='Output directory; defaults to a new timestamped directory')
    args = parser.parse_args()
    args.import_depth_mode = os.environ.get('IMPORT_DEPTH_MODE', 'plane')
    if args.import_depth_mode not in ('plane', 'rgbd'):
        parser.error('IMPORT_DEPTH_MODE must be plane or rgbd')
    args.forward = float(os.environ.get('CAMERA_DOLLY_FORWARD', '0.15'))
    args.sideways = float(os.environ.get('CAMERA_DOLLY_SIDEWAYS', '0.20'))
    return args


def preflight(args):
    if not args.prepare_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for diffusion; use --prepare-only on CPU')
        checkpoint = Path(os.environ.get('CHECKPOINT_DIR', 'checkpoints')) / 'Gen3C-Cosmos-7B/model.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing {checkpoint}; see docs/GCP_COMPOSITE_INFERENCE.md')
        import megatron.core
        import transformer_engine.pytorch
    if not args.generate_only:
        for batch in ['batch_0002', 'batch_0007', 'batch_0008']:
            for name in ('rgb.mp4', 'depth.npz', 'mask.npz', 'camera.npz'):
                path = Path('datasets/gen3c_examples') / batch / name
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(path)
        if not Path('checkpoints/sam2.1_hiera_tiny.pt').is_file():
            raise FileNotFoundError('Missing checkpoints/sam2.1_hiera_tiny.pt; see docs/REAL_CACHE_MVP_EVALUATION.md')


EXAMPLE = 'cat'
PROMPT = 'A small orange cat sitting on the tabletop to the man’s right, on the left side of the image. The cat is in front of the man and behind the foreground computer monitor, which partially occludes the cat. Between his hands is a lit red candle with melted wax and a wax puddle at its base. His hands naturally surround the candle; the handheld camera has been removed.'


def prepare(args, root):
    from dataclasses import replace
    import torch
    from scripts import evaluate_real_cache_mvp as mvp
    from scripts.composite_examples.camera_path import dolly_path

    depth_mode, forward, sideways = args.import_depth_mode, args.forward, args.sideways
    arguments = [
        '--environment-batch', 'batch_0007', '--object-batch', 'batch_0002',
        '--box', '62', '43', '189', '173',
        '--target-base', '115', '216', '--object-height', '100',
        '--remove-environment-box',
        *os.environ.get('CAMERA_REMOVE_BOX', '139 164 186 226').split(),
        '--object-depth-multiplier', os.environ.get('CAT_DEPTH_MULTIPLIER', '0.95'),
    ]
    arguments += ['--trajectory', 'lateral', '--camera-travel', '0', '--output-dir', str(root / 'inputs')]
    object_batch = arguments[arguments.index('--object-batch') + 1]
    load_rgbd = mvp.load_source
    export_previews = mvp.export_previews

    def load_source(batch):
        prepared = load_rgbd(batch)
        if batch == object_batch and depth_mode == 'plane':
            # A textured plane preserves the imported first-frame silhouette.
            # Keep destination depth intact for monitor/bed occlusion. Persist this
            # preparation before labeling, so fingerprints describe actual geometry.
            obs = prepared.observations[mvp.KEY]
            depth = obs.depth[obs.geometry_mask].median()
            observation = replace(obs, depth=torch.full_like(obs.depth, float(depth)))
            prepared = replace(prepared, observations={mvp.KEY: observation},
                               preparation=dict(prepared.preparation,
                                                imported_depth_mode='plane', plane_depth=float(depth)))
            print(f'Imported {batch}: planar depth proxy at {float(depth):.6g}')
        return prepared

    @torch.inference_mode()
    def replace_camera(scene, root):
        import os
        import cv2
        import numpy as np
        from PIL import Image
        from cosmos_predict1.diffusion.inference.composite_scene import load_prepared_source
        from cosmos_predict1.diffusion.inference.instance_labeling import load_labeled_source
        environment_source = next(s for s in scene.sources if s.source_id == scene.environment_source)
        prepared = load_prepared_source(environment_source.prepared_path, environment_source.fingerprint)
        labeled = load_labeled_source(environment_source.labels_path, prepared)
        env = prepared.observations[mvp.KEY]
        labels = labeled.labels[mvp.KEY]
        removed = (labels > 0) & env.geometry_mask
        if not removed.any():
            raise ValueError('No handheld camera mask; cannot place replacement candle')
        base_pixel, candle_height = (162, 223), 60.
        # The camera obscures the contact point. Fit the visible tabletop just below
        # it, then intersect the base ray with that plane (inverse depth is affine).
        table_y, table_x = torch.meshgrid(torch.arange(229, 238), torch.arange(140, 176), indexing='ij')
        table_valid = (labels[table_y, table_x] == 0) & env.geometry_mask[table_y, table_x]
        if int(table_valid.sum()) < 20:
            raise ValueError('Insufficient visible tabletop depth for candle contact')
        design = torch.stack((table_x.float() - base_pixel[0],
                              table_y.float() - base_pixel[1], torch.ones_like(table_x).float()), dim=-1)
        coefficients = torch.linalg.lstsq(design[table_valid],
                                          env.depth[table_y, table_x][table_valid].reciprocal()).solution
        target_depth = float(coefficients[2].reciprocal())
        if not np.isfinite(target_depth) or target_depth <= 0:
            raise ValueError('Invalid tabletop contact depth')
        sam = mvp.SAM21Segmenter.from_checkpoint('checkpoints/sam2.1_hiera_tiny.pt',
                                                model_size='tiny', device='cpu')
        candle = mvp.load_source('batch_0008')
        candle_obs = candle.observations[mvp.KEY]
        # Include flame, candle body and wax puddle, including its wide left/right tips.
        prompts = [mvp.SegmentPrompts(
            1, box=torch.tensor([float(v) for v in os.environ['CANDLE_BOX'].split()]),
            points=torch.tensor([[202., 181.], [183., 210.], [231., 210.], [204., 132.]]),
            point_labels=torch.ones(4, dtype=torch.int64), name='Candle with flame and wax puddle')]
        result = mvp.segment_source(candle, sam, rgb_range=(-1, 1), prompts={mvp.KEY: prompts})
        candle_labels = mvp.label_source(result, source_id='candle_original')
        mvp.save_prepared_source(root / 'candle_original.npz', candle)
        mvp.save_labeled_source(root / 'candle_original_labels', candle_labels)
        overlay = mvp.label_overlays(candle_labels, rgb_range=(-1, 1))[mvp.KEY]
        Image.fromarray(overlay.permute(1, 2, 0).numpy()).save(root / 'candle_original_labels.png')
        mask = candle_labels.labels[mvp.KEY] > 0
        if not mask.any() or not mask[205:218, 175:235].any():
            raise ValueError('Candle segmentation did not include its base wax')
        ys, xs = torch.where(mask)
        # A planar copy preserves flame and puddle silhouette; original RGB-D/masks
        # remain saved above. Its lowest selected pixels touch the fitted tabletop.
        source_depth = float(candle_obs.depth[mask].median())
        rays = torch.linalg.solve(candle_obs.intrinsics,
                                 torch.stack((xs.float(), ys.float(), torch.ones_like(xs).float())))
        points = rays.T * source_depth
        placement, placed_depth = mvp.object_placement(points, ys, env, base_pixel, candle_height)
        placement[:3, :] *= target_depth / placed_depth
        old_pivot = points[ys >= ys.max() - 2].median(dim=0).values
        base_pivot = points[ys == ys.max()].median(dim=0).values
        placement[:3, 3] += placement[:3, :3] @ (old_pivot - base_pivot)
        plane = placement[:3, :3].clone()
        plane[:, 2] += placement[:3, 3] / source_depth
        homography = env.intrinsics @ plane @ torch.linalg.inv(candle_obs.intrinsics)
        h, w = env.depth.shape
        kwargs = dict(dsize=(w, h), flags=cv2.INTER_NEAREST)
        projected_mask = cv2.warpPerspective(mask.byte().numpy(), homography.numpy(), **kwargs).astype(bool)
        rgb = cv2.warpPerspective(candle_obs.rgb.permute(1, 2, 0).numpy(), homography.numpy(), **kwargs)
        retained = (labels == 0) & env.geometry_mask
        visible = torch.from_numpy(projected_mask) & (~retained | (target_depth < env.depth))
        if not visible.any():
            raise ValueError('Replacement candle is completely occluded')
        new_rgb, new_depth = env.rgb.clone(), env.depth.clone()
        new_rgb[:, visible] = torch.from_numpy(rgb).permute(2, 0, 1)[:, visible]
        new_depth[visible] = target_depth
        validity = retained | visible
        updated = replace(prepared, observations={mvp.KEY: replace(
            env, rgb=new_rgb, depth=new_depth, geometry_mask=validity)},
            preparation=dict(prepared.preparation, replacement='batch_0008 candle and wax',
                             camera_removal_labels=str(root / 'environment_labels'),
                             candle_source=str(root / 'candle_original.npz')))
        segmentation = mvp.segment_source(updated, sam, rgb_range=(-1, 1),
                                          prompts={mvp.KEY: []}, environment_masks={mvp.KEY: validity})
        updated_labels = mvp.label_source(segmentation, source_id=environment_source.source_id)
        mvp.save_prepared_source(root / 'environment_with_candle.npz', updated)
        mvp.save_labeled_source(root / 'environment_with_candle_labels', updated_labels)
        environment_source.prepared_path = root / 'environment_with_candle.npz'
        environment_source.labels_path = root / 'environment_with_candle_labels'
        environment_source.fingerprint = updated_labels.source_fingerprint
        environment_source.selected_ids = [0]
        Image.fromarray((visible.numpy() * 255).astype(np.uint8)).save(root / 'candle_visible_mask.png')
        Image.fromarray(((removed & ~visible).numpy() * 255).astype(np.uint8)).save(root / 'seed_removal_mask.png')
        mvp.rgb_image(new_rgb.masked_fill(~validity[None], -1)).save(root / 'environment_with_candle.png')
        (root / 'camera_replacement.json').write_text(json.dumps(dict(
            source_batch='batch_0008', target_base=list(base_pixel), height=candle_height,
            contact_surface='Plane fitted to visible tabletop below the removed camera',
            target_depth=target_depth, placement=placement.tolist(),
            removed_camera_pixels=int(removed.sum()), visible_candle_pixels=int(visible.sum()),
            conditioning='Candle baked into environment RGB-D; cat retains second buffer',
            holes='Uncovered removed-camera pixels remain invalid for diffusion completion',
        ), indent=2) + '\n')
        scene.validate()
        return scene


    def export_dolly(cache, poses, intrinsics, root):
        scene = mvp.SceneSpec.load(root / 'scene.json')
        scene = replace_camera(scene, root)
        cache = mvp.build_composite_scene(scene, device='cpu')
        poses = dolly_path(scene, poses.shape[1], forward, sideways)
        scene.target_w2cs = poses
        scene.save(root / 'scene.json')
        (root / 'camera_path.json').write_text(json.dumps(
            dict(kind='dolly', forward=forward, sideways=sideways), indent=2) + '\n')
        print(f'Camera dolly: forward={forward:g}, sideways={sideways:g}')
        export_previews(cache, poses, intrinsics, root)

    mvp.load_source = load_source
    mvp.export_previews = export_dolly
    mvp.main(arguments)

    # Convert the same path to 121 cameras at output resolution.
    import torch
    from scripts.composite_examples.camera_path import dolly_path
    from cosmos_predict1.diffusion.inference.composite_scene import SceneSpec

    root = root / 'inputs'
    scene = SceneSpec.load(root / 'scene.json')
    poses, intrinsics = scene.cameras()
    settings = json.loads((root / 'camera_path.json').read_text())
    target = dolly_path(scene, 121, settings['forward'], settings['sideways'])
    h, w = scene.image_size
    k = intrinsics[:, :1].repeat(1, 121, 1, 1)
    for row, ratio in ((0, 1280 / w), (1, 704 / h)):
        k[..., row, :] *= ratio
        k[..., row, 2] += (ratio - 1) / 2
    scene.target_w2cs = target
    scene.target_intrinsics = k
    scene.image_size = (704, 1280)
    scene.save(root / 'scene_121.json')
    print(f'Saved {root / "scene_121.json"}')

def generate(root):
    scene_path = root / 'inputs/scene_121.json'
    if not scene_path.is_file():
        raise FileNotFoundError(f'Missing {scene_path}; prepare this example first')
    prompt = PROMPT
    prompt += ' The camera dollies forward and sideways toward the scene center, with visible depth parallax, following the supplied camera trajectory.'
    command = [
        sys.executable, '-m', 'cosmos_predict1.diffusion.inference.gen3c_composite',
        '--scene_spec', str(scene_path),
        '--checkpoint_dir', os.environ.get('CHECKPOINT_DIR', 'checkpoints'),
        '--prompt', prompt, '--num_video_frames', '121', '--height', '704', '--width', '1280',
        '--fps', '24', '--guidance', os.environ.get('GUIDANCE', '1'),
        '--seed', os.environ.get('SEED', '1'), '--num_steps', os.environ.get('NUM_STEPS', '35'),
        '--seed_occlusion_policy', os.environ.get('SEED_OCCLUSION_POLICY', 'reject'),
        '--offload_diffusion_transformer', '--offload_tokenizer',
        '--offload_text_encoder_model', '--offload_guardrail_models', '--disable_guardrail',
        '--video_save_folder', str(root), '--video_save_name', EXAMPLE,
    ]
    removal_mask = root / 'inputs/seed_removal_mask.png'
    if removal_mask.is_file():
        command += ['--seed_removal_mask', str(removal_mask)]
    with (root / 'inference.log').open('w') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
    manifest = json.loads((root / f'{EXAMPLE}_review/generation.json').read_text())
    if manifest.get('status') != 'saved' or not Path(manifest.get('final_video', '')).is_file():
        raise RuntimeError(f'Generation did not save a video: {manifest.get("status")}')


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    os.environ['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    if os.environ.get('CONDA_PREFIX'):
        os.environ.setdefault('CUDA_HOME', os.environ['CONDA_PREFIX'])
    os.environ.setdefault('CANDLE_BOX', '163 94 240 218')
    preflight(args)
    output_dir = (args.output_dir or Path('outputs') / f'{EXAMPLE}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    root = output_dir / EXAMPLE
    root.mkdir(parents=True, exist_ok=True)
    print(f'Outputs: {output_dir}', flush=True)
    if not args.generate_only:
        with log_output(root / 'prepare.log'):
            prepare(args, root)
    if not args.prepare_only:
        generate(root)
    print(f'Completed {EXAMPLE}. Outputs: {output_dir}')


if __name__ == '__main__':
    main()
