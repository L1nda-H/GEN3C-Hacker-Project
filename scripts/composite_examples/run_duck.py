#!/usr/bin/env python3
"""Run the duck composite preparation and GEN3C generation pipeline."""

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
        for batch in ['batch_0000', 'batch_0001']:
            for name in ('rgb.mp4', 'depth.npz', 'mask.npz', 'camera.npz'):
                path = Path('datasets/gen3c_examples') / batch / name
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(path)
        if not Path('checkpoints/sam2.1_hiera_tiny.pt').is_file():
            raise FileNotFoundError('Missing checkpoints/sam2.1_hiera_tiny.pt; see docs/REAL_CACHE_MVP_EVALUATION.md')


EXAMPLE = 'duck'
PROMPT = 'A giant yellow rubber duck sitting on the ocean surface in a seaside town harbor. Its red feet touch the water, with a natural water contact and small ripples around its base. The entire duck including its feet is about half the image height.'


def prepare(args, root):
    from dataclasses import replace
    import torch
    from scripts import evaluate_real_cache_mvp as mvp
    from scripts.composite_examples.camera_path import dolly_path

    depth_mode, forward, sideways = args.import_depth_mode, args.forward, args.sideways
    arguments = [
        '--environment-batch', 'batch_0001', '--object-batch', 'batch_0000',
        '--box', '136', '34', '224', '206',
        '--target-base', '160', '204', '--object-height', '120',
    ]
    arguments += ['--trajectory', 'lateral', '--camera-travel', '0', '--output-dir', str(root / 'inputs')]
    object_batch = arguments[arguments.index('--object-batch') + 1]
    load_rgbd = mvp.load_source
    export_previews = mvp.export_previews
    object_placement = mvp.object_placement

    def place_on_water(points, source_rows, environment, base_pixel, height):
        placement, old_depth = object_placement(points, source_rows, environment, base_pixel, height)
        u, v = base_pixel
        # Snap to the measured water point instead of the MVP's foreground offset.
        water_depth = float(environment.depth[v, u])
        placement[:3, :] *= water_depth / old_depth
        old_pivot = points[source_rows >= source_rows.max() - 2].median(dim=0).values
        bottom_pivot = points[source_rows == source_rows.max()].median(dim=0).values
        placement[:3, 3] += placement[:3, :3] @ (old_pivot - bottom_pivot)
        water_point = torch.linalg.solve(environment.intrinsics,
                                         torch.tensor([float(u), float(v), 1.])) * water_depth
        contact = placement[:3, :3] @ bottom_pivot + placement[:3, 3]
        if not torch.allclose(contact, water_point, atol=1e-5):
            raise ValueError('Duck base did not align with the water surface')
        print(f'Duck water contact: pixel=({u}, {v}), depth={water_depth:.6g}')
        return placement, water_depth

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

    def export_dolly(cache, poses, intrinsics, root):
        scene = mvp.SceneSpec.load(root / 'scene.json')
        poses = dolly_path(scene, poses.shape[1], forward, sideways)
        scene.target_w2cs = poses
        scene.save(root / 'scene.json')
        (root / 'camera_path.json').write_text(json.dumps(
            dict(kind='dolly', forward=forward, sideways=sideways), indent=2) + '\n')
        print(f'Camera dolly: forward={forward:g}, sideways={sideways:g}')
        export_previews(cache, poses, intrinsics, root)

    mvp.load_source = load_source
    mvp.export_previews = export_dolly
    mvp.object_placement = place_on_water
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
    pass
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
