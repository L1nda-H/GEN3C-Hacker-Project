"""CLI for static object/environment composition using pretrained GEN3C."""

import argparse
import json
from pathlib import Path
import subprocess

import torch

from cosmos_predict1.diffusion.inference.composite_scene import SceneSpec
from cosmos_predict1.diffusion.inference.gen3c_composite_pipeline import Gen3cCompositePipeline


def create_parser():
    # Defer heavyweight diffusion dependencies so scene tooling remains CPU usable.
    from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--scene_spec', required=True, help='SceneSpec JSON')
    parser.add_argument('--min_seed_coverage', type=float, default=0.5)
    parser.add_argument('--checkpoint_revision', help='Immutable checkpoint revision; otherwise hash model.pt')
    parser.add_argument('--prompt_upsampler_dir', default='Pixtral-12B')
    return parser


def demo(args, scene_spec=None):
    if scene_spec is not None and getattr(args, 'scene_spec', None):
        raise ValueError('Supply a Python SceneSpec or --scene_spec, not both')
    if scene_spec is None and not getattr(args, 'scene_spec', None):
        raise ValueError('A SceneSpec or --scene_spec is required')
    scene = scene_spec if scene_spec is not None else SceneSpec.load(args.scene_spec)
    scene.validate()
    if any(getattr(args, key, None) is not None for key in ('npz_path', 'labels_path', 'selected_ids')):
        raise ValueError('Composite generation does not accept legacy source inputs')
    if tuple(scene.image_size) != (args.height, args.width):
        raise ValueError('Scene target dimensions must match --height/--width')
    if (args.num_video_frames < 121 or (args.num_video_frames - 1) % 120
            or args.num_video_frames > scene.cameras()[0].shape[1]):
        raise ValueError('Trajectory must cover num_video_frames = 120*N+1, N >= 1')
    if not 0 < args.min_seed_coverage <= 1:
        raise ValueError('min_seed_coverage must be in (0,1]')
    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
    from cosmos_predict1.utils import misc
    from cosmos_predict1.utils.model_config import model_config_session

    misc.set_random_seed(args.seed)
    distributed_started = False
    try:
        if args.num_gpus > 1:
            from megatron.core import parallel_state
            from cosmos_predict1.utils import distributed
            distributed.init()
            distributed_started = True
            parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        with model_config_session(sam_config="configs/sam2.1/sam2.1_hiera_l.yaml"):
            pipeline = Gen3cPipeline(
                inference_type='video2world', checkpoint_dir=args.checkpoint_dir,
                checkpoint_name='Gen3C-Cosmos-7B', prompt_upsampler_dir=args.prompt_upsampler_dir,
                enable_prompt_upsampler=not args.disable_prompt_upsampler,
                offload_network=args.offload_diffusion_transformer,
                offload_tokenizer=args.offload_tokenizer,
                offload_text_encoder_model=args.offload_text_encoder_model,
                offload_prompt_upsampler=args.offload_prompt_upsampler,
                offload_guardrail_models=args.offload_guardrail_models,
                disable_guardrail=args.disable_guardrail,
                disable_prompt_encoder=args.disable_prompt_encoder,
                guidance=args.guidance, num_steps=args.num_steps,
                height=args.height, width=args.width, fps=args.fps,
                num_video_frames=121, seed=args.seed,
            )
            if args.num_gpus > 1:
                pipeline.model.net.enable_context_parallel(parallel_state.get_context_parallel_group())
            return _save_generations(args, pipeline, scene, torch.device('cuda'))
    finally:
        if distributed_started:
            parallel_state.destroy_model_parallel()
            import torch.distributed as dist
            dist.destroy_process_group()


def _save_generations(args, pipeline, scene, device):
    from cosmos_predict1.utils import log
    from cosmos_predict1.utils.io import read_prompts_from_file, save_video
    from cosmos_predict1.diffusion.inference.instance_segmentation import file_sha256

    prompts = read_prompts_from_file(args.batch_input_path) if args.batch_input_path else [{'prompt': args.prompt}]
    output_dir = Path(args.video_save_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[3], check=False).stdout.strip()
    settings = {key: getattr(args, key, None) for key in (
        'seed', 'guidance', 'num_steps', 'fps', 'height', 'width', 'num_gpus',
        'disable_guardrail', 'disable_prompt_encoder', 'disable_prompt_upsampler',
        'offload_diffusion_transformer', 'offload_tokenizer', 'offload_text_encoder_model',
        'offload_prompt_upsampler', 'offload_guardrail_models', 'prompt_upsampler_dir',
    )}
    settings.update(code_revision=revision, checkpoint_dir=str(Path(args.checkpoint_dir).resolve()),
                    checkpoint_name='Gen3C-Cosmos-7B', checkpoint_revision=getattr(args, 'checkpoint_revision', None))
    if is_writer and not settings['checkpoint_revision']:
        settings['checkpoint_revision'] = 'sha256:' + file_sha256(
            Path(args.checkpoint_dir) / 'Gen3C-Cosmos-7B' / 'model.pt')
    composite_pipeline = Gen3cCompositePipeline(pipeline)
    if is_writer:
        for source in scene.to_dict()['sources']:
            log.info(f'Composite source: {json.dumps(source)}')
    for index, entry in enumerate(prompts):
        prompt = entry.get('prompt')
        if prompt is None and args.disable_prompt_upsampler:
            log.critical('Prompt is missing, skipping world generation.')
            continue
        name = str(index) if args.batch_input_path else args.video_save_name
        review_dir = output_dir / f'{name}_review'
        output = composite_pipeline.generate(
            scene, prompt=prompt, negative_prompt=args.negative_prompt,
            num_frames=args.num_video_frames, device=device, review_dir=review_dir,
            generation_settings=settings, min_seed_coverage=getattr(args, 'min_seed_coverage', 0.5),
        )
        if output is None:
            log.critical('Guardrail blocked composite generation.')
            continue
        if not is_writer:
            continue
        video, final_prompt = output
        video_path = output_dir / f'{name}.mp4'
        save_video(video=video, fps=args.fps, H=args.height, W=args.width,
                   video_save_quality=5, video_save_path=str(video_path))
        (review_dir / 'prompt.txt').write_text(final_prompt or '')
        manifest_path = review_dir / 'generation.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(status='saved', final_video=str(video_path.resolve()))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        log.info(f'Saved composite video to {video_path}; review artifacts in {review_dir}')


if __name__ == '__main__':
    demo(create_parser().parse_args())
