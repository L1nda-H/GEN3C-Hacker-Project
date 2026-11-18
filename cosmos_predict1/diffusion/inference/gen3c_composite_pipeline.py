"""Composite conditioning at the GEN3C entry-point boundary (no cache updates)."""

from pathlib import Path

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.composite_scene import SceneSpec, build_composite_scene
from cosmos_predict1.diffusion.inference.composite_review import save_manifest, save_render, source_manifest
from cosmos_predict1.diffusion.inference.composite_review import save_seed_result
from cosmos_predict1.diffusion.inference.composite_seed import build_image_space_seed


class Gen3cCompositePipeline:
    """Static composition pipeline wrapping the existing pretrained diffusion API.

    Construct with an initialized Gen3cPipeline. This owns scene conditioning,
    seed selection, chunk continuation and review outputs, while diffusion owns
    model weights and its existing generate() tensor contract.
    """

    def __init__(self, diffusion_pipeline):
        self.diffusion_pipeline = diffusion_pipeline

    @torch.no_grad()
    def generate(self, scene: SceneSpec, *, prompt, negative_prompt=None,
                 num_frames=None, device='cuda', review_dir,
                 generation_settings=None, prepared_composite=None, seed_plane_depths=None,
                 seed_occlusion_policy='reject', seed_removal_mask=None):
        """Return normal pipeline output, or None if any chunk is guardrail-blocked.

        The first seed is always a complete image-space composite of source RGB.
        Its camera must match the environment source camera and covered field of view.
        Only warp conditioning has an accompanying spatial coverage mask.
        Later chunks use the preceding generated frame and overlap by one frame.
        prepared_composite may supply a validated snapshot of this scene for a
        batch of prompts. Source artifacts must remain unchanged during the batch.
        """
        pipeline = self.diffusion_pipeline
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Composite generation requires a nonempty explicit prompt")
        scene.validate()
        chunk_size = pipeline.model.chunk_size
        if type(chunk_size) is not int or chunk_size < 2:
            raise ValueError("GEN3C chunk size must be an integer greater than one")
        if getattr(pipeline, 'num_video_frames', chunk_size) != chunk_size:
            raise ValueError("Diffusion num_video_frames must match the model chunk size")
        # The existing encoder handles each RGB/mask buffer independently and
        # concatenates latents, padding unused checkpoint slots with zeros.
        if type(pipeline.model.frame_buffer_max) is not int or pipeline.model.frame_buffer_max < 1:
            raise ValueError("GEN3C must accept at least one conditioning buffer")
        if (pipeline.height, pipeline.width) != tuple(scene.image_size):
            raise ValueError("Scene target size must match the generation model; resizing is forbidden")
        poses, intrinsics = scene.cameras(device)
        num_frames = poses.shape[1] if num_frames is None else num_frames
        if (type(num_frames) is not int or num_frames < chunk_size or num_frames > poses.shape[1]
                or (num_frames - 1) % (chunk_size - 1)):
            raise ValueError("Trajectory length must cover complete chunks with one-frame overlap")
        composite = (build_composite_scene(scene, device=device)
                     if prepared_composite is None else prepared_composite)
        buffer_source_ids = composite.conditioning_source_ids
        if len(buffer_source_ids) > pipeline.model.frame_buffer_max:
            raise ValueError(f"Scene requires {len(buffer_source_ids)} conditioning buffers, but checkpoint "
                             f"supports {pipeline.model.frame_buffer_max}; use a compatible checkpoint")
        review_dir = Path(review_dir)
        is_writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_writer:
            review_dir.mkdir(parents=True, exist_ok=True)
            scene.save(review_dir / 'scene.json')
        settings = {name: getattr(pipeline, name, None) for name in (
            'checkpoint_dir', 'checkpoint_name', 'seed', 'guidance', 'num_steps', 'height', 'width', 'fps',
        )}
        settings.update(generation_settings or {})
        settings['torch_version'] = str(torch.__version__)
        manifest = dict(status='rendering', sources=source_manifest(scene, composite) if is_writer else [],
                        prompt=prompt, negative_prompt=negative_prompt,
                        generation_settings=settings, num_frames=num_frames,
                        chunk_size=chunk_size, frame_buffer_max=pipeline.model.frame_buffer_max,
                        generated_cache_updates=False, seed_hole_fill=None,
                        seed_strategy='image_space',
                        conditioning_source_ids=buffer_source_ids,
                        conditioning_strategy='per_source_latent_concat',
                        conditioning_visibility='joint_nearest_surface',
                        depth_tolerance=composite.depth_tolerance,
                        renderer=composite.render_settings)

        def record(**changes):
            manifest.update(changes)
            if is_writer:
                save_manifest(review_dir, manifest)

        record()
        segments = []
        last_frame = None
        completed_frames = 0
        final_prompt = prompt
        try:
            initial_seed = build_image_space_seed(
                scene, composite, poses[0, 0], intrinsics[0, 0],
                plane_depths=seed_plane_depths, occlusion_policy=seed_occlusion_policy,
                removal_mask=seed_removal_mask)
            record(seed_metadata=initial_seed.metadata, seed_coverage=initial_seed.metadata['completeness'])
            if is_writer:
                save_seed_result(review_dir, initial_seed)
            for start in range(0, num_frames - 1, chunk_size - 1):
                end = start + chunk_size
                rendered = composite.render_conditioning(poses[:, start:end], intrinsics[:, start:end],
                                                          start_frame_idx=start, include_depth=is_writer)
                rgb, coverage = rendered['rgb'], rendered['coverage']
                if is_writer:
                    save_render(review_dir, start, rendered)
                del rendered  # Release review depth before diffusion.
                if start == 0:
                    # Warp coverage is diagnostic only; it never selects or fills seed RGB.
                    record(initial_warp_coverage=float(coverage[:, :1].bool().any(dim=2).float().mean()))
                    seed = initial_seed.rgb.to(device=device).clone()
                    del initial_seed
                else:
                    seed = torch.as_tensor(last_frame.copy(), device=device, dtype=torch.float32)
                    seed = (seed.permute(2, 0, 1)[None, :, None] / 127.5) - 1
                output = pipeline.generate(prompt=prompt, image_path=seed, negative_prompt=negative_prompt,
                                           rendered_warp_images=rgb, rendered_warp_masks=coverage)
                del rgb, coverage, seed  # No previous chunk remains live during the next render.
                if output is None:
                    record(status='blocked', blocked_chunk=start)
                    return None
                segment, final_prompt = output
                if segment.shape != (chunk_size, *scene.image_size, 3) or segment.dtype != np.uint8:
                    raise ValueError("GEN3C output must be uint8 [chunk_size,H,W,3]")
                last_frame = segment[-1]
                segments.append(segment if start == 0 else segment[1:])
                completed_frames += len(segments[-1])
                record(status='generating', completed_frames=completed_frames)
        except Exception as error:
            record(status='failed', error=str(error))
            raise
        video = segments[0] if len(segments) == 1 else np.concatenate(segments, axis=0)
        record(status='generated', final_prompt=final_prompt)
        return video, final_prompt
