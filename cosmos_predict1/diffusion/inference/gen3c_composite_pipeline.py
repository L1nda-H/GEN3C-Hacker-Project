"""Composite conditioning at the GEN3C entry-point boundary (no cache updates)."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cosmos_predict1.diffusion.inference.composite_scene import SceneSpec, build_composite_scene
from cosmos_predict1.diffusion.inference.instance_segmentation import file_sha256


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
                 generation_settings=None, min_seed_coverage=0.5):
        """Return normal pipeline output, or None if any chunk is guardrail-blocked.

        The first trajectory camera must be well covered; choose it when authoring
        the scene. Holes in the separate seed path are deterministic black (-1),
        exactly as rendered. Only warp conditioning has an accompanying mask.
        Later chunks use the preceding generated frame and overlap by one frame.
        """
        pipeline = self.diffusion_pipeline
        scene.validate()
        if not 0 < min_seed_coverage <= 1:
            raise ValueError("min_seed_coverage must be in (0,1]")
        chunk_size = pipeline.model.chunk_size
        if type(chunk_size) is not int or chunk_size < 2:
            raise ValueError("GEN3C chunk size must be an integer greater than one")
        if getattr(pipeline, 'num_video_frames', chunk_size) != chunk_size:
            raise ValueError("Diffusion num_video_frames must match the model chunk size")
        # DiffusionGen3CModel.encode_warped_frames encodes the singleton and pads
        # missing buffers with zero latents up to frame_buffer_max.
        if type(pipeline.model.frame_buffer_max) is not int or pipeline.model.frame_buffer_max < 1:
            raise ValueError("GEN3C must accept at least one conditioning buffer")
        if (pipeline.height, pipeline.width) != tuple(scene.image_size):
            raise ValueError("Scene target size must match the generation model; resizing is forbidden")
        poses, intrinsics = scene.cameras(device)
        num_frames = poses.shape[1] if num_frames is None else num_frames
        if (type(num_frames) is not int or num_frames < chunk_size or num_frames > poses.shape[1]
                or (num_frames - 1) % (chunk_size - 1)):
            raise ValueError("Trajectory length must cover complete chunks with one-frame overlap")
        composite = build_composite_scene(scene, device=device)
        review_dir = Path(review_dir)
        is_writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_writer:
            review_dir.mkdir(parents=True, exist_ok=True)
            scene.save(review_dir / 'scene.json')
        sources = []
        for source in scene.sources:
            label_json = Path(source.labels_path).with_suffix('.json')
            labels = json.loads(label_json.read_text())
            sources.append(dict(source_id=source.source_id, fingerprint=source.fingerprint,
                                labels_sha256=file_sha256(label_json),
                                segmentation_revision=labels['segmentation_revision'],
                                provenance=labels['provenance'],
                                resolved_ids=sorted(composite.selections[source.source_id]),
                                source_transform=composite.source_transforms[source.source_id].tolist(),
                                object_transforms={str(i): t.tolist() for (owner, i), t in
                                                   composite.object_transforms.items() if owner == source.source_id}))
        settings = {name: getattr(pipeline, name, None) for name in (
            'checkpoint_dir', 'checkpoint_name', 'seed', 'guidance', 'num_steps', 'height', 'width', 'fps',
        )}
        settings.update(generation_settings or {})
        settings['torch_version'] = str(torch.__version__)
        manifest = dict(status='rendering', sources=sources, prompt=prompt, negative_prompt=negative_prompt,
                        generation_settings=settings, num_frames=num_frames,
                        chunk_size=chunk_size, frame_buffer_max=pipeline.model.frame_buffer_max,
                        generated_cache_updates=False, seed_hole_fill=-1,
                        min_seed_coverage=min_seed_coverage)

        def save_manifest():
            if is_writer:
                (review_dir / 'generation.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')

        save_manifest()
        video = None
        final_prompt = prompt
        try:
            for start in range(0, num_frames - 1, chunk_size - 1):
                end = start + chunk_size
                rendered = composite.render_diagnostics(poses[:, start:end], intrinsics[:, start:end],
                                                         start_frame_idx=start, include_rgb=True)
                rgb, coverage = rendered['rgb'], rendered['coverage']
                if is_writer:
                    np.savez_compressed(review_dir / f'render_{start:06d}.npz',
                                        rgb=rgb.cpu().numpy(), depth=rendered['depth'].cpu().numpy(),
                                        coverage=coverage.cpu().numpy())
                    # PNG previews supplement the lossless, metrically meaningful NPZ.
                    first_rgb = ((rgb[0, 0, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
                    cv2.imwrite(str(review_dir / f'rgb_{start:06d}.png'), first_rgb[:, :, ::-1])
                    cv2.imwrite(str(review_dir / f'coverage_{start:06d}.png'),
                                (coverage[0, 0, 0, 0].cpu().numpy() * 255).astype(np.uint8))
                    z = rendered['depth'][0, 0, 0].cpu().numpy()
                    z_preview = (z / max(float(z.max()), 1e-8) * 65535).round().astype(np.uint16)
                    cv2.imwrite(str(review_dir / f'depth_{start:06d}.png'), z_preview)
                if start == 0:
                    seed = rgb[:, 0, 0].unsqueeze(2).clone()  # [B,C,1,H,W], [-1,1]
                    if is_writer:
                        np.save(review_dir / 'seed.npy', seed.cpu().numpy())
                        cv2.imwrite(str(review_dir / 'seed.png'), first_rgb[:, :, ::-1])
                    manifest['seed_coverage'] = float(coverage[:, 0].mean())
                    save_manifest()
                    if manifest['seed_coverage'] < min_seed_coverage:
                        raise ValueError("Initial composed seed has insufficient coverage; choose a better starting camera")
                else:
                    seed = torch.as_tensor(video[-1].copy(), device=device, dtype=torch.float32)
                    seed = (seed.permute(2, 0, 1)[None, :, None] / 127.5) - 1
                output = pipeline.generate(prompt=prompt, image_path=seed, negative_prompt=negative_prompt,
                                           rendered_warp_images=rgb, rendered_warp_masks=coverage)
                if output is None:
                    manifest.update(status='blocked', blocked_chunk=start)
                    save_manifest()
                    return None
                segment, final_prompt = output
                if segment.shape != (chunk_size, *scene.image_size, 3) or segment.dtype != np.uint8:
                    raise ValueError("GEN3C output must be uint8 [chunk_size,H,W,3]")
                video = segment if video is None else np.concatenate((video, segment[1:]), axis=0)
                manifest.update(status='generating', completed_frames=len(video))
                save_manifest()
        except Exception as error:
            manifest.update(status='failed', error=str(error))
            save_manifest()
            raise
        manifest.update(status='generated', final_prompt=final_prompt)
        save_manifest()
        return video, final_prompt
