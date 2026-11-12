"""Exact artifact loading and end-to-end conditioning with mocked diffusion."""

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import ast
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.composite_scene import (
    SceneSource, SceneSpec, build_composite_scene, load_prepared_source, save_prepared_source,
)
from cosmos_predict1.diffusion.inference.gen3c_composite_pipeline import Gen3cCompositePipeline
from cosmos_predict1.diffusion.inference.gen3c_composite import create_parser, demo, _load_prompts, _save_generations
from cosmos_predict1.diffusion.inference.instance_labeling import (
    label_source, prepared_source_fingerprint, save_labeled_source,
)
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    InstanceProposal, PreparedSource, SegmentedSource, SourceObservation,
    _segmented_observation, prepare_source,
)


class CompositePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.review = self.root / 'review'
        self.key = (4, 7)
        self.h, self.w = 4, 6
        sources = []
        self.prepared = {}
        for source_id in ('room', 'objects'):
            rgb = torch.full((3, self.h, self.w), -1.)
            rgb[2 if source_id == 'room' else 0] = 1
            if source_id == 'objects':
                rgb[:, :, 3:] = torch.tensor([-1., 1., -1.])[:, None, None]
            depth = torch.full((self.h, self.w), 2. if source_id == 'room' else 1.)
            prepared = prepare_source({self.key: SourceObservation(rgb, depth, torch.eye(3), torch.eye(4))})
            proposals = []
            environment = torch.ones_like(depth, dtype=torch.bool) if source_id == 'room' else None
            if source_id == 'objects':
                left = torch.zeros_like(depth, dtype=torch.bool)
                left[:, :3] = True
                proposals = [InstanceProposal(left, None, 1), InstanceProposal(~left, None, 2)]
            segmented = SegmentedSource(prepared, {self.key: _segmented_observation(
                self.key, prepared.observations[self.key], proposals, environment)})
            labeled = label_source(segmented, source_id=source_id)
            save_prepared_source(self.root / f'{source_id}.npz', prepared)
            save_labeled_source(self.root / f'{source_id}_labels', labeled)
            self.prepared[source_id] = prepared
            sources.append(SceneSource(source_id, self.root / f'{source_id}.npz',
                                       self.root / f'{source_id}_labels', labeled.source_fingerprint,
                                       [0] if source_id == 'room' else [1]))
        self.scene = SceneSpec(sources, 'room', torch.eye(4).repeat(1, 5, 1, 1),
                               torch.eye(3).repeat(1, 5, 1, 1), (self.h, self.w))
        self.diffusion = SimpleNamespace(model=SimpleNamespace(chunk_size=3, frame_buffer_max=2),
                                         height=self.h, width=self.w)
        self.segments = [np.full((3, self.h, self.w, 3), n, dtype=np.uint8) for n in (64, 192)]
        self.diffusion.generate = Mock(side_effect=[(v, 'final prompt') for v in self.segments])
        self.pipeline = Gen3cCompositePipeline(self.diffusion)

    def generate(self, scene=None, **kwargs):
        return self.pipeline.generate(self.scene if scene is None else scene, prompt='a red object',
                                      device='cpu', review_dir=self.review, **kwargs)

    def test_json_round_trip_relative_paths_and_equivalent_composites(self):
        data = self.scene.to_dict()
        for source in data['sources']:
            source['prepared_path'] = Path(source['prepared_path']).name
            source['labels_path'] = Path(source['labels_path']).name
        path = self.root / 'scene.json'
        path.write_text(json.dumps(data))
        loaded = SceneSpec.load(path)
        self.assertEqual(loaded.to_dict(), self.scene.to_dict())
        loaded.save(self.root / 'roundtrip.json')
        again = SceneSpec.load(self.root / 'roundtrip.json')
        original = build_composite_scene(self.scene)
        restored = build_composite_scene(again)
        a = original.render_cache(*self.scene.cameras())
        b = restored.render_cache(*again.cameras())
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(restored.sources['room'].observation_keys, (self.key,))
        self.assertEqual(restored.selections, {'room': {0}, 'objects': {1}})

    def test_conditioning_seed_continuation_and_static_children(self):
        composite = build_composite_scene(self.scene)
        snapshots = {name: (cache.input_points.clone(), cache.input_image.clone(), cache.input_labels.clone(),
                             cache.geometry_revision) for name, cache in composite.sources.items()}
        with patch('cosmos_predict1.diffusion.inference.gen3c_composite_pipeline.build_composite_scene',
                   return_value=composite), patch.object(composite, 'render_conditioning',
                                                        wraps=composite.render_conditioning) as render:
            video, prompt = self.generate()
        self.assertEqual([call.kwargs['start_frame_idx'] for call in render.call_args_list], [0, 2])
        for call, start in zip(render.call_args_list, (0, 2)):
            torch.testing.assert_close(call.args[0], self.scene.cameras()[0][:, start:start + 3])
        self.assertEqual(video.shape, (5, self.h, self.w, 3))
        np.testing.assert_array_equal(video[:3], self.segments[0])
        np.testing.assert_array_equal(video[3:], self.segments[1][1:])
        self.assertEqual(prompt, 'final prompt')
        first, second = [call.kwargs for call in self.diffusion.generate.call_args_list]
        self.assertEqual(first['rendered_warp_images'].shape, (1, 3, 1, 3, self.h, self.w))
        self.assertEqual(first['rendered_warp_masks'].shape, (1, 3, 1, 1, self.h, self.w))
        self.assertEqual(first['rendered_warp_images'].dtype, torch.float32)
        torch.testing.assert_close(first['rendered_warp_masks'], torch.ones(1, 3, 1, 1, self.h, self.w))
        torch.testing.assert_close(first['image_path'], first['rendered_warp_images'][:, 0, 0].unsqueeze(2))
        self.assertEqual(first['image_path'].shape, (1, 3, 1, self.h, self.w))
        # Selected foreground is red; excluded green object never reaches the seed.
        torch.testing.assert_close(first['image_path'][0, :, 0, :, :3],
                                   torch.tensor([1., -1., -1.])[:, None, None].expand(3, self.h, 3))
        torch.testing.assert_close(first['image_path'][0, :, 0, :, 3:],
                                   torch.tensor([-1., -1., 1.])[:, None, None].expand(3, self.h, 3))
        torch.testing.assert_close(second['image_path'], torch.full((1, 3, 1, self.h, self.w), 64 / 127.5 - 1))
        for name, cache in composite.sources.items():
            points, rgb, labels, revision = snapshots[name]
            torch.testing.assert_close(cache.input_points, points, rtol=0, atol=0)
            torch.testing.assert_close(cache.input_image, rgb, rtol=0, atol=0)
            torch.testing.assert_close(cache.input_labels, labels, rtol=0, atol=0)
            self.assertEqual(cache.geometry_revision, revision)
        manifest = json.loads((self.review / 'generation.json').read_text())
        self.assertFalse(manifest['generated_cache_updates'])
        self.assertEqual(manifest['status'], 'generated')
        self.assertEqual(manifest['seed_coverage'], 1)
        self.assertTrue((self.review / 'seed.png').is_file())
        with np.load(self.review / 'render_000000.npz') as arrays:
            self.assertEqual(set(arrays.files), {'rgb', 'depth', 'coverage'})
            np.testing.assert_array_equal(arrays['rgb'], first['rendered_warp_images'])

    def test_review_depth_released_before_diffusion_and_conditioning_before_next_render(self):
        composite = build_composite_scene(self.scene)
        render = composite.render_conditioning
        references = {}
        generated = iter(self.segments)

        def render_chunk(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in references.values()))
            result = render(*args, **kwargs)
            self.assertEqual(set(result), {'rgb', 'coverage', 'depth'})
            references.update({key: weakref.ref(value) for key, value in result.items()})
            return result

        def diffuse(**kwargs):
            self.assertIsNone(references['depth']())
            return next(generated), 'final prompt'

        self.diffusion.generate = diffuse  # Avoid mocks retaining input tensors.
        with patch('cosmos_predict1.diffusion.inference.gen3c_composite_pipeline.build_composite_scene',
                   return_value=composite), patch.object(composite, 'render_conditioning', render_chunk), \
                patch.object(composite, 'render_diagnostics', side_effect=AssertionError('Unused diagnostics')):
            self.generate()
        self.assertTrue(all(ref() is None for ref in references.values()))

    def test_nonwriter_omits_review_depth_hashes_and_files(self):
        composite = build_composite_scene(self.scene)
        with patch('cosmos_predict1.diffusion.inference.gen3c_composite_pipeline.build_composite_scene',
                   return_value=composite), patch.object(composite, 'render_conditioning',
                                                        wraps=composite.render_conditioning) as render, \
                patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_rank', return_value=1), \
                patch('cosmos_predict1.diffusion.inference.composite_review.file_sha256',
                      side_effect=AssertionError('Nonwriter must not hash review metadata')):
            self.generate()
        self.assertTrue(all(not call.kwargs['include_depth'] for call in render.call_args_list))
        self.assertFalse(self.review.exists())

    def test_writer_uses_validated_provenance_without_reparsing_labels(self):
        composite = build_composite_scene(self.scene)
        with patch('cosmos_predict1.diffusion.inference.gen3c_composite_pipeline.build_composite_scene',
                   return_value=composite), patch('json.loads', side_effect=AssertionError('Repeated JSON parse')):
            self.generate()
        sources = json.loads((self.review / 'generation.json').read_text())['sources']
        for source in sources:
            self.assertEqual(source['segmentation_revision'],
                             composite.source_provenance[source['source_id']]['segmentation_revision'])
            self.assertEqual(len(source['labels_sha256']), 64)

    def test_multi_chunk_video_concatenated_once(self):
        scene = replace(self.scene, target_w2cs=torch.eye(4).repeat(1, 7, 1, 1),
                        target_intrinsics=torch.eye(3).repeat(1, 7, 1, 1))
        third = np.full_like(self.segments[0], 224)
        self.diffusion.generate.side_effect = [(v, 'prompt') for v in (*self.segments, third)]
        with patch('numpy.concatenate', wraps=np.concatenate) as concatenate:
            video, _ = self.generate(scene)
        self.assertEqual(concatenate.call_count, 1)
        np.testing.assert_array_equal(video[-2:], third[1:])
        self.assertEqual(video.shape[0], 7)

    def test_explicit_prompts_required_for_python_and_batch_callers(self):
        for prompt in (None, '', '  '):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, 'explicit prompt'):
                self.pipeline.generate(self.scene, prompt=prompt, device='cpu', review_dir=self.review)
            with self.assertRaisesRegex(ValueError, 'explicit prompt'):
                _load_prompts(SimpleNamespace(batch_input_path=None, prompt=prompt))
        io_module = SimpleNamespace(read_prompts_from_file=Mock(return_value=[{'prompt': 'ok'}, {}]))
        with patch.dict(sys.modules, {'cosmos_predict1.utils.io': io_module}), \
                self.assertRaisesRegex(ValueError, 'every entry'):
            _load_prompts(SimpleNamespace(batch_input_path='batch.jsonl'))
        self.diffusion.generate.assert_not_called()

    def test_cli_omits_upsampler_flags_and_initializes_without_sam(self):
        # Exercise the actual shared argument helpers without importing CUDA dependencies.
        path = Path(__file__).parents[1] / 'cosmos_predict1/diffusion/inference/inference_utils.py'
        functions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
                     and node.name in ('add_common_arguments', 'remove_argument')]
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        with patch.dict(sys.modules, {'cosmos_predict1.diffusion.inference.inference_utils': SimpleNamespace(**namespace)}):
            parser = create_parser()
        args = parser.parse_args(['--scene_spec', 'unused', '--prompt', 'red object',
                                  '--height', str(self.h), '--width', str(self.w)])
        for name in ('prompt_upsampler_dir', 'disable_prompt_upsampler', 'offload_prompt_upsampler', 'tokenizer_dir'):
            self.assertFalse(hasattr(args, name))
            self.assertNotIn('--' + name, parser.format_help())
        args.scene_spec = None
        scene = replace(self.scene, target_w2cs=torch.eye(4).repeat(1, 121, 1, 1),
                        target_intrinsics=torch.eye(3).repeat(1, 121, 1, 1))
        factory = Mock()
        with patch.dict(sys.modules, {
            'cosmos_predict1.diffusion.inference.gen3c_pipeline': SimpleNamespace(Gen3cPipeline=factory),
            'cosmos_predict1.utils.misc': SimpleNamespace(set_random_seed=Mock()),
        }), patch('cosmos_predict1.utils.model_config.model_config_session', return_value=nullcontext()) as session, \
                patch('cosmos_predict1.diffusion.inference.gen3c_composite._save_generations'):
            demo(args, scene)
        session.assert_called_once_with()
        self.assertIs(factory.call_args.kwargs['enable_prompt_upsampler'], False)
        self.assertNotIn('prompt_upsampler_dir', factory.call_args.kwargs)

    def test_deterministic_holes_and_low_coverage_failure(self):
        scene = deepcopy(self.scene)
        # Shift the camera one pixel right: the rightmost column becomes a hole.
        scene.target_w2cs[:, :, 0, 3] = -2
        self.generate(scene, min_seed_coverage=0.1)
        seed = self.diffusion.generate.call_args_list[0].kwargs['image_path']
        torch.testing.assert_close(seed[..., -1], -torch.ones_like(seed[..., -1]))
        self.diffusion.generate.reset_mock()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.generate(scene, min_seed_coverage=1)
        self.diffusion.generate.assert_not_called()
        self.assertTrue((self.review / 'seed.npy').is_file())
        self.assertEqual(json.loads((self.review / 'generation.json').read_text())['status'], 'failed')

    def test_guardrail_failure_on_continuation_stops_generation(self):
        self.diffusion.generate.side_effect = [(self.segments[0], 'prompt'), None]
        self.assertIsNone(self.generate())
        manifest = json.loads((self.review / 'generation.json').read_text())
        self.assertEqual((manifest['status'], manifest['blocked_chunk']), ('blocked', 2))

    def test_shared_validation_for_direct_and_json_models(self):
        mutations = [dict(version=2), dict(environment_source='missing'), dict(image_size=(0, 6)),
                     dict(sources=[self.scene.sources[0], self.scene.sources[0]]),
                     dict(target_w2cs=torch.zeros(1, 5, 4, 4)),
                     dict(target_intrinsics=torch.eye(3).repeat(1, 4, 1, 1))]
        for mutation in mutations:
            with self.subTest(mutation=list(mutation)):
                with self.assertRaises(ValueError):
                    replace(self.scene, **mutation)
                data = self.scene.to_dict()
                for name, value in mutation.items():
                    data[name] = value.tolist() if isinstance(value, torch.Tensor) else value
                if 'sources' in mutation:
                    data['sources'] = [self.scene.to_dict()['sources'][0]] * 2
                path = self.root / 'invalid.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    SceneSpec.load(path)
        for changes in (dict(selected_ids=[-1]), dict(selected_ids=[True]), dict(selected_ids=[99],
                        object_transforms={1: torch.eye(4)}), dict(selected_ids=[0]),
                        dict(source_transform=torch.zeros(4, 4))):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                replace(self.scene, sources=[self.scene.sources[0], replace(self.scene.sources[1], **changes)])

    def test_invalid_runtime_configuration(self):
        for count in (1, 4, 7):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.generate(num_frames=count)
        self.diffusion.model.frame_buffer_max = 0
        with self.assertRaisesRegex(ValueError, 'buffer'):
            self.generate()
        self.diffusion.model.frame_buffer_max = 2
        self.diffusion.height += 1
        with self.assertRaisesRegex(ValueError, 'size'):
            self.generate()
        self.diffusion.generate.assert_not_called()

    def test_invalid_artifacts_and_unknown_selections(self):
        scene = deepcopy(self.scene)
        scene.sources[1].selected_ids = [99]
        with self.assertRaises(ValueError):
            build_composite_scene(scene)
        scene = deepcopy(self.scene)
        scene.sources[1].fingerprint = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            build_composite_scene(scene)
        prepared = self.prepared['objects']
        obs = prepared.observations[self.key]
        changed = PreparedSource({self.key: replace(obs, rgb=obs.rgb * 0.5)}, prepared.preparation)
        save_prepared_source(self.root / 'objects.npz', changed)
        scene.sources[1].fingerprint = prepared_source_fingerprint(changed)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            build_composite_scene(scene)

    def test_tampered_labels_and_source_ownership_are_rejected(self):
        scene = deepcopy(self.scene)
        scene.sources[1].source_id = 'wrong_owner'
        with self.assertRaisesRegex(ValueError, 'ownership'):
            build_composite_scene(scene)
        path = self.root / 'objects_labels.npz'
        with path.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            build_composite_scene(self.scene)

    def test_malformed_scene_json_requires_version_and_integer_transform_keys(self):
        data = self.scene.to_dict()
        del data['version']
        path = self.root / 'bad.json'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'version'):
            SceneSpec.load(path)
        data = self.scene.to_dict()
        data['sources'][1]['object_transforms'] = {'01': torch.eye(4).tolist()}
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'canonical'):
            SceneSpec.load(path)

    def test_prepared_geometry_not_clamped_and_original_keys_settings_preserved(self):
        prepared = self.prepared['room']
        obs = prepared.observations[self.key]
        prepared = PreparedSource({self.key: replace(obs, depth=torch.full_like(obs.depth, 150.))},
                                  {'crop': None, 'output_size': [4, 6], 'custom': 'exact'})
        segmented = SegmentedSource(prepared, {self.key: _segmented_observation(
            self.key, prepared.observations[self.key], [], torch.ones(self.h, self.w, dtype=torch.bool))})
        labeled = label_source(segmented, source_id='room')
        save_prepared_source(self.root / 'room.npz', prepared)
        save_labeled_source(self.root / 'room_labels', labeled)
        self.scene.sources[0].fingerprint = labeled.source_fingerprint
        restored = load_prepared_source(self.root / 'room.npz', labeled.source_fingerprint)
        self.assertEqual(restored.preparation, prepared.preparation)
        composite = build_composite_scene(self.scene)
        self.assertTrue((composite.sources['room'].input_points[..., 2] == 150).all())

    def test_nontrivial_transform_round_trip(self):
        angle = torch.tensor(0.123456789, dtype=torch.float64)
        matrix = torch.eye(4, dtype=torch.float64)
        matrix[:2, :2] = torch.stack((torch.stack((angle.cos(), -angle.sin())),
                                     torch.stack((angle.sin(), angle.cos()))))
        matrix[:3, :3] *= 1.23456789
        self.scene.sources[1].source_transform = matrix
        self.scene.sources[1].object_transforms = {1: torch.eye(4, dtype=torch.float64)}
        self.scene.save(self.root / 'scene.json')
        restored = SceneSpec.load(self.root / 'scene.json')
        a, b = build_composite_scene(self.scene), build_composite_scene(restored)
        torch.testing.assert_close(a.source_transforms['objects'], b.source_transforms['objects'], rtol=0, atol=0)
        self.assertEqual(self.scene.to_dict(), restored.to_dict())

    def test_cli_rejects_legacy_inputs_before_loading_diffusion(self):
        for name in ('npz_path', 'labels_path', 'selected_ids'):
            args = SimpleNamespace(height=self.h, width=self.w, scene_spec=None, **{name: [] if name == 'selected_ids' else 'source'})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'legacy'):
                demo(args, self.scene)
        with self.assertRaisesRegex(ValueError, 'both'):
            demo(SimpleNamespace(scene_spec='scene.json'), self.scene)

    def test_cli_saves_final_video_and_checkpoint_revision(self):
        args = SimpleNamespace(batch_input_path=None, prompt='red object', negative_prompt=None,
                               video_save_folder=str(self.root / 'outputs'), video_save_name='result',
                               checkpoint_dir=str(self.root / 'checkpoints'), checkpoint_revision='abc123',
                               num_video_frames=5, fps=24, height=self.h, width=self.w,
                               disable_prompt_upsampler=True)
        save_video = Mock()
        io_module = SimpleNamespace(save_video=save_video, read_prompts_from_file=Mock())
        with patch.dict(sys.modules, {'cosmos_predict1.utils.io': io_module}):
            _save_generations(args, self.diffusion, self.scene, 'cpu')
        save_video.assert_called_once()
        self.assertEqual(save_video.call_args.kwargs['video'].shape, (5, self.h, self.w, 3))
        manifest_path = self.root / 'outputs' / 'result_review' / 'generation.json'
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest['status'], 'saved')
        self.assertEqual(manifest['generation_settings']['checkpoint_revision'], 'abc123')
        self.assertTrue(manifest['final_video'].endswith('result.mp4'))

    def test_real_model_encoder_accepts_singleton_and_pads_zero_latents(self):
        # Load the actual model module with training/distributed imports stubbed;
        # execute its real conditioning method with a mocked tokenizer encoder.
        path = Path(__file__).parents[1] / 'cosmos_predict1/diffusion/model/model_gen3c.py'
        spec = importlib.util.spec_from_file_location('_test_gen3c_model', path)
        module = importlib.util.module_from_spec(spec)
        stubs = {
            'megatron.core': SimpleNamespace(parallel_state=Mock()),
            'cosmos_predict1.diffusion.conditioner': SimpleNamespace(VideoExtendCondition=object),
            'cosmos_predict1.diffusion.model.model_v2w': SimpleNamespace(
                DiffusionV2WModel=object, broadcast_condition=Mock()),
        }
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        fake = SimpleNamespace(frame_buffer_max=2, encode=Mock(side_effect=lambda value: value + 2))
        rgb, coverage = build_composite_scene(self.scene).render_cache(*self.scene.cameras())
        encoded = module.DiffusionGen3CModel.encode_warped_frames(fake, rgb, coverage, torch.float32)
        self.assertEqual(fake.encode.call_count, 2)
        self.assertEqual(encoded.shape, (1, 12, 5, self.h, self.w))
        torch.testing.assert_close(encoded[:, :3], rgb[:, :, 0].permute(0, 2, 1, 3, 4) + 2)
        torch.testing.assert_close(encoded[:, 3:6], torch.full_like(encoded[:, 3:6], 3))
        torch.testing.assert_close(encoded[:, 6:], torch.zeros_like(encoded[:, 6:]))


if __name__ == '__main__':
    unittest.main()
