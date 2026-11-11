"""CPU geometry/rendering and entry-point contracts for labeled source caches."""

import ast
import tempfile
import types
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base, Cache3D_Buffer, Cache3D_BufferSelector, Cache4D
from cosmos_predict1.diffusion.inference.instance_labeling import (
    ObjectRegistry, label_source, label_overlays, load_cache_label_kwargs, load_labeled_source, save_labeled_source,
)
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    InstanceProposal, SegmentedSource, SourceObservation, _segmented_observation, prepare_source,
)


CACHE_MODULE = 'cosmos_predict1.diffusion.inference.cache_3d'


def labeled_source(keys=((4, 7),), *, invalid=False, registry=None, right=False):
    inputs = {}
    for index, key in enumerate(keys):
        depth = torch.full((4, 6), 2.)
        if invalid:
            depth[0, :4] = torch.tensor([float('nan'), float('inf'), 0., -1.])
        inputs[key] = SourceObservation(torch.full((3, 4, 6), index / 4), depth, torch.eye(3), torch.eye(4))
    source = prepare_source(inputs)
    observations = {}
    for key, obs in source.observations.items():
        mask = torch.zeros(4, 6, dtype=torch.bool)
        mask[:, 3:5] = right
        mask[:, :2] = not right
        proposal = InstanceProposal(mask, None, 10, 'Cup', track_id=('scene', key[1], 'session', 10))
        observations[key] = _segmented_observation(key, obs, [proposal], torch.ones(4, 6, dtype=torch.bool))
    return label_source(SegmentedSource(source, observations), source_id='scene', registry=registry)


def cameras(count=1):
    return torch.eye(4).expand(1, count, 4, 4).clone(), torch.eye(3).expand(1, count, 3, 3).clone()


def raw_kwargs(count=1, *, labels=True):
    registry = ObjectRegistry('raw')
    registry.allocate('Cup')
    return dict(input_image=torch.full((1, count, 3, 4, 6), .5),
                input_depth=torch.ones(1, count, 1, 4, 6),
                input_w2c=torch.eye(4).expand(1, count, 4, 4),
                input_intrinsics=torch.eye(3).expand(1, count, 3, 3),
                input_format=['B', 'N', 'C', 'H', 'W'],
                input_labels=torch.ones(1, count, 1, 4, 6, dtype=torch.int64) if labels else None,
                object_registry=registry if labels else None, device='cpu')


class CacheLabelTests(unittest.TestCase):
    def test_packing_sparse_keys_point_alignment_and_save_load(self):
        labeled = labeled_source(((90, 7), (4, 7)), invalid=True)
        kwargs = labeled.cache_kwargs()
        cache = Cache4D(**kwargs, device='cpu')
        self.assertEqual(cache.input_labels.shape, (1, 2, 1, 1, 1, 4, 6))
        self.assertEqual(cache.observation_to_offset, {(4, 7): (0, 0, 0), (90, 7): (1, 0, 0)})
        torch.testing.assert_close(cache.input_labels[0, 0, 0, 0, 0], labeled.labels[4, 7])
        # (1,1), depth=2 under identity camera, unprojects to (2,2,2).
        torch.testing.assert_close(cache.input_points[0, 0, 0, 0, 1, 1], torch.tensor([2., 2., 2.]))
        self.assertEqual(cache.input_labels[0, 0, 0, 0, 0, 1, 1], 1)
        report = cache.check_label_consistency()
        self.assertEqual(report[1]['labeled'], report[1]['renderable'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene'
            save_labeled_source(path, labeled)
            loaded = load_labeled_source(path, labeled.prepared_source)
            restored = Cache4D(**loaded.cache_kwargs(), device='cpu')
            torch.testing.assert_close(cache.input_labels, restored.input_labels)
            restored.check_label_consistency()

    def test_selected_rgb_depth_environment_empty_and_legacy_render(self):
        labeled = labeled_source()
        cache = Cache3D_Base(**labeled.cache_kwargs(), device='cpu')
        before = cache.input_mask.clone(), cache.input_points.clone(), cache.input_labels.clone()
        rgb, selected = cache.render_cache(*cameras(), selected_ids=[1])
        depth, depth_mask = cache.render_cache(*cameras(), render_depth=True, selected_ids=[1])
        self.assertEqual(rgb.shape, (1, 1, 1, 3, 4, 6))
        self.assertEqual(depth.shape, (1, 1, 1, 4, 6))
        torch.testing.assert_close(selected, depth_mask)
        self.assertTrue((selected[..., :2] > 0).all())
        self.assertFalse(selected[..., 3:].any())
        torch.testing.assert_close(depth[depth_mask.squeeze(3).bool()], torch.full_like(depth[depth_mask.squeeze(3).bool()], 2.))
        _, environment = cache.render_cache(*cameras(), selected_ids=[0])
        self.assertFalse(environment[..., :1].any())
        self.assertTrue((environment[..., 3:] > 0).all())
        for render_depth in (True, False):
            _, empty = cache.render_cache(*cameras(), selected_ids=[], render_depth=render_depth)
            self.assertFalse(empty.any())
        legacy = Cache3D_Base(**labeled.prepared_source.cache_kwargs(), device='cpu')
        legacy_rgb, legacy_mask = legacy.render_cache(*cameras())
        rgb_all, mask_all = cache.render_cache(*cameras())
        torch.testing.assert_close(rgb_all, legacy_rgb)
        torch.testing.assert_close(mask_all, legacy_mask)
        for old, current in zip(before, (cache.input_mask, cache.input_points, cache.input_labels)):
            torch.testing.assert_close(old, current)
        for invalid in ([99], [-1], [True]):
            with self.assertRaises(ValueError):
                cache.render_cache(*cameras(), selected_ids=invalid)
        with self.assertRaisesRegex(ValueError, 'labeled'):
            legacy.render_cache(*cameras(), selected_ids=[])

    def test_reliability_and_invalid_depth_never_become_renderable(self):
        kwargs = raw_kwargs()
        kwargs['input_depth'][0, 0, 0, 0, :4] = torch.tensor([float('nan'), float('inf'), 0., -1.])
        reliable = torch.ones(1, 1, 4, 6, dtype=torch.bool)
        reliable[..., 2, 1] = False
        with patch(f'{CACHE_MODULE}.reliable_depth_mask_range_batch', return_value=reliable):
            cache = Cache3D_Base(**kwargs, filter_points_threshold=.5)
        report = cache.check_label_consistency()
        self.assertEqual(report[1], dict(labeled=24, renderable=19))
        # Splatting may cover a destination pixel from a valid neighbor; inspect
        # source validity entering the real warp instead of assuming exact holes.
        from cosmos_predict1.diffusion.inference.cache_3d import forward_warp
        with patch(f'{CACHE_MODULE}.forward_warp', wraps=forward_warp) as warp:
            cache.render_cache(*cameras(), selected_ids=[1])
            mask = warp.call_args.kwargs['mask1']
            self.assertFalse(mask[..., 0, :4].any())
            self.assertFalse(mask[..., 2, 1].any())

    def test_large_ids_reordered_layout_and_validation(self):
        kwargs = raw_kwargs()
        big = 2**60 + 7
        kwargs['object_registry'].next_id = big
        kwargs['object_registry'].allocate('Large')
        kwargs['input_labels'].fill_(big)
        # Deliberately nonstandard channel placement exercises identical RGB/label permutation.
        kwargs['input_image'] = kwargs['input_image'].permute(0, 1, 3, 4, 2)
        kwargs['input_labels'] = kwargs['input_labels'].permute(0, 1, 3, 4, 2)
        kwargs['input_format'] = ['B', 'N', 'H', 'W', 'C']
        cache = Cache3D_Base(**kwargs, weight_dtype=torch.float16)
        self.assertTrue((cache.input_labels == big).all())
        self.assertEqual(cache.input_labels.dtype, torch.int64)
        for bad in (torch.ones(1, 1, 4, 6, 1), torch.full((1, 1, 4, 6, 1), -2),
                    torch.full((1, 1, 4, 6, 1), 99), torch.ones(1, 1, 4, 6, 2, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                Cache3D_Base(**{**kwargs, 'input_labels': bad})
        with self.assertRaisesRegex(ValueError, 'registry'):
            Cache3D_Base(**{**kwargs, 'object_registry': None})

    def test_temporal_windows_static_repeat_and_boundary_slice(self):
        labeled = labeled_source(((4, 7), (90, 7), (100, 7)))
        cache = Cache4D(**labeled.cache_kwargs(), device='cpu')
        _, mask = cache.render_cache(*cameras(2), start_frame_idx=1, selected_ids=[1])
        self.assertEqual(mask.shape[1], 2)
        cache.boundary_mask = torch.zeros_like(cache.input_mask)
        # Capture the boundary grid passed for offset 1, rather than exercising mesh rasterization.
        with patch(f'{CACHE_MODULE}.forward_warp', wraps=__import__(CACHE_MODULE, fromlist=['forward_warp']).forward_warp) as warp:
            cache.render_cache(*cameras(1), start_frame_idx=1, selected_ids=[1])
            self.assertEqual(warp.call_args.kwargs['boundary_mask'].shape, (1, 4, 6))
        for offset, count in ((-1, 1), (2, 2), (90, 1)):
            with self.assertRaises(ValueError):
                cache.render_cache(*cameras(count), start_frame_idx=offset)
        static = Cache3D_Buffer(**labeled_source().cache_kwargs(), device='cpu')
        _, masks = static.render_cache(*cameras(3), start_frame_idx=90, selected_ids=[1])
        torch.testing.assert_close(masks[:, 0], masks[:, 2])

    def test_selector_ranks_after_selection_and_keeps_complementary_views(self):
        kwargs = raw_kwargs(2)
        # View 0 is mostly environment; object is only in view 1.
        kwargs['input_labels'][:, 0] = 0
        selector = Cache3D_BufferSelector(**kwargs, frame_buffer_max=1)
        _, masks = selector.render_cache(*cameras(), selected_ids=[1])
        self.assertTrue(masks.any())
        # Near-full view 0 and a complementary object in view 1 must both survive.
        kwargs = raw_kwargs(2)
        kwargs['input_labels'][:, 0, :, 0, 0] = -1
        kwargs['input_labels'][:, 1] = -1
        kwargs['input_labels'][:, 1, :, 0, 0] = 1
        selector = Cache3D_BufferSelector(**kwargs, frame_buffer_max=2, mask_full_threshold=.8)
        _, masks = selector.render_cache(*cameras(), selected_ids=[1])
        self.assertTrue(masks[:, :, 0].any())
        self.assertTrue(masks[:, :, 1].any())
        _, original_masks = selector.render_cache(*cameras())
        self.assertFalse(original_masks[:, :, 1].any())

    def test_consistency_rejects_mutated_metadata_and_multiview_render_is_deferred(self):
        cache = Cache3D_Base(**labeled_source().cache_kwargs(), device='cpu')
        cache.source_fingerprint = 'other-source'
        with self.assertRaisesRegex(ValueError, 'source/revision'):
            cache.check_label_consistency()
        kwargs = raw_kwargs()
        for key in ('input_image', 'input_depth', 'input_labels'):
            tensor = kwargs[key].unsqueeze(2)
            kwargs[key] = tensor.expand(*tensor.shape[:2], 2, *tensor.shape[3:])
        kwargs['input_w2c'] = torch.eye(4).expand(1, 1, 2, 4, 4)
        kwargs['input_intrinsics'] = torch.eye(3).expand(1, 1, 2, 3, 3)
        kwargs['input_format'] = ['B', 'N', 'V', 'C', 'H', 'W']
        cache = Cache3D_Base(**kwargs)
        cache.check_label_consistency()
        with self.assertRaisesRegex(NotImplementedError, 'V > 1'):
            cache.render_cache(*cameras())

    def test_corrections_merges_source_and_revision_validation(self):
        original = labeled_source()
        cache = Cache3D_Base(**original.cache_kwargs(), device='cpu')
        points = cache.input_points
        updated = labeled_source(registry=original.registry, right=True)
        cache.replace_labels(updated)
        self.assertIs(cache.input_points, points)
        _, selected = cache.render_cache(*cameras(), selected_ids=[1])
        self.assertTrue(selected[..., 4].any())
        self.assertFalse(selected[..., 0].any())
        with self.assertRaisesRegex(ValueError, 'Stale'):
            cache.replace_labels(original)
        merged_registry = deepcopy(updated.registry)
        surviving = merged_registry.allocate('Merged')
        merged_registry.merge_ids(1, surviving)
        merged = label_source(updated.segmented_source, source_id='scene', registry=merged_registry)
        cache.replace_labels(merged)
        _, mask_old = cache.render_cache(*cameras(), selected_ids=[1])
        _, mask_new = cache.render_cache(*cameras(), selected_ids=[surviving])
        torch.testing.assert_close(mask_old, mask_new)
        with self.assertRaisesRegex(ValueError, 'merge'):
            cache.replace_labels(updated)
        different = labeled_source(((5, 7),))
        with self.assertRaisesRegex(ValueError, 'source'):
            cache.replace_labels(different)
        stale = labeled_source()
        stale.labels[4, 7].fill_(0)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            stale.cache_kwargs()

    def test_buffer_updates_keep_every_field_aligned_and_reject_missing_labels(self):
        for size in (1, 2):
            cache = Cache3D_Buffer(**labeled_source().cache_kwargs(), device='cpu', frame_buffer_max=size)
            args = dict(new_image=torch.ones(1, 3, 4, 6), new_depth=torch.ones(1, 1, 4, 6) * 3,
                        new_w2c=torch.eye(4)[None], new_intrinsics=torch.eye(3)[None], depth_alignment=False)
            for labels in (None, torch.zeros(1, 1, 4, 6), torch.full((1, 1, 4, 6), 99)):
                with self.assertRaises(ValueError):
                    cache.update_cache(**args, new_labels=labels)
            for iteration in range(3):
                args['new_depth'][..., 0, 0] = float('nan')
                args['new_image'].fill_(iteration / 2)
                cache.update_cache(**args, new_labels=torch.full((1, 1, 4, 6), iteration % 2, dtype=torch.int64))
                self.assertEqual(cache.input_labels.shape[:4], cache.input_image.shape[:4])
                self.assertFalse(cache.input_mask[0, 0, 0, 0, 0, 0, 0])
                self.assertEqual(cache.input_labels[0, 0, 0, 0, 0, 1, 1], iteration % 2)
                torch.testing.assert_close(cache.input_points[0, 0, 0, 0, 1, 1], torch.tensor([3., 3., 3.]))
                cache.check_label_consistency()
            with self.assertRaisesRegex(ValueError, 'source'):
                cache.replace_labels(labeled_source())
        cache = Cache3D_Buffer(**raw_kwargs(labels=False))
        cache.update_cache(**args)
        self.assertIsNone(cache.input_labels)
        self.assertEqual(cache.input_image.shape[2], 1)

    def test_overlay_and_artifact_adapter_preserve_keys_and_alignment(self):
        labeled = labeled_source(((4, 7), (90, 7)))
        overlays = label_overlays(labeled, rgb_range=(-1, 1))
        self.assertEqual(overlays[4, 7].shape, (3, 4, 6))
        self.assertEqual(overlays[4, 7].dtype, torch.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene'
            save_labeled_source(path, labeled)
            kwargs = labeled.prepared_source.cache_kwargs()
            attached = load_cache_label_kwargs(path, kwargs)
            cache = Cache4D(**{**kwargs, **attached}, device='cpu')
            self.assertEqual(cache.observation_keys, ((4, 7), (90, 7)))
            torch.testing.assert_close(cache.input_labels[0, 1, 0, 0, 0], labeled.labels[90, 7])
            kwargs['input_image'] = kwargs['input_image'] + .01
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_cache_label_kwargs(path, kwargs)


class PersistentEntryPointTests(unittest.TestCase):
    def test_seed_and_continue_labeled_image_and_clip(self):
        # Execute the production methods without importing CUDA model dependencies.
        path = Path(__file__).resolve().parents[1] / 'cosmos_predict1/diffusion/inference/gen3c_persistent.py'
        model = next(node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.ClassDef) and node.name == 'Gen3cPersistentModel')
        methods = [node for node in model.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('seed_model_from_values', 'inference_on_cameras')]
        namespace = dict(np=np, torch=torch, Cache3D_Buffer=Cache3D_Buffer, Cache4D=Cache4D,
                         load_cache_label_kwargs=load_cache_label_kwargs, device_with_rank=lambda value: value,
                         log=Mock(), is_rank0=lambda: False, os=__import__('os'))
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
        for keys in (((4, 7),), ((4, 7), (90, 7), (100, 7))):
            labeled = labeled_source(keys, invalid=True)
            observations = [labeled.prepared_source.observations[key] for key in keys]
            first = observations[0]
            namespace['_predict_moge_depth'] = Mock(return_value=(
                first.rgb[None, None], first.depth[None, None, None], first.geometry_mask[None, None, None],
                first.w2c[None, None], first.intrinsics[None, None],
            ))
            namespace['_predict_moge_depth_from_tensor'] = Mock(side_effect=AssertionError('Unexpected depth inference'))
            with tempfile.TemporaryDirectory() as directory:
                labels_path = Path(directory) / 'scene'
                save_labeled_source(labels_path, labeled)
                args = types.SimpleNamespace(height=4, width=6, noise_aug_strength=0., filter_points_threshold=1.,
                                             foreground_masking=False, save_buffer=False, video_save_name='test',
                                             video_save_folder=directory, prompt='Scene', negative_prompt='',
                                             disable_prompt_upsampler=True)
                instance = types.SimpleNamespace(args=args, H=4, W=6, frame_buffer_max=2, generator=None,
                                                 device_with_rank='cpu', moge_model=object(), sample_n_frames=2)
                n = len(keys)
                images = torch.stack([(obs.rgb + 1) / 2 for obs in observations]).permute(0, 2, 3, 1).numpy()
                # The method imports torchvision only for optional seeding-image resizing.
                torchvision = types.ModuleType('torchvision')
                transforms = types.ModuleType('torchvision.transforms')
                functional = types.ModuleType('torchvision.transforms.functional')
                with patch.dict('sys.modules', {'torchvision': torchvision, 'torchvision.transforms': transforms,
                                                'torchvision.transforms.functional': functional}):
                    namespace['seed_model_from_values'](
                        instance, images, None if n == 1 else torch.stack([obs.depth for obs in observations]).numpy(),
                        np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)), np.ones((n, 2), dtype=np.float32),
                        np.zeros((n, 2), dtype=np.float32), np.tile([[6, 4]], (n, 1)),
                        None if n == 1 else torch.stack([obs.geometry_mask for obs in observations]).numpy(),
                        labels_path=labels_path, observation_keys=keys, selected_ids=[1],
                    )
                instance.cache.device = 'cpu'
                instance.cache.check_label_consistency()
                self.assertEqual(instance.cache.observation_keys, keys)
                instance.prepare_camera_for_inference = lambda poses, intrinsics, **kwargs: (poses, intrinsics)
                instance.pipeline = types.SimpleNamespace(generate=Mock(return_value=(np.zeros((2, 4, 6, 3), dtype=np.uint8), '')))
                before = instance.cache.input_points.clone()
                with patch.object(instance.cache, 'update_cache', side_effect=AssertionError('Generated update')):
                    result = namespace['inference_on_cameras'](instance, *cameras(3), fps=24)
                self.assertEqual(result['video'].shape, (1, 3, 3, 4, 6))
                self.assertEqual(instance.pipeline.generate.call_count, 2)
                for call in instance.pipeline.generate.call_args_list:
                    coverage = call.kwargs['rendered_warp_masks']
                    self.assertTrue(coverage[..., :2].any())
                    self.assertFalse(coverage[..., 3:].any())
                torch.testing.assert_close(before, instance.cache.input_points)
                namespace['_predict_moge_depth_from_tensor'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
