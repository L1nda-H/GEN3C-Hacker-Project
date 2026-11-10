"""Mocked checks of the official SAM image API boundary, without model downloads."""

import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.instance_segmentation import (
    InstanceObservation,
    SAM21_MODELS,
    SAM21Segmenter,
    SegmentPrompts,
    SourceObservation,
    prepare_source,
    segment_source,
)


def observation(h=6, w=8):
    return SourceObservation(torch.ones(3, h, w), torch.ones(h, w), torch.eye(3), torch.eye(4))


def rectangle(top, left, bottom, right, h=6, w=8):
    mask = np.zeros((h, w), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def record(mask, quality=0.9, stability=0.98):
    return dict(segmentation=mask, predicted_iou=quality, stability_score=stability)


def prediction(masks, qualities=None):
    masks = np.stack(masks).astype(np.float32)
    qualities = np.array(qualities if qualities is not None else [0.9] * len(masks))
    return masks, qualities, np.zeros((len(masks), 256, 256), dtype=np.float32)


def wrapper(records=(), responses=()):
    predictor = Mock()
    predictor.predict.side_effect = list(responses)
    generator = Mock()
    generator.generate.return_value = list(records)
    return SAM21Segmenter(predictor, generator)


class SAM21SegmenterTests(unittest.TestCase):
    def test_automatic_rgb_conversion_and_quality_metadata(self):
        rgb = torch.zeros(3, 6, 8)
        rgb[0] = -1
        rgb[2] = 1
        mask = rectangle(1, 1, 4, 5)
        segmenter = wrapper([record(mask)])
        result = segmenter.segment(rgb, rgb_range=(-1, 1))
        image = segmenter.automatic_mask_generator.generate.call_args.args[0]
        self.assertEqual(image.shape, (6, 8, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[0, 0], [0, 128, 255])
        self.assertTrue(image.flags.c_contiguous)
        segmenter.predictor.set_image.assert_not_called()
        self.assertEqual(len(result.proposals), 1)
        candidate = result.proposals[0]
        self.assertEqual(candidate.local_id, 1)
        self.assertEqual(candidate.confidence, 0.9)
        self.assertEqual(candidate.stability_score, 0.98)
        self.assertIsNone(candidate.track_id)
        torch.testing.assert_close(candidate.mask, torch.from_numpy(mask))

    def test_prompt_groups_use_prepared_coordinates_and_single_masks(self):
        a, b = rectangle(0, 0, 3, 3), rectangle(3, 5, 6, 8)
        segmenter = wrapper(responses=[prediction([a]), prediction([b])])
        points, labels = torch.tensor([[1., 1.], [2., 2.]]), torch.tensor([1, 0])
        box = torch.tensor([5., 3., 7., 5.])
        result = segmenter.segment(observation().rgb, rgb_range=(0, 1), prompts=[
            SegmentPrompts(8, points=points, point_labels=labels, name="Cup"), SegmentPrompts(3, box=box),
        ])
        segmenter.predictor.set_image.assert_called_once()
        segmenter.automatic_mask_generator.generate.assert_not_called()
        first, second = [call.kwargs for call in segmenter.predictor.predict.call_args_list]
        np.testing.assert_array_equal(first['point_coords'], points.numpy())
        np.testing.assert_array_equal(first['point_labels'], labels.numpy())
        np.testing.assert_array_equal(second['box'], box.numpy())
        self.assertFalse(first['multimask_output'])
        self.assertFalse(first['return_logits'])
        self.assertTrue(first['normalize_coords'])
        self.assertEqual([p.local_id for p in result.proposals], [8, 3])
        self.assertEqual(result.proposals[0].name, 'Cup')

    def test_one_alternative_per_object_fixes_containment_regression(self):
        outer = np.ones((20, 20), dtype=bool)
        inner = outer.copy()
        inner[0, 0] = False
        segmenter = wrapper(responses=[prediction([outer, inner], [0.8, 0.95])])
        result = segment_source(prepare_source({(2, 7): observation(20, 20)}), segmenter,
                                rgb_range=(0, 1), prompts={(2, 7): [SegmentPrompts(
                                    9, box=torch.tensor([0., 0., 19., 19.]), multimask_output=True,
                                )]})
        image = result.observations[2, 7]
        self.assertEqual(len(image.instances), 1)
        self.assertEqual(image.instances[0].local_id, 9)
        torch.testing.assert_close(image.instances[0].mask, torch.from_numpy(inner))
        self.assertFalse(image.ambiguous_mask.any())
        self.assertEqual(segmenter.predictor.predict.call_args.kwargs['multimask_output'], True)

    def test_different_contained_objects_remain_separate(self):
        outer = rectangle(0, 0, 6, 8)
        inner = rectangle(2, 2, 4, 4)
        segmenter = wrapper(responses=[prediction([outer]), prediction([inner])])
        result = segment_source(prepare_source({(0, 0): observation()}), segmenter, rgb_range=(0, 1),
                                prompts={(0, 0): [SegmentPrompts(1, box=torch.tensor([0., 0., 7., 5.])),
                                                  SegmentPrompts(2, box=torch.tensor([2., 2., 3., 3.]))]})
        image = result.observations[0, 0]
        self.assertEqual(len(image.instances), 2)
        torch.testing.assert_close(image.ambiguous_mask, torch.from_numpy(inner))

    def test_generator_postprocessing_is_not_reimplemented(self):
        # Treat the official generator's result as candidates, including unresolved overlap.
        outer = rectangle(0, 0, 6, 8)
        inner = outer.copy()
        inner[0, 0] = False
        result = segment_source(prepare_source({(0, 0): observation()}), wrapper([record(outer), record(inner)]),
                                rgb_range=(0, 1)).observations[0, 0]
        self.assertEqual(len(result.instances), 2)
        torch.testing.assert_close(result.ambiguous_mask, torch.from_numpy(inner))

    def test_empty_hypotheses_are_skipped_before_alternative_selection(self):
        empty, mask = rectangle(0, 0, 0, 0), rectangle(1, 1, 3, 3)
        segmenter = wrapper(responses=[prediction([empty, mask], [0.99, 0.8])])
        result = segmenter.segment(observation().rgb, rgb_range=(0, 1), prompts=[
            SegmentPrompts(1, box=torch.tensor([0., 0., 3., 3.]), multimask_output=True),
        ])
        torch.testing.assert_close(result.proposals[0].mask, torch.from_numpy(mask))
        segmenter = wrapper([record(empty)])
        self.assertEqual(segmenter.segment(observation().rgb, rgb_range=(0, 1)).proposals, ())

    def test_observation_keys_empty_results_and_variable_sizes(self):
        source = prepare_source({(3, 9): observation(), (7, 2): observation(4, 5)})
        segmenter = wrapper()
        result = segment_source(source, segmenter, rgb_range=(0, 1))
        self.assertIs(result.prepared_source, source)
        self.assertEqual(set(result.observations), set(source.observations))
        self.assertEqual([c.args[0].shape for c in segmenter.automatic_mask_generator.generate.call_args_list],
                         [(6, 8, 3), (4, 5, 3)])
        for image in result.observations.values():
            self.assertEqual(image.instances, ())
            self.assertTrue(image.unassigned_mask.all())
            self.assertFalse(image.ambiguous_mask.any())
            self.assertFalse(image.environment_mask.any())

    def test_keyed_prompts_and_explicit_empty_list(self):
        source = prepare_source({(0, 1): observation(), (2, 1): observation(), (3, 1): observation()})
        mask = rectangle(1, 1, 4, 4)
        segmenter = wrapper(responses=[prediction([mask])])
        result = segment_source(source, segmenter, rgb_range=(0, 1), prompts={
            (2, 1): [SegmentPrompts(7, box=torch.tensor([1., 1., 3., 3.]))], (3, 1): [],
        })
        segmenter.automatic_mask_generator.generate.assert_called_once()
        segmenter.predictor.set_image.assert_called_once()
        self.assertEqual(result.observations[2, 1].instances[0].name, 'Object 7')
        self.assertEqual(result.observations[3, 1].instances, ())

    def test_depth_independence_environment_confirmation_and_no_mutation(self):
        obs = observation()
        obs.depth[:, :3] = float('nan')
        mask = rectangle(0, 0, 3, 3)
        source = prepare_source({(4, 7): obs})
        confirmed = torch.ones(6, 8, dtype=torch.bool)
        before = confirmed.clone()
        result = segment_source(source, wrapper([record(mask)]), rgb_range=(0, 1),
                                environment_masks={(4, 7): confirmed}).observations[4, 7]
        self.assertFalse(source.observations[4, 7].geometry_mask[torch.from_numpy(mask)].any())
        torch.testing.assert_close(result.instances[0].mask, torch.from_numpy(mask))
        torch.testing.assert_close(result.environment_mask, ~torch.from_numpy(mask))
        torch.testing.assert_close(confirmed, before)
        self.assertFalse(result.unassigned_mask.any())
        self.assertIsNone(source.observations[4, 7].labels)
        self.assertIsNone(result.instances[0].track_id)
        unconfirmed = segment_source(source, wrapper([record(mask)]), rgb_range=(0, 1)).observations[4, 7]
        self.assertFalse(unconfirmed.environment_mask.any())
        torch.testing.assert_close(unconfirmed.unassigned_mask, ~torch.from_numpy(mask))

    def test_optional_boundary_uncertainty(self):
        mask = rectangle(0, 0, 5, 5)
        result = segment_source(prepare_source({(0, 0): observation()}), wrapper([record(mask)]),
                                rgb_range=(0, 1), boundary_width=1).observations[0, 0]
        torch.testing.assert_close(result.ambiguous_mask, torch.from_numpy(mask & ~rectangle(1, 1, 4, 4)))

    def test_refinement_replaces_mask_preserves_explicit_local_id_and_name(self):
        before, after = rectangle(0, 0, 3, 3), rectangle(1, 1, 4, 4)
        source = prepare_source({(0, 0): observation()})
        segmenter = wrapper(responses=[prediction([before]), prediction([after])])
        group = SegmentPrompts(42, points=torch.tensor([[1., 1.]]), point_labels=torch.tensor([1]), name='Mug')
        first = segment_source(source, segmenter, rgb_range=(0, 1), prompts={(0, 0): [group]})
        revised = replace(group, points=torch.tensor([[1., 1.], [2., 2.]]), point_labels=torch.tensor([1, 1]))
        second = segment_source(source, segmenter, rgb_range=(0, 1), prompts={(0, 0): [revised]})
        old, new = first.observations[0, 0].instances[0], second.observations[0, 0].instances[0]
        self.assertEqual((old.local_id, old.name), (new.local_id, new.name))
        torch.testing.assert_close(old.mask, torch.from_numpy(before))
        torch.testing.assert_close(new.mask, torch.from_numpy(after))
        self.assertFalse(torch.equal(old.mask, new.mask))

    def test_invalid_rgb_prompts_and_duplicate_ids(self):
        rgb = observation().rgb
        segmenter = wrapper()
        with self.assertRaisesRegex(ValueError, 'rgb_range'):
            segmenter.segment(rgb + 1, rgb_range=(0, 1))
        bad_groups = [
            SegmentPrompts(0, box=torch.tensor([0., 0., 2., 2.])),
            SegmentPrompts(1),
            SegmentPrompts(1, points=torch.tensor([[1., 2.]])),
            SegmentPrompts(1, points=torch.tensor([[99., 2.]]), point_labels=torch.tensor([1])),
            SegmentPrompts(1, points=torch.tensor([[1., 2.]]), point_labels=torch.tensor([2])),
            SegmentPrompts(1, box=torch.tensor([4., 1., 2., 3.])),
            SegmentPrompts(1, point_labels=torch.tensor([1])),
        ]
        for group in bad_groups:
            with self.subTest(group=group), self.assertRaises(ValueError):
                segmenter.segment(rgb, rgb_range=(0, 1), prompts=[group])
        good = SegmentPrompts(1, box=torch.tensor([0., 0., 2., 2.]))
        with self.assertRaisesRegex(ValueError, 'unique local_id'):
            segmenter.segment(rgb, rgb_range=(0, 1), prompts=[good, good])
        segmenter.predictor.set_image.assert_not_called()

    def test_malformed_sam_outputs_are_rejected(self):
        mask = rectangle(1, 1, 3, 3)
        bad_records = [record(np.ones((2, 3))), record(np.full((6, 8), float('nan'))),
                       record(np.full((6, 8), 0.4)), record(mask, float('nan'))]
        for invalid in bad_records:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                wrapper([invalid]).segment(observation().rgb, rgb_range=(0, 1))
        bad_predictions = [prediction([np.ones((2, 3))]),
                           (np.ones((1, 6, 8)), np.ones((1, 1)), None),
                           prediction([mask], [float('nan')])]
        for invalid in bad_predictions:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                wrapper(responses=[invalid]).segment(observation().rgb, rgb_range=(0, 1), prompts=[
                    SegmentPrompts(1, box=torch.tensor([0., 0., 2., 2.])),
                ])

    def test_source_validation_and_prompt_only_wrapper(self):
        source = prepare_source({(0, 0): observation()})
        segmenter = wrapper()
        with self.assertRaisesRegex(ValueError, 'unknown observation'):
            segment_source(source, segmenter, rgb_range=(0, 1), prompts={(9, 9): []})
        with self.assertRaisesRegex(ValueError, 'Confirmed environment'):
            segment_source(source, segmenter, rgb_range=(0, 1),
                           environment_masks={(0, 0): torch.ones(6, 8)})
        with self.assertRaisesRegex(ValueError, 'boundary_width'):
            segment_source(source, segmenter, rgb_range=(0, 1), boundary_width=-1)
        with self.assertRaisesRegex(ValueError, 'Automatic discovery requires'):
            SAM21Segmenter(Mock()).segment(observation().rgb, rgb_range=(0, 1))

    def test_optional_video_quality_and_scoped_track_contract(self):
        instance = InstanceObservation(3, 2, 1, torch.ones(6, 8, dtype=torch.bool), None, 'Object 1',
                                       track_id=('scene', 2, 'session', 17))
        self.assertIsNone(instance.confidence)
        self.assertEqual(instance.track_id, ('scene', 2, 'session', 17))
        self.assertFalse(hasattr(instance, 'appearance'))

    def test_factory_builds_matching_config_and_delegates_generator_options(self):
        model = object()
        build, predictor_cls, generator_cls = Mock(return_value=model), Mock(), Mock()
        modules = {
            'sam2': types.ModuleType('sam2'),
            'sam2.build_sam': types.SimpleNamespace(build_sam2=build),
            'sam2.sam2_image_predictor': types.SimpleNamespace(SAM2ImagePredictor=predictor_cls),
            'sam2.automatic_mask_generator': types.SimpleNamespace(SAM2AutomaticMaskGenerator=generator_cls),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            checkpoint = Path(directory) / 'sam2.1_hiera_small.pt'
            checkpoint.touch()
            options = dict(pred_iou_thresh=0.8, stability_score_thresh=0.95, box_nms_thresh=0.7)
            result = SAM21Segmenter.from_checkpoint(checkpoint, model_size='small',
                                                   automatic_mask_options=options)
            build.assert_called_once_with(SAM21_MODELS['small'][0], str(checkpoint), device='cuda', mode='eval')
            predictor_cls.assert_called_once_with(model)
            generator_cls.assert_called_once_with(model, output_mode='binary_mask', **options)
            self.assertIs(result.predictor, predictor_cls.return_value)
            self.assertNotIn('output_mode', options)
            with self.assertRaisesRegex(ValueError, 'output_mode'):
                SAM21Segmenter.from_checkpoint(checkpoint, automatic_mask_options={'output_mode': 'coco_rle'})
        with self.assertRaisesRegex(ValueError, 'model_size'):
            SAM21Segmenter.from_checkpoint('unused.pt', model_size='sam1')
        with self.assertRaises(FileNotFoundError):
            SAM21Segmenter.from_checkpoint('/no/checkpoint/here.pt')

    def test_missing_sam_package_has_install_instructions(self):
        with tempfile.NamedTemporaryFile(suffix='.pt') as checkpoint, patch.dict(sys.modules, {'sam2': None}):
            with self.assertRaisesRegex(ImportError, r'pip install -r requirements\.txt'):
                SAM21Segmenter.from_checkpoint(checkpoint.name)


if __name__ == '__main__':
    unittest.main()
