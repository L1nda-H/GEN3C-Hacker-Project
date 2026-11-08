"""CPU contract tests for promptable segmentation, cleanup, and appearance."""

import unittest

import torch

from cosmos_predict1.diffusion.inference.instance_segmentation import (
    AppearanceEncoderAdapter,
    InstanceProposal,
    SegmenterAdapter,
    SegmentPrediction,
    SegmentPrompts,
    SourceObservation,
    clean_instance_proposals,
    prepare_source,
    segment_source,
)


def observation(h=6, w=8):
    return SourceObservation(
        torch.ones(3, h, w), torch.ones(h, w), torch.eye(3), torch.eye(4),
    )


def rectangle(top, left, bottom, right, h=6, w=8):
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[top:bottom, left:right] = True
    return mask


def encoder():
    return AppearanceEncoderAdapter(lambda image, mask: image[:, mask].mean(dim=1))


class SegmentInstancesTests(unittest.TestCase):
    def test_keys_empty_results_and_source_are_preserved(self):
        source = prepare_source({(3, 9): observation(), (7, 2): observation(4, 5)})
        calls = []

        def backend(image, prompts):
            calls.append(image.shape)
            self.assertIsNone(prompts.boxes)  # Automatic proposal request.
            return SegmentPrediction()

        result = segment_source(source, SegmenterAdapter(backend), encoder(), rgb_range=(0, 1))
        self.assertIs(result.prepared_source, source)
        self.assertEqual(set(result.observations), set(source.observations))
        self.assertEqual(calls, [torch.Size([6, 8, 3]), torch.Size([4, 5, 3])])
        for segmented in result.observations.values():
            self.assertEqual(segmented.instances, ())
            self.assertTrue(segmented.unassigned_mask.all())
            self.assertFalse(segmented.ambiguous_mask.any())
            self.assertFalse(segmented.environment_mask.any())

    def test_backend_rgb_conversion_prompts_and_mask_remapping(self):
        rgb = torch.zeros(3, 4, 6)
        rgb[0] = -1
        rgb[2] = 1
        boxes = torch.tensor([[1., 1., 4., 3.]])
        points = torch.tensor([[2., 1.]])
        prompts = SegmentPrompts(boxes, points, torch.tensor([1]))
        predicted = torch.tensor([[False, True, False], [False, True, False]])

        def backend(image, mapped):
            self.assertEqual(image.shape, (2, 3, 3))
            self.assertEqual(image.dtype, torch.uint8)
            torch.testing.assert_close(image[0, 0], torch.tensor([0, 128, 255], dtype=torch.uint8))
            torch.testing.assert_close(mapped.boxes, (boxes + 0.5) / 2 - 0.5)
            torch.testing.assert_close(mapped.points, (points + 0.5) / 2 - 0.5)
            torch.testing.assert_close(mapped.point_labels, torch.tensor([1]))
            return SegmentPrediction((InstanceProposal(predicted.float(), 0.8),), ~predicted)

        adapter = SegmenterAdapter(backend, input_size=(2, 3))
        result = adapter.segment(rgb, rgb_range=(-1, 1), prompts=prompts)
        expected = predicted.repeat_interleave(2, 0).repeat_interleave(2, 1)
        torch.testing.assert_close(result.proposals[0].mask, expected)
        torch.testing.assert_close(result.environment_mask, ~expected)
        torch.testing.assert_close(prompts.boxes, boxes)

    def test_noninteger_backend_mask_remapping(self):
        mask = torch.eye(5, dtype=torch.bool)
        adapter = SegmenterAdapter(lambda image, prompts: SegmentPrediction((InstanceProposal(mask, 1),)),
                                   input_size=(5, 5))
        result = adapter.segment(torch.zeros(3, 3, 3), rgb_range=(0, 1))
        indices = torch.tensor([0, 2, 4])
        torch.testing.assert_close(result.proposals[0].mask, mask[indices[:, None], indices[None, :]])

    def test_alternate_backend_layouts_and_ranges(self):
        def segment_backend(image, prompts):
            self.assertEqual(image.shape, (3, 6, 8))
            torch.testing.assert_close(image, torch.ones(3, 6, 8))
            return SegmentPrediction()

        SegmenterAdapter(segment_backend, input_layout="CHW", input_range="-1_1").segment(
            observation().rgb, rgb_range=(0, 1),
        )

        def appearance_backend(image, mask):
            self.assertEqual(image.shape, (2, 2, 3))
            self.assertEqual(image.dtype, torch.uint8)
            self.assertTrue((image == 255).all())
            return image[mask].float().mean(dim=0)

        feature = AppearanceEncoderAdapter(appearance_backend, input_layout="HWC", input_range="uint8").encode(
            observation().rgb, rectangle(1, 1, 3, 3), rgb_range=(0, 1),
        )
        torch.testing.assert_close(feature.norm(), torch.tensor(1.))

    def test_all_rejected_proposals_remain_unassigned(self):
        prediction = SegmentPrediction((InstanceProposal(rectangle(1, 1, 3, 3), 0.2),))
        result = segment_source(prepare_source({(0, 0): observation()}),
                                SegmenterAdapter(lambda *args: prediction), encoder(),
                                rgb_range=(0, 1), min_confidence=0.8).observations[0, 0]
        self.assertEqual(result.instances, ())
        self.assertTrue(result.unassigned_mask.all())
        self.assertFalse(result.environment_mask.any())

    def test_same_class_instances_invalid_depth_and_environment(self):
        obs = observation()
        obs.depth[:, :3] = float("nan")
        first = rectangle(0, 0, 3, 3)
        second = rectangle(3, 5, 6, 8)
        prediction = SegmentPrediction((InstanceProposal(first, 0.9, "cup"),
                                        InstanceProposal(second, 0.8, "cup")))
        source = prepare_source({(4, 7): obs})
        confirmed = rectangle(0, 0, 6, 4)
        result = segment_source(source, SegmenterAdapter(lambda *args: prediction), encoder(),
                                rgb_range=(0, 1), environment_masks={(4, 7): confirmed}).observations[4, 7]
        self.assertEqual(len(result.instances), 2)
        self.assertEqual({i.local_id for i in result.instances}, {1, 2})
        for instance in result.instances:
            self.assertEqual((instance.frame_idx, instance.view_id), (4, 7))
            self.assertIsNone(instance.track_id)
            self.assertEqual(instance.class_name, "cup")
            torch.testing.assert_close(instance.appearance.norm(), torch.tensor(1.))
        torch.testing.assert_close(result.instances[0].mask, first)
        self.assertFalse(source.observations[4, 7].geometry_mask[first].any())
        torch.testing.assert_close(result.environment_mask, confirmed & ~first & ~second)
        torch.testing.assert_close(result.unassigned_mask, ~confirmed & ~first & ~second)
        self.assertTrue(result.unassigned_mask.any())
        self.assertIsNone(source.observations[4, 7].labels)

    def test_duplicate_suppression_prefers_confidence_and_ignores_class(self):
        mask = rectangle(0, 0, 4, 5)
        jittered = mask.clone()
        jittered[0, 0] = False
        jittered[4, 0] = True
        low = InstanceProposal(mask, 0.6, "cup")
        high = InstanceProposal(jittered, 0.9, "bowl")
        kept, ambiguity = clean_instance_proposals([low, high])
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0], high)
        self.assertFalse(ambiguity.any())

    def test_contained_objects_survive_even_with_high_iou(self):
        outer = torch.ones(20, 20, dtype=torch.bool)
        inner = outer.clone()
        inner[0, 0] = False
        kept, ambiguity = clean_instance_proposals([InstanceProposal(outer, 0.9), InstanceProposal(inner, 0.8)])
        self.assertEqual(len(kept), 2)
        torch.testing.assert_close(ambiguity, inner)

    def test_overlap_uncertainty_and_environment_exclusion(self):
        a, b = rectangle(0, 0, 4, 4), rectangle(2, 2, 6, 6)
        prediction = SegmentPrediction((InstanceProposal(a, 1), InstanceProposal(b, 1)), torch.ones_like(a))
        result = segment_source(prepare_source({(0, 0): observation()}),
                                SegmenterAdapter(lambda *args: prediction), encoder(), rgb_range=(0, 1))
        segmented = result.observations[0, 0]
        torch.testing.assert_close(segmented.ambiguous_mask, a & b)
        torch.testing.assert_close(segmented.environment_mask, ~(a | b))
        self.assertFalse(segmented.unassigned_mask.any())
        torch.testing.assert_close(segmented.instances[0].mask, a)
        torch.testing.assert_close(segmented.instances[1].mask, b)

    def test_quality_filters_and_optional_boundary_band(self):
        mask = rectangle(0, 0, 5, 5)
        kept, ambiguity = clean_instance_proposals([
            InstanceProposal(torch.zeros_like(mask), 1),
            InstanceProposal(rectangle(5, 7, 6, 8), 1),
            InstanceProposal(rectangle(0, 5, 6, 8), 0.1),
            InstanceProposal(mask, 0.8),
        ], min_confidence=0.5, min_area=2, boundary_width=1)
        self.assertEqual(len(kept), 1)
        torch.testing.assert_close(ambiguity, mask & ~rectangle(1, 1, 4, 4))

    def test_masked_crops_remove_background_and_normalize_features(self):
        mask = rectangle(1, 2, 5, 6)
        mask[2, 3] = False  # Background inside the bounding box.
        rgb = torch.rand(3, 6, 8)
        changed = rgb.clone()
        changed[:, ~mask] = 1
        captured = []

        def backend(crop, crop_mask):
            self.assertEqual(crop.shape, (3, 4, 4))
            self.assertFalse(crop[:, ~crop_mask].any())
            captured.append(crop.clone())
            return crop[:, crop_mask].mean(dim=1) * 5

        adapter = AppearanceEncoderAdapter(backend)
        first = adapter.encode(rgb, mask, rgb_range=(0, 1))
        second = adapter.encode(changed, mask, rgb_range=(0, 1))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(captured[0], captured[1])
        torch.testing.assert_close(first.norm(), torch.tensor(1.))

    def test_prompt_routing_and_no_cross_image_identity_claim(self):
        source = prepare_source({(2, 5): observation(), (3, 5): observation()})
        calls = []

        def backend(image, prompts):
            calls.append(prompts.points)
            return SegmentPrediction((InstanceProposal(rectangle(1, 1, 3, 3), 1),))

        result = segment_source(source, SegmenterAdapter(backend), encoder(), rgb_range=(0, 1),
                                prompts={(3, 5): SegmentPrompts(points=torch.tensor([[1., 2.]]),
                                                               point_labels=torch.tensor([1]))})
        self.assertIsNone(calls[0])
        torch.testing.assert_close(calls[1], torch.tensor([[1., 2.]]))
        self.assertTrue(all(v.instances[0].track_id is None for v in result.observations.values()))

    def test_invalid_backend_masks_confidences_rgb_and_prompts(self):
        rgb = torch.zeros(3, 6, 8)
        for proposal in (InstanceProposal(torch.ones(2, 3), 1),
                         InstanceProposal(torch.full((6, 8), float("nan")), 1),
                         InstanceProposal(torch.ones(6, 8), float("nan")),
                         InstanceProposal(torch.ones(6, 8), 1.1)):
            adapter = SegmenterAdapter(lambda *args: SegmentPrediction((proposal,)))
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                adapter.segment(rgb, rgb_range=(0, 1))
        adapter = SegmenterAdapter(lambda *args: SegmentPrediction())
        with self.assertRaisesRegex(ValueError, "rgb_range"):
            adapter.segment(rgb - 1, rgb_range=(0, 1))
        for prompts in (SegmentPrompts(points=torch.tensor([[1., 2.]])),
                        SegmentPrompts(points=torch.tensor([[99., 2.]]), point_labels=torch.tensor([1])),
                        SegmentPrompts(boxes=torch.tensor([[4., 1., 2., 3.]])),
                        SegmentPrompts(point_labels=torch.tensor([1]))):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                adapter.segment(rgb, rgb_range=(0, 1), prompts=prompts)

    def test_embedding_validation_and_quality_settings_for_empty_results(self):
        mask = rectangle(1, 1, 3, 3)
        for value in (torch.zeros(3), torch.tensor([float("nan")]), torch.ones(1, 3), torch.ones(3).long()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AppearanceEncoderAdapter(lambda *args: value).encode(observation().rgb, mask, rgb_range=(0, 1))
        source = prepare_source({(0, 0): observation()})
        adapter = SegmenterAdapter(lambda *args: SegmentPrediction())
        for kwargs in (dict(min_area=0), dict(min_confidence=2), dict(duplicate_iou=0), dict(boundary_width=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                segment_source(source, adapter, encoder(), rgb_range=(0, 1), **kwargs)
        with self.assertRaisesRegex(ValueError, "unknown observation"):
            segment_source(source, adapter, encoder(), rgb_range=(0, 1), prompts={(9, 9): SegmentPrompts()})
        with self.assertRaisesRegex(ValueError, "Confirmed environment"):
            segment_source(source, adapter, encoder(), rgb_range=(0, 1),
                           environment_masks={(0, 0): torch.ones(6, 8)})

    def test_source_requires_fixed_embedding_dimension(self):
        calls = []

        def inconsistent_encoder(*args):
            calls.append(None)
            return torch.ones(len(calls) + 1)

        source = prepare_source({(0, 0): observation(), (1, 0): observation()})
        prediction = SegmentPrediction((InstanceProposal(rectangle(1, 1, 3, 3), 1),))
        with self.assertRaisesRegex(ValueError, "fixed dimension"):
            segment_source(source, SegmenterAdapter(lambda *args: prediction),
                           AppearanceEncoderAdapter(inconsistent_encoder), rgb_range=(0, 1))


if __name__ == "__main__":
    unittest.main()
