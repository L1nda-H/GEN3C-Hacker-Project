"""CPU contracts for static composite selection, placement, and child ownership."""

import unittest
from unittest.mock import patch

import torch

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base, Cache3D_Buffer
from cosmos_predict1.diffusion.inference.cache3d_composite import Cache3D_Composite
from cosmos_predict1.diffusion.inference.instance_labeling import ObjectRegistry, label_source
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    InstanceProposal, SegmentedSource, SourceObservation, _segmented_observation, prepare_source,
)


def child(source_id, *, count=1, axis='N', labels=True, cache_type=Cache3D_Base):
    registry = ObjectRegistry(source_id)
    registry.allocate('Cup')
    registry.allocate('Plate')
    return cache_type(
        input_image=torch.arange(18).reshape(1, 1, 3, 2, 3).expand(1, count, 3, 2, 3) / 18,
        input_depth=torch.ones(1, count, 1, 2, 3),
        input_w2c=torch.eye(4).expand(1, count, 4, 4),
        input_intrinsics=torch.eye(3).expand(1, count, 3, 3),
        input_format=['B', axis, 'C', 'H', 'W'],
        input_labels=torch.tensor([0, 1, 2, -1, 1, 0]).reshape(1, 1, 1, 2, 3).expand(
            1, count, 1, 2, 3) if labels else None,
        object_registry=registry if labels else None, device='cpu',
    )


def translation(x, y=0, z=0):
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, 3] = torch.tensor([x, y, z])
    return matrix


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.environment = child('room')
        self.object = child('objects')
        self.composite = Cache3D_Composite()
        self.add_source('room', self.environment, is_environment=True)
        self.add_source('objects', self.object)

    def add_source(self, *args, **kwargs):
        return self.composite.add_source(*args, source_w2c=torch.eye(4), source_intrinsics=torch.eye(3), **kwargs)

    def samples(self):
        return {(s.source_id, s.instance_id): s for s in self.composite._iter_render_inputs()}

    def test_environment_namespaces_replacement_and_empty_selection(self):
        self.assertEqual(set(self.samples()), {('room', 0)})
        self.assertEqual(self.composite.selections['room'], {0})
        self.composite.set_selection('room', [1])
        self.composite.set_selection('objects', [1, 2, 1])
        samples = self.samples()
        self.assertEqual(set(samples), {('room', 0), ('room', 1), ('objects', 1), ('objects', 2)})
        self.composite.set_transform('objects', translation(10), instance_id=1)
        shifted = self.samples()
        torch.testing.assert_close(shifted['room', 1].points, samples['room', 1].points)
        torch.testing.assert_close(shifted['objects', 1].points, samples['objects', 1].points + torch.tensor([10., 0, 0]))
        self.composite.set_selection('objects', [])
        self.composite.set_selection('room', [])
        self.assertEqual(set(self.samples()), {('room', 0)})

    def test_selection_validation_is_atomic(self):
        for source, ids in [('missing', [1]), ('objects', [0]), ('room', [-1]),
                            ('objects', [True]), ('objects', [1.]), ('objects', [1, 99])]:
            with self.subTest(source=source, ids=ids), self.assertRaises(ValueError):
                self.composite.set_selection(source, ids)
        self.assertEqual(self.composite.selections['objects'], set())
        self.assertEqual(set(self.samples()), {('room', 0)})

    def test_registration_and_static_layout_validation(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.add_source('room', self.environment)
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            self.add_source('second', child('second'), is_environment=True)
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.add_source('wrong', self.object)
        for axis in ['B', 'F', 'N', 'V']:
            # B needs its own layout because child() already supplies B.
            cache = child('multi', count=2, axis=axis if axis != 'B' else 'N')
            if axis == 'B':
                cache.input_image = cache.input_image[:, :, :1].expand(2, 1, 1, 1, 3, 2, 3)
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, 'B=F=N=V=1'):
                self.add_source('multi', cache)
        with self.assertRaises(ValueError):
            self.add_source('unlabeled', child('unlabeled', labels=False))
        cache = child('bad')
        cache.input_image.fill_(2)
        with self.assertRaisesRegex(ValueError, 'RGB'):
            self.add_source('bad', cache)
        no_environment = Cache3D_Composite()
        no_environment.add_source('objects', self.object, source_w2c=torch.eye(4), source_intrinsics=torch.eye(3))
        with self.assertRaisesRegex(ValueError, 'environment'):
            list(no_environment._iter_render_inputs())
        self.assertNotIn('bad', self.composite.sources)

    def test_identity_validity_pixel_alignment_and_static_reuse(self):
        self.composite.set_selection('objects', [1, 2])
        self.object.input_mask[..., 0, 1] = False
        self.object.input_points[..., 1, 1, 0] = float('nan')
        samples = self.samples()
        self.assertNotIn(('objects', 1), samples)
        sample = samples['objects', 2]
        self.assertEqual(sample.pixel_indices.tolist(), [2])
        torch.testing.assert_close(sample.points, self.object.input_points.reshape(-1, 3)[[2]].float())
        expected_rgb = self.object.input_image[0, 0, 0, 0, :, 0, 2][None]
        torch.testing.assert_close(sample.rgb, expected_rgb)
        # Each target frame consumes the same static geometry, without a time mapping.
        torch.testing.assert_close(sample.points, self.samples()['objects', 2].points)
        sample.rgb.fill_(99)
        torch.testing.assert_close(expected_rgb, self.samples()['objects', 2].rgb)

    def test_scale_rotation_translation_order_and_pivot(self):
        self.composite.set_selection('objects', [1])
        original = self.samples()['objects', 1].points
        rotate_scale = torch.tensor([[0., -2, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], dtype=torch.float64)
        pivot = translation(1, 2, 3)
        object_transform = pivot @ rotate_scale @ torch.linalg.inv(pivot)
        source_transform = translation(10, -4, 2) @ rotate_scale
        self.composite.set_transform('objects', object_transform, instance_id=1)
        self.composite.set_transform('objects', source_transform)
        placed = self.samples()['objects', 1].points
        homogeneous = torch.cat([original, torch.ones(len(original), 1)], dim=1)
        expected = (source_transform @ object_transform @ homogeneous.double().T).T[:, :3].float()
        torch.testing.assert_close(placed, expected)
        torch.testing.assert_close(placed, torch.tensor([[6., 6, 0], [6, 2, 0]], dtype=torch.float32))
        # A point at the chosen source-space pivot remains there under object placement.
        torch.testing.assert_close(object_transform @ pivot[:, 3], pivot[:, 3])
        object_transform.fill_(0)  # The setter owns its matrix.
        torch.testing.assert_close(self.samples()['objects', 1].points, expected)
        self.composite.set_transform('room', translation(3), instance_id=1)
        self.composite.set_selection('room', [1])
        torch.testing.assert_close(self.samples()['room', 0].points,
                                   self.environment.input_points.reshape(-1, 3)[[0, 5]].float())

    def test_reject_invalid_transforms(self):
        invalid = [torch.eye(3), torch.full((4, 4), float('nan')), torch.full((4, 4), float('inf')),
                   torch.diag(torch.tensor([0., 0, 0, 1])), torch.diag(torch.tensor([-1., 1, 1, 1])),
                   torch.diag(torch.tensor([1., 2, 1, 1]))]
        shear = torch.eye(4)
        shear[0, 1] = .2
        perspective = torch.eye(4)
        perspective[3, 0] = .1
        invalid.extend([shear, perspective, torch.eye(4, dtype=torch.complex64)])
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                self.composite.set_transform('objects', matrix)
        for source, instance in [('room', None), ('room', 0), ('objects', 0), ('objects', -1), ('objects', 99)]:
            with self.subTest(source=source, instance=instance), self.assertRaises(ValueError):
                self.composite.set_transform(source, translation(1), instance_id=instance)
        self.composite.set_transform('room', torch.eye(4))
        self.composite.set_transform('objects', torch.diag(torch.tensor([1e-9, 1e-9, 1e-9, 1.])))

    def test_alias_corrections_conflicts_and_revalidation(self):
        self.composite.set_selection('objects', [1, 2])
        self.composite.set_transform('objects', translation(1), instance_id=1)
        self.composite.set_transform('objects', translation(2), instance_id=2)
        self.object.object_registry.merge_ids(1, 2)
        # A registry merge without rebuilding labels is invalid.
        with self.assertRaisesRegex(ValueError, 'Rebuild'):
            self.samples()
        self.object.input_labels[self.object.input_labels == 1] = 2
        with self.assertRaisesRegex(ValueError, 'conflicting transforms'):
            self.samples()
        self.composite.set_transform('objects', translation(3), instance_id=1)
        self.composite.set_selection('objects', [1])
        self.assertEqual(set(self.samples()), {('room', 0), ('objects', 2)})
        # Corrected labels can remove every visible sample of a selected object.
        self.object.input_labels[self.object.input_labels == 2] = -1
        self.assertEqual(set(self.samples()), {('room', 0)})
        del self.object.object_registry.names[2]
        with self.assertRaisesRegex(ValueError, 'Unknown canonical'):
            self.samples()

    def test_child_state_unchanged_and_attachment_invalidated(self):
        snapshots = {name: getattr(self.object, name).clone() for name in
                     ['input_points', 'input_image', 'input_labels', 'input_mask']}
        self.composite.set_selection('objects', [1])
        self.composite.set_transform('objects', translation(5))
        self.composite.set_transform('objects', translation(3), instance_id=1)
        self.samples()
        for name, before in snapshots.items():
            torch.testing.assert_close(getattr(self.object, name), before)
        # Buffer insertion clears original source attachment. Composite must fail.
        buffer = child('buffer', cache_type=Cache3D_Buffer)
        buffer.source_fingerprint = 'original'
        buffer._label_binding = ('buffer', 'original', None)
        self.add_source('buffer', buffer)
        buffer.update_cache(torch.ones(1, 3, 2, 3), torch.ones(1, 1, 2, 3),
                            torch.eye(4)[None], new_intrinsics=torch.eye(3)[None], depth_alignment=False,
                            new_labels=torch.ones(1, 1, 2, 3, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, 'attachment'):
            self.samples()

    def test_replace_labels_preserves_selection_and_placement_through_merge(self):
        prepared = prepare_source({(7, 9): SourceObservation(
            torch.zeros(3, 2, 3), torch.ones(2, 3), torch.eye(3), torch.eye(4),
        )})

        def labeled(mask, registry=None):
            proposal = InstanceProposal(mask, None, 1, 'Cup', track_id=('corrected', 9, 'session', 1))
            observation = _segmented_observation(
                (7, 9), prepared.observations[7, 9], [proposal], torch.ones(2, 3, dtype=torch.bool),
            )
            return label_source(SegmentedSource(prepared, {(7, 9): observation}),
                                source_id='corrected', registry=registry)

        initial = labeled(torch.tensor([[True, False, False], [False, False, False]]))
        cache = Cache3D_Base(**initial.cache_kwargs(), device='cpu')
        self.add_source('corrected', cache)
        self.composite.set_selection('corrected', [1])
        self.composite.set_transform('corrected', translation(4), instance_id=1)
        updated = labeled(torch.tensor([[False, False, False], [False, False, True]]), initial.registry)
        survivor = updated.registry.allocate('Merged')
        updated.registry.merge_ids(1, survivor)
        updated = label_source(updated.segmented_source, source_id='corrected', registry=updated.registry)
        cache.replace_labels(updated)
        sample = self.samples()['corrected', survivor]
        self.assertEqual(sample.pixel_indices.tolist(), [5])
        torch.testing.assert_close(sample.points, torch.tensor([[6., 1, 1]], dtype=torch.float32))

    def test_unfingerprinted_buffer_update_invalidates_attachment(self):
        buffer = child('buffer', cache_type=Cache3D_Buffer)
        self.assertIsNone(buffer.source_fingerprint)
        self.assertIsNone(buffer.observation_keys)
        self.add_source('buffer', buffer)
        buffer.update_cache(torch.ones(1, 3, 2, 3), torch.ones(1, 1, 2, 3),
                            torch.eye(4)[None], new_intrinsics=torch.eye(3)[None], depth_alignment=False,
                            new_labels=torch.ones(1, 1, 2, 3, dtype=torch.int64))
        for action in (self.samples, lambda: self.composite.set_selection('buffer', [1]),
                       lambda: self.composite.set_transform('buffer', translation(1)),
                       lambda: self.composite.refresh_source('buffer')):
            with self.assertRaisesRegex(ValueError, 'attachment'):
                action()

    def test_placement_and_render_skip_full_validation_until_correction(self):
        with patch.object(self.object, 'check_label_consistency',
                          wraps=self.object.check_label_consistency) as validate:
            for offset in range(3):
                self.composite.set_selection('objects', [1])
                self.composite.set_transform('objects', translation(offset), instance_id=1)
                self.samples()
            validate.assert_not_called()
            # A replacement label snapshot triggers one full validation/refresh.
            self.object.input_labels = self.object.input_labels.clone()
            self.samples()
            self.samples()
            self.assertEqual(validate.call_count, 1)
            # Direct in-place edits require explicit refresh, which validates RGB too.
            self.object.input_image.fill_(2)
            with self.assertRaisesRegex(ValueError, 'RGB'):
                self.composite.refresh_source('objects')

    def test_unselected_merge_conflict_does_not_block_other_objects(self):
        self.composite.set_transform('objects', translation(1), instance_id=1)
        self.composite.set_transform('objects', translation(2), instance_id=2)
        self.object.object_registry.merge_ids(1, 2)
        self.object.input_labels[self.object.input_labels == 1] = 2
        self.assertEqual(set(self.samples()), {('room', 0)})
        self.composite.set_selection('objects', [1])
        with self.assertRaisesRegex(ValueError, 'conflicting transforms'):
            self.samples()
        self.composite.refresh_source('objects')  # Conflict survives additional refreshes.
        with self.assertRaisesRegex(ValueError, 'conflicting transforms'):
            self.samples()
        self.composite.set_transform('objects', translation(3), instance_id=1)
        self.assertIn(('objects', 2), self.samples())

    def test_render_samples_use_float32_without_changing_child_precision(self):
        self.object.input_points = self.object.input_points.double()
        self.composite.set_selection('objects', [1])
        sample = self.samples()['objects', 1]
        self.assertEqual(sample.points.dtype, torch.float32)
        self.assertEqual(self.object.input_points.dtype, torch.float64)
        self.assertEqual(self.composite.source_transforms['objects'].dtype, torch.float64)


if __name__ == '__main__':
    unittest.main()
