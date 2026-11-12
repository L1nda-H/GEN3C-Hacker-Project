"""Joint visibility contracts, using explicit synthetic world geometry on CPU."""

import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base
from cosmos_predict1.diffusion.inference.cache3d_composite import Cache3D_Composite
from cosmos_predict1.diffusion.inference.instance_labeling import ObjectRegistry


def source(source_id, points, *, ids=None, colors=None, size=None, mask=None):
    points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)
    size = size or (1, len(points))
    ids = torch.tensor(ids if ids is not None else [0] * len(points), dtype=torch.int64)
    registry = ObjectRegistry(source_id)
    registry.names = {i: f'Object {i}' for i in ids.tolist() if i > 0}
    registry.next_id = max([0, *registry.names]) + 1
    colors = torch.as_tensor(colors if colors is not None else [[1., -1, -1]] * len(points), dtype=torch.float32)
    return Cache3D_Base(
        input_image=colors.reshape(*size, 3).permute(2, 0, 1)[None],
        input_depth=None, input_points=points.reshape(1, *size, 3),
        input_mask=None if mask is None else torch.tensor(mask).reshape(1, 1, *size),
        input_w2c=torch.eye(4)[None], input_intrinsics=torch.eye(3)[None],
        input_labels=ids.reshape(1, 1, *size), object_registry=registry, device='cpu',
    )


def cameras(frames=1):
    return torch.eye(4).repeat(1, frames, 1, 1), torch.eye(3).repeat(1, frames, 1, 1)


def composite(environment, *objects, **options):
    result = Cache3D_Composite(image_size=options.pop('image_size', (3, 4)), **options)
    result.add_source(environment.object_registry.source_id, environment, is_environment=True)
    for cache in objects:
        name = cache.object_registry.source_id
        result.add_source(name, cache)
        result.set_selection(name, list(cache.object_registry.names))
    return result


class RenderingTests(unittest.TestCase):
    def render(self, cache, target=None):
        target = cameras() if target is None else target
        rgb, coverage = cache.render_cache(*target)
        depth, depth_coverage = cache.render_cache(*target, render_depth=True)
        torch.testing.assert_close(coverage, depth_coverage, rtol=0, atol=0)
        self.assertEqual(rgb.dtype, torch.float32)
        self.assertEqual(depth.dtype, torch.float32)
        self.assertEqual(coverage.dtype, torch.float32)
        self.assertTrue(torch.all((coverage == 0) | (coverage == 1)))
        holes = ~coverage.bool()
        self.assertTrue((rgb[holes.expand_as(rgb)] == -1).all())
        self.assertTrue((depth[holes.squeeze(3)] == 0).all())
        return rgb, coverage, depth

    def test_identity_projection_layout_and_legacy_parity(self):
        y, x = torch.meshgrid(torch.arange(3), torch.arange(4), indexing='ij')
        points = torch.stack([x * 2, y * 2, torch.full_like(x, 2)], -1)
        colors = torch.linspace(-1, 1, 36).reshape(-1, 3)
        cache = source('room', points, colors=colors, size=(3, 4))
        rgb, coverage, depth = self.render(composite(cache))
        self.assertEqual(rgb.shape, (1, 1, 1, 3, 3, 4))
        self.assertEqual(coverage.shape, (1, 1, 1, 1, 3, 4))
        self.assertEqual(depth.shape, (1, 1, 1, 3, 4))
        torch.testing.assert_close(rgb[0, 0, 0], cache.input_image[0, 0, 0, 0])
        self.assertTrue((coverage == 1).all())
        self.assertTrue((depth == 2).all())
        legacy_rgb, legacy_coverage = cache.render_cache(*cameras(), selected_ids=[0])
        torch.testing.assert_close(rgb, legacy_rgb)
        torch.testing.assert_close(coverage, legacy_coverage)

    def test_joint_occlusion_within_and_across_sources(self):
        env = source('room', [[0, 0, 4], [2, 0, 2]], colors=[[-1, -1, 1]] * 2)
        obj = source('props', [[0, 0, 5], [0, 0, 2], [3, 0, 3]], ids=[1, 1, 1],
                     colors=[[1, -1, -1], [-1, 1, -1], [1, 1, -1]])
        scene = composite(env, obj)
        rgb, coverage, depth = self.render(scene)
        torch.testing.assert_close(rgb[0, 0, 0, :, 0, :2], torch.tensor([[-1., -1], [1, -1], [-1, 1]]))
        torch.testing.assert_close(depth[0, 0, 0, 0], torch.tensor([2., 2, 0, 0]))
        self.assertEqual(coverage.sum(), 2)
        diagnostics = scene.render_diagnostics(*cameras())
        self.assertEqual(diagnostics['source_ids'], ('props', 'room'))
        self.assertEqual(diagnostics['winners'][0, 0, 0, 0].tolist(), [0, 1, 1])
        self.assertEqual(diagnostics['winners'][0, 0, 0, 1].tolist(), [1, 0, 1])
        self.assertEqual(diagnostics['winners'][0, 0, 1, 0].tolist(), [-1, -1, -1])
        torch.testing.assert_close(diagnostics['depth'], depth)
        torch.testing.assert_close(diagnostics['coverage'], coverage)

    def test_tolerance_compares_to_global_minimum_without_chains(self):
        env = source('z-room', [[0, 0, 1]], ids=[-1])
        # Lexical priority prefers a, but it is outside tolerance of c. b wins.
        objects = [source(name, [[0, 0, z]], ids=[1], colors=[color]) for name, z, color in [
            ('a', 1.18, [1, -1, -1]), ('b', 1.09, [-1, 1, -1]), ('c', 1., [-1, -1, 1])]]
        for chunk in [1, 2, 100]:
            for order in [objects, objects[::-1]]:
                scene = composite(env, *order, depth_tolerance=.1, point_chunk_size=chunk)
                rgb, _, depth = self.render(scene)
                torch.testing.assert_close(rgb[0, 0, 0, :, 0, 0], torch.tensor([-1., 1, -1]))
                self.assertAlmostEqual(depth[0, 0, 0, 0, 0].item(), 1.09, places=6)
                details = scene.render_diagnostics(*cameras())
                self.assertEqual(details['nearest_depth'][0, 0, 0, 0, 0], 1)

    def test_exact_ties_use_source_then_canonical_id_then_pixel(self):
        big = 2**60 + 7
        env = source('z-room', [[0, 0, 2]])
        obj = source('a', [[0, 0, 2]] * 3, ids=[big, 1, 1],
                     colors=[[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        scene = composite(env, obj, depth_tolerance=0, point_chunk_size=1)
        rgb, _, _ = self.render(scene)
        torch.testing.assert_close(rgb[0, 0, 0, :, 0, 0], torch.tensor([-1., 1, -1]))
        self.assertEqual(scene.render_diagnostics(*cameras())['winners'][0, 0, 0, 0].tolist(), [0, 1, 1])
        scene.set_selection('a', [big])
        self.assertEqual(scene.render_diagnostics(*cameras())['winners'][0, 0, 0, 0, 1].item(), big)

    def test_nearest_pixel_rounding_bounds_and_invalid_geometry(self):
        points = [[-.5, 0, 1], [.5, 0, 1], [3.499, 0, 1], [3.5, 0, 1], [-.501, 1, 1],
                  [0, 0, 0], [0, 0, -1], [float('nan'), 0, 1], [float('inf'), 0, 1],
                  [1e30, 0, 1e-30], [0, 1, 1], [0, 2, 1], [0, 2.5, 1]]
        ids = [0] * len(points)
        ids[11] = -1
        mask = [True] * len(points)
        mask[10] = False
        scene = composite(source('room', points, ids=ids, mask=mask))
        _, coverage, depth = self.render(scene)
        torch.testing.assert_close(coverage[0, 0, 0, 0], torch.tensor([[1., 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]))
        self.assertEqual(depth.sum(), 3)

    def test_empty_selection_no_environment_pixels_and_no_environment_source(self):
        scene = composite(source('room', [[0, 0, 1]], ids=[-1]), source('prop', [[0, 0, 1]], ids=[1]))
        scene.set_selection('prop', [])
        _, coverage, _ = self.render(scene)
        self.assertFalse(coverage.any())
        self.assertTrue((scene.render_diagnostics(*cameras())['winners'] == -1).all())
        with self.assertRaisesRegex(ValueError, 'environment'):
            Cache3D_Composite(image_size=(3, 4)).render_cache(*cameras())

    def test_placement_unequal_source_sizes_and_excluded_objects(self):
        env = source('room', [[0, 0, 4], [4, 0, 4], [8, 0, 4]], size=(1, 3))
        obj = source('prop', [[0, 0, 2], [0, 0, 1]], size=(2, 1), ids=[1, 2], colors=[[-1, 1, -1]] * 2)
        scene = composite(env, obj)
        scene.set_selection('prop', [1])
        transform = torch.eye(4)
        transform[0, 3] = 2
        scene.set_transform('prop', transform, instance_id=1)
        rgb, _, depth = self.render(scene)
        self.assertEqual(depth[0, 0, 0, 0, 0], 4)  # excluded ID 2 cannot occlude
        self.assertEqual(depth[0, 0, 0, 0, 1], 2)
        torch.testing.assert_close(rgb[0, 0, 0, :, 0, 1], torch.tensor([-1., 1, -1]))

    def test_moving_camera_static_repeat_and_chunk_equivalence(self):
        env = source('room', [[0, 0, 2], [2, 0, 2], [4, 0, 2]])
        scene = composite(env)
        target = cameras(5)
        target[0][0, 1, 0, 3] = 2  # OpenCV w2c moves projections right.
        target[0][0, 2, 2, 3] = 2  # camera-Z changes from 2 to 4.
        baseline = self.render(scene, target)
        self.assertEqual(baseline[1][0, 1, 0, 0, 0, 0], 0)
        self.assertEqual(baseline[2][0, 2, 0, 0, 0], 4)
        for frame_chunk, point_chunk in [(1, 1), (3, 2), (20, 100)]:
            scene.frame_chunk_size, scene.point_chunk_size = frame_chunk, point_chunk
            rendered = self.render(scene, target)
            for expected, actual in zip(baseline, rendered):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            rgb, mask = scene.render_cache(*target, start_frame_idx=123)
            torch.testing.assert_close(rgb, baseline[0], rtol=0, atol=0)
            torch.testing.assert_close(mask, baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(baseline[0][:, 0], baseline[0][:, 4])

    def test_rotated_world_to_camera_and_nontrivial_intrinsics(self):
        scene = composite(source('room', [[2, 0, 0], [2, 0, -1], [2, 2, 0]]))
        pose, k = cameras()
        pose[0, 0, :3, :3] = torch.tensor([[0., 0, -1], [0, 1, 0], [1, 0, 0]])
        k[0, 0] = torch.tensor([[2., 0, 1], [0, 3, 1], [0, 0, 1]])
        _, coverage, depth = self.render(scene, (pose, k))
        torch.testing.assert_close(coverage[0, 0, 0, 0], torch.tensor([[0., 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]))
        torch.testing.assert_close(depth[0, 0, 0, 1, 1:3], torch.tensor([2., 2]))

    def test_equal_depth_ties_independent_of_environment_registration_order(self):
        env = source('a', [[0, 0, 2]], colors=[[-1, -1, 1]])
        obj = source('z', [[0, 0, 2]], ids=[1])
        baseline = self.render(composite(env, obj, depth_tolerance=0))
        scene = Cache3D_Composite(image_size=(3, 4), depth_tolerance=0)
        scene.add_source('z', obj)
        scene.set_selection('z', [1])
        scene.add_source('a', env, is_environment=True)
        for expected, actual in zip(baseline, self.render(scene)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(baseline[0][0, 0, 0, :, 0, 0], torch.tensor([-1., -1, 1]))

    def test_float32_contract_under_autocast_and_default_dtype_changes(self):
        scene = composite(source('room', [[0, 0, 1], [0, 0, 1.001]]), depth_tolerance=0)
        target = cameras()
        baseline = self.render(scene, target)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = self.render(scene, target)
        for expected, rendered in zip(baseline, actual):
            torch.testing.assert_close(rendered, expected, rtol=0, atol=0)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            actual = self.render(scene, target)
        finally:
            torch.set_default_dtype(previous)
        for expected, rendered in zip(baseline, actual):
            torch.testing.assert_close(rendered, expected, rtol=0, atol=0)

    def test_configuration_and_camera_validation(self):
        for options in [dict(image_size=(0, 4)), dict(image_size=(True, 4)), dict(image_size=(3,)),
                        dict(depth_tolerance=-1), dict(depth_tolerance=float('nan')),
                        dict(frame_chunk_size=0), dict(point_chunk_size=1.5)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                Cache3D_Composite(**options)
        with self.assertRaisesRegex(ValueError, 'image_size'):
            Cache3D_Composite().render_cache(*cameras())
        scene = composite(source('room', [[0, 0, 1]]))
        for offset in [-1, True, .5]:
            with self.assertRaises(ValueError):
                scene.render_cache(*cameras(), start_frame_idx=offset)
        for target in [cameras(0), (torch.eye(4).repeat(2, 1, 1, 1), torch.eye(3).repeat(2, 1, 1, 1)),
                       (torch.eye(4), torch.eye(3))]:
            with self.assertRaises(ValueError):
                scene.render_cache(*target)
        for field, row, col, value in [(0, 3, 0, 1), (0, 0, 0, -1), (0, 0, 0, 2),
                                       (1, 0, 0, 0), (1, 2, 0, 1), (1, 0, 0, float('nan'))]:
            target = cameras()
            target[field][0, 0, row, col] = value
            with self.subTest(field=field, row=row, value=value), self.assertRaises(ValueError):
                scene.render_cache(*target)

    def test_rgb_range_conversion_and_children_unchanged(self):
        env = source('room', [[0, 0, 1]], colors=[[0, .5, 1]])
        snapshots = {name: getattr(env, name).clone() for name in ['input_points', 'input_image', 'input_labels', 'input_mask']}
        scene = composite(env, rgb_range=(0, 1))
        rgb, _, _ = self.render(scene)
        torch.testing.assert_close(rgb[0, 0, 0, :, 0, 0], torch.tensor([-1., 0, 1]))
        for name, snapshot in snapshots.items():
            torch.testing.assert_close(snapshot, getattr(env, name))

    def test_combined_diagnostics_matches_separate_outputs(self):
        scene = composite(source('room', [[0, 0, 2], [2, 0, 2], [4, 0, 2]]),
                          frame_chunk_size=3, point_chunk_size=4)
        target = cameras(5)
        target[0][0, 2, 0, 3] = 2
        rgb, coverage, depth = self.render(scene, target)
        separate = scene.render_diagnostics(*target)
        self.assertNotIn('rgb', separate)
        combined = scene.render_diagnostics(*target, include_rgb=True)
        for name, expected in [('rgb', rgb), ('depth', depth), ('coverage', coverage)]:
            torch.testing.assert_close(combined[name], expected, rtol=0, atol=0)
        for name in ('winners', 'nearest_depth'):
            torch.testing.assert_close(combined[name], separate[name], rtol=0, atol=0)
        self.assertEqual(combined['source_ids'], separate['source_ids'])
        # Also exercise the optional buffers for a scene without valid samples.
        scene.set_selection('room', [])  # environment stays selected
        scene.sources['room'].input_mask.zero_()
        empty = scene.render_diagnostics(*target, include_rgb=True)
        self.assertTrue((empty['rgb'] == -1).all())
        self.assertFalse(empty['coverage'].any())
        self.assertTrue((empty['winners'] == -1).all())

    def test_output_allocations_and_winner_gathers_are_bounded(self):
        # Three points cover nine pixels across frames. Four-sample batches
        # therefore exercise both a frame boundary and a partial final batch.
        scene = composite(source('room', [[0, 0, 1], [1, 0, 1], [2, 0, 1]]),
                          image_size=(2, 3), frame_chunk_size=3, point_chunk_size=4)
        target = cameras(3)
        allocations, gathers = [], []

        class MemoryOperations(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func in (torch.ops.aten.full.default, torch.ops.aten.zeros_like.default):
                    allocations.append((func, tuple(result.shape), result.dtype))
                if (func == torch.ops.aten.index.Tensor and args[0].ndim == 2
                        and args[0].shape[-1] == 3 and len(args[1]) == 1):
                    gathers.append(len(result))
                return result

        with MemoryOperations():
            _, coverage = scene.render_cache(*target)
        self.assertEqual(coverage.sum(), 9)
        self.assertNotIn((torch.ops.aten.zeros_like.default, (18,), torch.float32), allocations)
        allocations.clear()
        with MemoryOperations():
            depth, depth_coverage = scene.render_cache(*target, render_depth=True)
        self.assertNotIn((torch.ops.aten.full.default, (18, 3), torch.float32), allocations)
        self.assertEqual(depth.sum(), 9)
        torch.testing.assert_close(coverage, depth_coverage, rtol=0, atol=0)
        allocations.clear()
        with MemoryOperations():
            details = scene.render_diagnostics(*target)
        self.assertNotIn((torch.ops.aten.full.default, (18, 3), torch.float32), allocations)
        torch.testing.assert_close(details['depth'], depth, rtol=0, atol=0)
        self.assertTrue(gathers)
        self.assertLessEqual(max(gathers), scene.point_chunk_size)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_device_and_cpu_parity(self):
        scene = composite(source('room', [[0, 0, 1], [0, 0, 2]]), point_chunk_size=1)
        expected = self.render(scene)
        actual = self.render(scene, tuple(t.cuda() for t in cameras()))
        for left, right in zip(expected, actual):
            self.assertEqual(right.device.type, 'cuda')
            torch.testing.assert_close(left, right.cpu())


if __name__ == '__main__':
    unittest.main()
