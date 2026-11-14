"""Adaptive source-cell coverage, visibility, and shared conditioning contracts."""

import unittest
from unittest.mock import patch

import torch

from cosmos_predict1.diffusion.inference.cache3d_composite import Cache3D_Composite, combine_visible_buffers
from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    adaptive_surface_winners, nearest_surface_winners, pixel_cell_winner_depth,
)
from tests.test_composite_rendering import source


MODULE = 'cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch'
CACHE_MODULE = 'cosmos_predict1.diffusion.inference.cache3d_composite'


def grid_points(depth, intrinsic):
    height, width = depth.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    rays = torch.stack((x, y, torch.ones_like(x)), -1).float() @ torch.linalg.inv(intrinsic).T
    return rays * depth[..., None]


def scene_with_patch(labels=None, mask=None, depth=None, **options):
    target_k = torch.tensor([[4., 0, 11.5], [0, 4, 11.5], [0, 0, 1]])
    source_k = torch.tensor([[4., 0, 1.5], [0, 4, 1.5], [0, 0, 1]])
    depth = torch.full((4, 4), 2.) if depth is None else depth
    labels = torch.ones(4, 4, dtype=torch.int64) if labels is None else labels
    scene = Cache3D_Composite(image_size=(24, 24), **options)
    room = source('room', grid_points(torch.full((24, 24), 8.), target_k), size=(24, 24),
                  colors=[[-1., -1, 1]] * 576)
    prop = source('prop', grid_points(depth, source_k), ids=labels.reshape(-1).tolist(), size=(4, 4),
                  mask=None if mask is None else mask.reshape(-1).tolist())
    scene.add_source('room', room, is_environment=True, source_w2c=torch.eye(4), source_intrinsics=target_k)
    scene.add_source('prop', prop, source_w2c=torch.eye(4), source_intrinsics=source_k)
    scene.set_selection('prop', [1])
    return scene, target_k


class AdaptiveSplattingTests(unittest.TestCase):
    def test_approach_has_continuous_foreground_and_conditioning_matches_preview(self):
        scene, intrinsic = scene_with_patch(point_chunk_size=101)
        poses = torch.eye(4).repeat(1, 3, 1, 1)
        poses[0, :, 2, 3] = torch.tensor([0., -1., -1.5])
        k = intrinsic.repeat(1, 3, 1, 1)
        details = scene.render_diagnostics(poses, k, include_rgb=True)
        for frame, expected in enumerate((16, 64, 256)):
            object_pixels = details['winners'][0, frame, ..., 1] == 1
            self.assertEqual(int(object_pixels.sum()), expected)
            side = int(expected ** .5)
            start = (24 - side) // 2
            self.assertTrue(object_pixels[start:start + side, start:start + side].all())
            torch.testing.assert_close(details['depth'][0, frame, 0][object_pixels],
                                       torch.full((expected,), (2., 1., .5)[frame]))
        buffers = scene.render_conditioning(poses, k, include_depth=True)
        rgb, coverage = combine_visible_buffers(buffers['rgb'], buffers['coverage'])
        torch.testing.assert_close(rgb, details['rgb'], rtol=0, atol=0)
        torch.testing.assert_close(coverage, details['coverage'], rtol=0, atol=0)
        torch.testing.assert_close(buffers['depth'].sum(2, keepdim=True), details['depth'], rtol=0, atol=0)
        self.assertTrue((buffers['coverage'].sum(2) <= 1).all())

    def test_large_offscreen_center_splat_is_streamed_without_size_fallback(self):
        # Center projects outside the frame; its valid cell covers the whole image.
        points = torch.tensor([[-10., 0, 2.]])
        tangents = torch.tensor([[[128., 0, 0], [0, 128., 0]]])
        pose, k = torch.eye(4)[None], torch.eye(3)[None]
        from cosmos_predict1.diffusion.inference import forward_warp_utils_pytorch as warp
        original = warp._pixel_cell_rays
        for budget in (7, 53, 1000):
            sizes = []
            def rays(xy, intrinsics):
                sizes.append(len(xy))
                return original(xy, intrinsics)
            with patch(MODULE + '._pixel_cell_rays', side_effect=rays):
                winners, depths = adaptive_surface_winners(points, tangents, pose, k, (16, 16),
                                                           candidate_batch_size=budget)
            self.assertTrue((winners == 0).all())
            self.assertTrue((depths == 2).all())
            self.assertLessEqual(max(sizes), budget)

    def test_plane_depth_controls_visibility_and_tolerance_winner_depth(self):
        points = torch.tensor([[0., 0, 2.], [0, 0, 2.1]])
        axes = torch.tensor([[[2., 0, 1], [0, 2, 0]], [[8., 0, 0], [0, 8, 0]]])
        pose = torch.eye(4)[None]
        k = torch.tensor([[[4., 0, 4], [0, 4, 4], [0, 0, 1]]])
        winners, nearest = adaptive_surface_winners(points, axes, pose, k, (9, 9), candidate_batch_size=13)
        self.assertEqual(int(winners[0, 4, 3]), 0)
        self.assertEqual(int(winners[0, 4, 5]), 1)  # Center-depth testing would choose point 0.
        self.assertAlmostEqual(float(nearest[0, 4, 3]), 2 / 1.125, places=5)
        # Priorities choose index 1 within tolerance of the true minimum, not index 0.
        points = torch.tensor([[0., 0, 1.18], [0, 0, 1.09], [0, 0, 1.]])
        axes = torch.tensor([[[4., 0, .1], [0, 4, 0]]] * 3)
        for point_batch, budget in ((1, 1), (2, 7), (99, 1000)):
            winners, nearest = adaptive_surface_winners(points, axes, pose, k, (3, 3), depth_tolerance=.1,
                                                        point_chunk_size=point_batch, candidate_batch_size=budget)
            # This camera's principal pixel lies outside (3,3), so use its interior ray analytically.
            xy = torch.tensor([[2., 2.]])
            rank = int(winners[0, 2, 2])
            self.assertEqual(rank, 1)
            chosen = pixel_cell_winner_depth(points[rank:rank + 1], axes[rank:rank + 1], pose, k, xy)
            self.assertAlmostEqual(float(chosen[0]), 1.09 / 1.0125, places=5)
            self.assertGreater(float(chosen[0]), float(nearest[0, 2, 2]))

    def test_degenerate_and_near_plane_cells_fall_back_to_points(self):
        points = torch.tensor([[0., 0, 2], [2, 0, 2], [4, 0, 2], [0, 0, -1], [float('nan'), 0, 2]])
        axes = torch.tensor([[[0., 0, 0], [0, 0, 0]], [[1, 0, 0], [2, 0, 0]],
                             [[1, 0, 6], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]],
                             [[1, 0, 0], [0, 1, 0]]])
        pose, k = torch.eye(4)[None], torch.eye(3)[None]
        expected = nearest_surface_winners(points, pose, k, (3, 4))
        actual = adaptive_surface_winners(points, axes, pose, k, (3, 4), candidate_batch_size=2)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        empty = adaptive_surface_winners(torch.empty(0, 3), torch.empty(0, 2, 3), pose, k, (3, 4))
        self.assertTrue((empty[0] == -1).all())
        self.assertFalse(empty[1].any())

    def test_cutouts_thin_strips_invalid_depth_and_label_edges(self):
        for mode in ('cutout', 'strip', 'invalid', 'other_label'):
            labels = torch.ones(4, 4, dtype=torch.int64)
            mask = torch.ones(4, 4, dtype=torch.bool)
            depth = torch.full((4, 4), 2.)
            if mode == 'cutout':
                labels[1:3, 1:3] = -1
            elif mode == 'strip':
                labels[:, 1:] = -1
            elif mode == 'invalid':
                depth[1:3, 1:3] = float('nan')
                mask[1:3, 1:3] = False
            else:
                labels[:, 2:] = 2
            scene, intrinsic = scene_with_patch(labels, mask, depth)
            pose = torch.eye(4).reshape(1, 1, 4, 4)
            pose[0, 0, 2, 3] = -1.5
            details = scene.render_diagnostics(pose, intrinsic[None, None])
            pixels = details['winners'][0, 0, ..., 1] == 1
            expected = int(((labels == 1) & mask).sum()) * 16
            self.assertEqual(int(pixels.sum()), expected, mode)
            # Excluded/invalid source cells remain environment, not filled foreground.
            expected_mask = ((labels == 1) & mask).repeat_interleave(4, 0).repeat_interleave(4, 1)
            torch.testing.assert_close(pixels[4:20, 4:20], expected_mask, rtol=0, atol=0)

    def test_source_depth_edges_do_not_produce_long_tangents(self):
        depth = torch.full((4, 4), 2.)
        depth[:, 2:] = 20.
        scene, _ = scene_with_patch(depth=depth)
        sample = next(s for s in scene._iter_render_inputs() if s.source_id == 'prop')
        # The discontinuity is excluded; steps come from neighbors on each flat side.
        torch.testing.assert_close(sample.tangents[:, 0, 2], torch.zeros(16))
        torch.testing.assert_close(sample.tangents[:, 0, 0].reshape(4, 4), depth / 4)

    def test_support_invalidation_and_linear_placement(self):
        scene, intrinsic = scene_with_patch()
        target = (torch.eye(4).reshape(1, 1, 4, 4), intrinsic[None, None])
        scene.render_cache(*target)
        table = scene._sample_tables['prop']
        scene.render_conditioning(*target)
        torch.testing.assert_close(scene._sample_tables['prop'].tangents, table.tangents)
        transform = torch.eye(4)
        transform[:3, :3] *= 2
        transform[:3, 3] = torch.tensor([1., 2., -2.])
        scene.set_transform('prop', transform)
        scene.render_cache(*target)
        torch.testing.assert_close(scene._sample_tables['prop'].tangents, table.tangents * 2)
        # Replacing labels invalidates cached support; no cross-label differences survive.
        scene.sources['prop'].input_labels = scene.sources['prop'].input_labels.clone()
        scene.sources['prop'].input_labels[..., 1:3, 1:3] = -1
        scene.render_cache(*target)
        self.assertEqual(len(scene._sample_tables['prop'].tangents), 12)
        scene.sources['prop'].input_points[..., 0] *= 1.1
        scene.refresh_source('prop')
        self.assertNotIn('prop', scene._sample_tables)

    def test_fractional_motion_oblique_and_noisy_depth_batch_invariance(self):
        y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        depth = 2 + x * .02 + y * .01 + torch.sin(x.float() + y) * .005
        scene, intrinsic = scene_with_patch(depth=depth)
        poses = torch.eye(4).repeat(1, 3, 1, 1)
        poses[0, :, 0, 3] = torch.tensor([.03, -.08, .12])
        poses[0, :, 2, 3] = -1
        k = intrinsic.repeat(1, 3, 1, 1)
        expected = scene.render_diagnostics(poses, k, include_rgb=True)
        for frame_batch, point_batch, candidates in ((1, 1, 7), (3, 57, 211)):
            scene.frame_chunk_size, scene.point_chunk_size = frame_batch, point_batch
            with patch(CACHE_MODULE + '._CANDIDATE_BATCH_SIZE', candidates):
                actual = scene.render_diagnostics(poses, k, include_rgb=True)
            for name in ('rgb', 'depth', 'coverage', 'winners', 'nearest_depth'):
                torch.testing.assert_close(actual[name], expected[name])
        visible = expected['coverage'].bool().squeeze(3)
        self.assertTrue(torch.isfinite(expected['depth']).all())
        self.assertTrue((expected['depth'][visible] > 0).all())

    def test_gathered_tie_depth_is_winning_plane_depth_not_center_or_minimum(self):
        # Two selected sources observe the same gently slanted plane. Lexical
        # priority prefers the slightly farther surface within depth tolerance.
        y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        depth = 2 / (1 - .05 * ((x - 1.5) / 4))
        scene, intrinsic = scene_with_patch(depth=depth, depth_tolerance=.1)
        original = scene.sources['prop']
        other = source('a-priority', original.input_points.reshape(-1, 3) * 1.02,
                       size=(4, 4), ids=[1] * 16)
        source_k = torch.tensor([[4., 0, 1.5], [0, 4, 1.5], [0, 0, 1]])
        scene.add_source('a-priority', other, source_w2c=torch.eye(4), source_intrinsics=source_k)
        scene.set_selection('a-priority', [1])
        pose = torch.eye(4).reshape(1, 1, 4, 4)
        pose[0, 0, 0, 3] = .07
        pose[0, 0, 2, 3] = -1
        result = scene.render_diagnostics(pose, intrinsic[None, None])
        owned = result['winners'][0, 0, ..., 0] == result['source_ids'].index('a-priority')
        self.assertTrue(owned.any())
        table = scene._joint_table
        metadata = result['winners'][0, 0][owned]
        source_points = other.input_points.reshape(-1, 3)[metadata[:, 2]]
        center_depth = source_points[:, 2] - 1
        selected_depth = result['depth'][0, 0, 0][owned]
        self.assertGreater(float((selected_depth - center_depth).abs().max()), 1e-4)
        self.assertTrue((selected_depth > result['nearest_depth'][0, 0, 0][owned]).all())
        xy_y, xy_x = torch.where(owned)
        # Analytic tilted plane after translating the camera.
        ray_x = (xy_x - 11.5) / 4
        expected = (2 * 1.02 - 1 - .05 * .07) / (1 - .05 * ray_x)
        torch.testing.assert_close(selected_depth, expected, atol=3e-4, rtol=0)

    def test_runtime_camera_validation(self):
        scene = Cache3D_Composite(image_size=(2, 2))
        scene.add_source('room', source('room', [[0, 0, 1]]), is_environment=True)
        with self.assertRaisesRegex(ValueError, 'requires source_w2c'):
            scene.render_cache(torch.eye(4).reshape(1, 1, 4, 4), torch.eye(3).reshape(1, 1, 3, 3))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_adaptive_matches_cpu(self):
        scene, intrinsic = scene_with_patch()
        poses = torch.eye(4).repeat(1, 2, 1, 1)
        poses[0, 1, 2, 3] = -1.5
        k = intrinsic.repeat(1, 2, 1, 1)
        expected = scene.render_diagnostics(poses, k, include_rgb=True)
        actual = scene.render_diagnostics(poses.cuda(), k.cuda(), include_rgb=True)
        for name in ('rgb', 'depth', 'coverage', 'winners', 'nearest_depth'):
            torch.testing.assert_close(actual[name].cpu(), expected[name])


if __name__ == '__main__':
    unittest.main()
