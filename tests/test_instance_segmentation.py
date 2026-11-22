"""CPU checks for source alignment and the existing cache geometry contract."""

import unittest
from dataclasses import replace

import torch

from cosmos_predict1.diffusion.inference.instance_segmentation import SourceObservation, prepare_source


def observation(h=3, w=4):
    return SourceObservation(
        rgb=torch.arange(3 * h * w, dtype=torch.float32).reshape(3, h, w),
        depth=torch.full((h, w), 2.0),
        intrinsics=torch.tensor([[2.0, 0, 1], [0, 2, 1], [0, 0, 1]]),
        w2c=torch.eye(4),
    )


class PrepareSourceTests(unittest.TestCase):
    def test_aligned_inputs_preserve_values_and_capture_validity(self):
        obs = observation()
        obs.depth[0] = torch.tensor([float("nan"), float("inf"), 0, -1])
        mask = torch.ones_like(obs.depth, dtype=torch.bool)
        mask[1, 0] = False
        labels = torch.full_like(obs.depth, 5, dtype=torch.int64)
        source = prepare_source({(7, 9): replace(obs, geometry_mask=mask, labels=labels)})
        aligned = source.observations[7, 9]
        self.assertIs(aligned.rgb, obs.rgb)
        self.assertIs(aligned.depth, obs.depth)
        self.assertIs(aligned.labels, labels)
        expected = mask.clone()
        expected[0] = False
        torch.testing.assert_close(aligned.geometry_mask, expected)
        self.assertEqual(source.cache_keys, ((7, 9),))
        kwargs = source.cache_kwargs()
        self.assertEqual(kwargs["input_image"].shape, (1, 1, 3, 3, 4))
        self.assertEqual(kwargs["input_mask"].shape, (1, 1, 1, 3, 4))
        self.assertNotIn("input_labels", kwargs)

    def test_crop_resize_preserves_ids_validity_and_camera_projection(self):
        obs = observation(4, 6)
        labels = torch.arange(24, dtype=torch.int64).reshape(4, 6) + 2**60
        obs.depth[1, 2] = float("nan")
        source = prepare_source({(0, 0): replace(obs, labels=labels)}, crop=(1, 2, 2, 3), output_size=(4, 6))
        result = source.observations[0, 0]
        expected_labels = labels[1:3, 2:5].repeat_interleave(2, 0).repeat_interleave(2, 1)
        torch.testing.assert_close(result.labels, expected_labels)
        self.assertFalse(result.geometry_mask[:2, :2].any())
        self.assertTrue(result.geometry_mask[2:].all())
        expected_rgb = torch.nn.functional.interpolate(
            obs.rgb[None, :, 1:3, 2:5], size=(4, 6), mode="bilinear", align_corners=False,
        )[0]
        torch.testing.assert_close(result.rgb, expected_rgb)
        point = torch.tensor([0.3, 0.2, 1.0])
        before = obs.intrinsics @ point
        expected_uv = (before[:2] - torch.tensor([2, 1]) + 0.5) * 2 - 0.5
        torch.testing.assert_close((result.intrinsics @ point)[:2], expected_uv)
        torch.testing.assert_close(obs.intrinsics, observation().intrinsics)

    def test_noninteger_resize_uses_half_pixel_sampling(self):
        for n, indices, centers in ((5, [0, 2, 4], [1 / 3, 2, 11 / 3]),
                                    (3, [0, 0, 1, 2, 2], [0, 0.4, 1, 1.6, 2])):
            with self.subTest(input_size=n):
                obs = observation(n, n)
                ramp = torch.arange(n, dtype=torch.float32).expand(n, n)
                depth = 1 + ramp + 10 * ramp.T
                labels = depth.long() + 2**60
                mask = torch.ones(n, n, dtype=torch.bool)
                mask[n // 2, n // 2] = False
                result = prepare_source({(0, 0): replace(
                    obs, rgb=ramp.expand(3, n, n), depth=depth, labels=labels, geometry_mask=mask,
                )}, output_size=(len(indices), len(indices))).observations[0, 0]
                index = torch.tensor(indices)
                for name, original in (("depth", depth), ("labels", labels), ("geometry_mask", mask)):
                    torch.testing.assert_close(getattr(result, name), original[index[:, None], index[None, :]])
                torch.testing.assert_close(result.rgb[0], torch.tensor(centers).expand(len(indices), -1))
                pixel = torch.tensor([1.0, 1.0, 1.0])
                ray = torch.linalg.solve(result.intrinsics, pixel)
                expected_center = (1.5 * n / len(indices)) - 0.5
                torch.testing.assert_close((obs.intrinsics @ ray)[:2], torch.full((2,), expected_center))

    def test_source_shapes_and_mapping(self):
        for frames, views in (([4], [8]), ([4, 7], [8]), ([4], [3, 8]), ([4, 7], [3, 8])):
            inputs = {(f, v): replace(observation(), depth=torch.full((3, 4), float(f + v)))
                      for f in reversed(frames) for v in reversed(views)}
            source = prepare_source(inputs)
            kwargs = source.cache_kwargs()
            self.assertEqual(kwargs["input_depth"].shape, (len(frames), len(views), 1, 3, 4))
            for i, f in enumerate(frames):
                for j, v in enumerate(views):
                    self.assertEqual(kwargs["input_depth"][i, j, 0, 0, 0], f + v)
            self.assertEqual(source.cache_keys, tuple((f, v) for f in frames for v in views))

    def test_sparse_and_variable_size_sources_require_explicit_cache_alignment(self):
        source = prepare_source({(0, 0): observation(), (1, 1): observation()})
        with self.assertRaisesRegex(ValueError, "same views"):
            source.cache_kwargs()
        inputs = {(0, 0): observation(), (0, 1): observation(5, 6)}
        with self.assertRaisesRegex(ValueError, "common image size"):
            prepare_source(inputs).cache_kwargs()
        self.assertEqual(prepare_source(inputs, output_size=(2, 2)).cache_kwargs()["input_image"].shape,
                         (1, 2, 3, 2, 2))

    def test_multiview_video_cache_retains_camera_and_frame_mapping(self):
        from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base

        inputs = {}
        for frame in (5, 2):
            for view in (9, 3):
                obs = observation()
                pose = obs.w2c.double()
                pose[0, 3] = view
                inputs[frame, view] = replace(obs, depth=obs.depth * frame, w2c=pose)
        source = prepare_source(inputs)
        cache = Cache3D_Base(**source.cache_kwargs(), device="cpu")
        self.assertEqual(cache.input_image.shape, (1, 2, 2, 1, 3, 3, 4))
        for i, frame in enumerate((2, 5)):
            for j, view in enumerate((3, 9)):
                # The principal-point pixel lies on camera Z; inverse pose
                # translates it back to x=-view in shared world coordinates.
                torch.testing.assert_close(cache.input_points[0, i, j, 0, 1, 1],
                                           torch.tensor([-view, 0, 2 * frame], dtype=torch.float32))

    def test_source_and_target_camera_conventions_agree(self):
        from cosmos_predict1.diffusion.inference.camera_validation import validate_opencv_cameras

        obs = observation()
        validate_opencv_cameras(obs.w2c, obs.intrinsics)
        validate_opencv_cameras(obs.w2c.repeat(1, 2, 1, 1), obs.intrinsics.repeat(1, 2, 1, 1))
        # Apply the same absolute rigidity tolerance to individual and batched
        # cameras, including matrices just outside the accepted tolerance.
        for delta in (.0004, .0006):
            pose = obs.w2c.clone()
            pose[0, 0] += delta
            for batched in (False, True):
                w2c = pose[None] if batched else pose
                k = obs.intrinsics[None] if batched else obs.intrinsics
                if delta == .0004:
                    validate_opencv_cameras(w2c, k)
                else:
                    with self.assertRaises(ValueError):
                        validate_opencv_cameras(w2c, k)

    def test_rejects_misalignment_and_invalid_cameras(self):
        obs = observation()
        bad_pose = torch.eye(4)
        bad_pose[0, 0] = 2
        bad_k = obs.intrinsics.clone()
        bad_k[0, 0] = -1
        invalid = [
            replace(obs, depth=torch.ones(4, 3)),
            replace(obs, geometry_mask=torch.ones(4, 3)),
            replace(obs, geometry_mask=torch.full((3, 4), float("nan"))),
            replace(obs, labels=torch.ones(3, 4)),
            replace(obs, labels=torch.full((3, 4), -2)),
            replace(obs, intrinsics=bad_k),
            replace(obs, w2c=bad_pose),
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                prepare_source({(0, 0): item})
        # Reject batched cameras at the observation boundary even when their
        # conventions are valid. Crop/resize must not receive batched matrices.
        for count in (1, 2):
            for batch_pose, batch_intrinsics in ((True, False), (False, True), (True, True)):
                item = replace(
                    obs,
                    w2c=obs.w2c.repeat(count, 1, 1) if batch_pose else obs.w2c,
                    intrinsics=obs.intrinsics.repeat(count, 1, 1) if batch_intrinsics else obs.intrinsics,
                )
                for kwargs in ({}, dict(crop=(0, 0, 2, 2)), dict(output_size=(2, 2))):
                    with self.subTest(count=count, pose=batch_pose, intrinsics=batch_intrinsics, kwargs=kwargs):
                        with self.assertRaisesRegex(ValueError, "Observation .* must have shape"):
                            prepare_source({(0, 0): item}, **kwargs)
        with self.assertRaisesRegex(ValueError, "OpenCV"):
            prepare_source({(0, 0): obs}, camera_convention="opengl")
        for kwargs in (dict(crop=(2, 3, 2, 2)), dict(output_size=(0, 4)), dict(crop=(-1, 0, 1, 1))):
            with self.assertRaises(ValueError):
                prepare_source({(0, 0): obs}, **kwargs)
        for inputs in ({}, {(-1, 0): obs}, {(0,): obs}):
            with self.assertRaises(ValueError):
                prepare_source(inputs)

    def test_ray_distance_conversion_and_cache_geometry(self):
        from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base
        from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import unproject_points

        obs = observation()
        obs.w2c[0, 3] = 3  # Conversion must not depend on world translation.
        obs.depth[0, 0] = float("nan")
        source = prepare_source({(0, 0): obs}, is_depth=False)
        aligned = source.observations[0, 0]
        y, x = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
        expected = 2 / torch.sqrt(((x - 1) / 2) ** 2 + ((y - 1) / 2) ** 2 + 1)
        torch.testing.assert_close(aligned.depth[aligned.geometry_mask], expected[aligned.geometry_mask])
        self.assertTrue(torch.isnan(aligned.depth[0, 0]))
        cache = Cache3D_Base(**source.cache_kwargs(), device="cpu")
        points = unproject_points(obs.depth[None, None], obs.w2c[None], obs.intrinsics[None],
                                  is_depth=False, mask=aligned.geometry_mask[None, None])[0]
        torch.testing.assert_close(cache.input_points[0, 0, 0, 0][aligned.geometry_mask],
                                   points[aligned.geometry_mask])
        torch.testing.assert_close(cache.input_mask[0, 0, 0, 0, 0].bool(), aligned.geometry_mask)
        self.assertEqual(cache.input_image.shape, (1, 1, 1, 1, 3, 3, 4))


if __name__ == "__main__":
    unittest.main()
