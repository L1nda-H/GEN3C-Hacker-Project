"""Image-space seed appearance, proxy geometry, and applicability."""
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from tests.test_composite_rendering import source
from tests.test_adaptive_splatting import grid_points
from cosmos_predict1.diffusion.inference.cache3d_composite import Cache3D_Composite
from cosmos_predict1.diffusion.inference.composite_seed import build_image_space_seed, prepare_initial_seed


class CompositeSeedTests(unittest.TestCase):
    def fixture(self, size=(8, 8)):
        k = torch.tensor([[8., 0, 3.5], [0, 8, 3.5], [0, 0, 1]])
        c = Cache3D_Composite(image_size=size)
        env = source('room', grid_points(torch.full((8, 8), 4.), k), size=(8, 8),
                     colors=[[-1., -1, 1]]*64)
        labels = torch.zeros(8, 8, dtype=torch.int64)
        labels[2:6, 2:6] = 1
        obj = source('object', grid_points(torch.full((8, 8), 2.), k), size=(8, 8),
                     ids=labels.flatten().tolist(), colors=[[1., -1, -1]]*64)
        c.add_source('room', env, is_environment=True, source_w2c=torch.eye(4), source_intrinsics=k)
        c.add_source('object', obj, source_w2c=torch.eye(4), source_intrinsics=k)
        c.set_selection('object', [1])
        kt = k.clone()
        for axis, scale in ((0, size[1]/8), (1, size[0]/8)):
            kt[axis] *= scale
            kt[axis, 2] += (scale-1)/2
        return SimpleNamespace(image_size=size), c, kt

    def build(self, fixture, **kwargs):
        scene, c, k = fixture
        return build_image_space_seed(scene, c, torch.eye(4), k, **kwargs)

    def test_identity_and_magnification_have_complete_rgb_and_clean_edges(self):
        for size in ((8, 8), (32, 48)):
            f = self.fixture(size)
            result = self.build(f)
            rgb = result.rgb[0, :, 0]
            self.assertTrue(result.diagnostics['observed'].all())
            self.assertFalse(result.diagnostics['synthesized'].any())
            self.assertFalse(result.diagnostics['uncertainty'].any())
            # Red foreground + blue background, including alpha edges: no black halo.
            torch.testing.assert_close(rgb[0] + rgb[2], torch.zeros(size))
            self.assertTrue((rgb[1] == -1).all())
            self.assertEqual(float(rgb[0, size[0]//2, size[1]//2]), 1.)
            self.assertEqual(float(rgb[2, 0, 0]), 1.)

    def test_depth_occlusion_and_plane_override(self):
        f = self.fixture()
        behind = self.build(f, plane_depths={'object:1': 5.})
        self.assertTrue((behind.rgb[0, 2, 0] == 1).all())
        front = self.build(f, plane_depths={'object:1': 1.})
        self.assertEqual(front.metadata['planes'][0]['depth_method'], 'override')
        self.assertEqual(front.diagnostics['ownership'][3, 3, 1], 1)
        with self.assertRaisesRegex(ValueError, 'Unused'):
            self.build(f, plane_depths={'wrong:1': 2.})
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            self.build(f, plane_depths={'object:1': float('nan')})

    def test_noisy_object_depth_does_not_create_slivers(self):
        f = self.fixture((32, 48))
        before = self.build(f)
        child = f[1].sources['object']
        # Perturb one object's sample along its ray; median plane stays identical.
        child.input_points[..., 3, 3, :] *= 1.4
        after = self.build(f)
        torch.testing.assert_close(before.rgb, after.rgb)
        np.testing.assert_array_equal(before.diagnostics['ownership'], after.diagnostics['ownership'])

    def test_uncertain_occlusion_is_explicit_and_saved(self):
        f = self.fixture()
        f[1].sources['room'].input_mask[..., 3, 3] = False
        with self.assertRaisesRegex(ValueError, 'Uncertain environment occlusion'):
            self.build(f)
        result = self.build(f, occlusion_policy='allow')
        self.assertGreater(result.metadata['uncertain_pixels'], 0)
        self.assertTrue(result.diagnostics['uncertainty'][3, 3])

    def test_background_rgb_survives_missing_depth_without_object_overlap(self):
        f = self.fixture()
        f[1].sources['room'].input_mask[..., 0, 0] = False
        result = self.build(f)
        self.assertEqual(float(result.rgb[0, 2, 0, 0, 0]), 1.)
        self.assertFalse(result.diagnostics['environment_depth_valid'][0, 0])

    def test_pose_fov_removal_and_degenerate_proxy_rejected(self):
        scene, c, k = self.fixture()
        moved = torch.eye(4); moved[0, 3] = .1
        with self.assertRaisesRegex(ValueError, 'first camera'):
            build_image_space_seed(scene, c, moved, k)
        wide = k.clone(); wide[0, 0] /= 2
        with self.assertRaisesRegex(ValueError, 'field of view'):
            build_image_space_seed(scene, c, torch.eye(4), wide)
        c.sources['room'].input_labels[..., 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'removed or unresolved'):
            self.build((scene, c, k))
        c.sources['room'].input_labels[..., 0, 0] = 0
        transform = torch.eye(4); transform[2, 3] = -3
        c.set_transform('object', transform)
        with self.assertRaisesRegex(ValueError, 'near-plane'):
            self.build((scene, c, k))

    def test_crossing_planes_resolve_per_pixel_and_ties_are_stable(self):
        f = self.fixture((32, 32))
        scene, c, k = f
        obj = c.sources['object']
        other = source('second', obj.input_points.reshape(-1, 3), size=(8, 8),
                       ids=obj.input_labels.flatten().tolist(), colors=[[-1., 1, -1]]*64)
        c.add_source('second', other, source_w2c=torch.eye(4),
                     source_intrinsics=c._source_cameras['object'][1])
        c.set_selection('second', [1])
        tied = self.build(f)
        self.assertEqual(tied.diagnostics['ownership'][16, 16, 0], 0)  # object wins tie
        angle = .35
        transform = torch.eye(4)
        transform[:3, :3] = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                         [-np.sin(angle), 0, np.cos(angle)]], dtype=torch.float32)
        transform[:3, 3] = torch.tensor([-2*np.sin(angle), 0, 2-2*np.cos(angle)])
        c.set_transform('second', transform)
        result = self.build(f)
        # A plane rotated about its center crosses the original plane.
        ids = result.diagnostics['ownership'][16, 12:20, 0]
        self.assertIn(0, ids)
        self.assertIn(2, ids)

    def test_provided_seed_validation_and_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'seed.png'
            Image.new('RGB', (8, 8), 'red').save(path)
            result = prepare_initial_seed(SimpleNamespace(image_size=(8, 8)), None,
                                          strategy='provided', provided_seed=path)
            self.assertEqual(len(result.metadata['sha256']), 64)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                prepare_initial_seed(SimpleNamespace(image_size=(9, 8)), None,
                                     strategy='provided', provided_seed=path)
            with self.assertRaisesRegex(ValueError, 'required only'):
                prepare_initial_seed(None, None, provided_seed=path)


if __name__ == '__main__':
    unittest.main()
