"""Exercise CLI edits through labeling, persisted scenes and real CPU rendering."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from scripts import evaluate_real_cache_mvp as mvp
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    SAM21Segmenter, SourceObservation, prepare_source,
)


class RealCacheMVPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.size = (6, 8)
        self.addCleanup(patch.stopall)
        patch.object(mvp, 'IMAGE_SIZE', self.size).start()
        self.load = patch.object(mvp, 'load_source', side_effect=self.source).start()
        predictor = Mock()
        predictor.predict.side_effect = self.predict
        self.sam_loader = patch.object(
            mvp.SAM21Segmenter, 'from_checkpoint', return_value=SAM21Segmenter(predictor),
        ).start()

    def source(self, batch):
        rgb = torch.full((3, *self.size), -1.)
        rgb[2 if batch == 'batch_0004' else 0] = 1.
        intrinsics = torch.tensor([[4., 0, 3], [0, 4., 2], [0, 0, 1.]])
        return prepare_source({mvp.KEY: SourceObservation(
            rgb, torch.full(self.size, 2.), intrinsics, torch.eye(4),
        )})

    def predict(self, *, box, **kwargs):
        x1, y1, x2, y2 = map(int, box)
        mask = np.zeros(self.size, dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask[None], np.array([0.9]), None

    def run_scene(self, *args):
        with redirect_stdout(io.StringIO()):
            mvp.main([*args, '--output-dir', str(self.root)])
        return mvp.SceneSpec.load(self.root / 'scene.json')

    def test_removal_only_excludes_multiple_objects_and_needs_no_import_source(self):
        scene = self.run_scene('--remove-environment-box', '1', '1', '3', '3',
                               '--remove-environment-box', '4', '1', '6', '3')
        self.load.assert_called_once_with('batch_0004')
        self.assertEqual([(s.source_id, s.selected_ids) for s in scene.sources], [('environment', [0])])
        self.assertFalse((self.root / 'object_input.png').exists())
        self.assertAlmostEqual(float(scene.cameras()[0][0, -1, 0, 3]), -0.012)
        with np.load(self.root / 'render_000000.npz') as rendered:
            coverage = rendered['coverage'][0, 0, 0, 0]
            self.assertFalse(coverage[1:3, 1:3].any())
            self.assertFalse(coverage[1:3, 4:6].any())
            self.assertEqual(coverage[4, 4], 1)
            self.assertTrue((rendered['rgb'][0, 0, 0, :, 1:3, 1:3] == -1).all())
        self.assertTrue((self.root / 'environment_labels.png').exists())
        self.assertTrue((self.root / 'preview.gif').exists())

    def test_import_remains_available_with_or_without_removal(self):
        root = self.root
        for remove in (False, True):
            with self.subTest(remove=remove):
                self.root = root / ('combined' if remove else 'import')
                args = ['--box', '1', '1', '3', '4', '--target-base', '6', '5', '--object-height', '3']
                if remove:
                    args.extend(['--remove-environment-box', '1', '1', '3', '3'])
                scene = self.run_scene(*args)
                self.assertEqual([s.source_id for s in scene.sources], ['environment', 'object'])
                self.assertEqual(scene.sources[0].selected_ids, [0])
                self.assertEqual(len(scene.sources[1].selected_ids), 1)
                self.assertGreater(scene.sources[1].selected_ids[0], 0)
                with np.load(self.root / 'render_000000.npz') as rendered:
                    self.assertEqual(rendered['coverage'].shape[2], 2)
                    environment_coverage = rendered['coverage'][0, 0, 0, 0]
                    self.assertEqual(bool(environment_coverage[1:3, 1:3].all()), not remove)
                    self.assertTrue(rendered['coverage'][0, 0, 1].any())

    def test_preview_recombination_matches_joint_without_duplicate_projection(self):
        scene = self.run_scene('--box', '1', '1', '3', '4',
                               '--remove-environment-box', '1', '1', '3', '3',
                               '--target-base', '6', '5', '--object-height', '3')
        cache = mvp.build_composite_scene(scene, device='cpu')
        poses, intrinsics = scene.cameras()
        root = self.root / 'recombined'
        with patch.object(cache, 'render_cache', side_effect=AssertionError('Duplicate projection')), \
                redirect_stdout(io.StringIO()):
            mvp.export_previews(cache, poses, intrinsics, root)
        for frame in range(poses.shape[1]):
            rgb, _ = cache.render_cache(poses[:, frame:frame + 1], intrinsics[:, frame:frame + 1])
            np.testing.assert_array_equal(np.asarray(Image.open(root / f'joint_reference_{frame:06d}.png')),
                                          np.asarray(mvp.rgb_image(rgb[0, 0, 0])))
            if frame == 0:
                np.testing.assert_array_equal(np.load(root / 'seed.npy'), rgb[:, 0, 0].unsqueeze(2))
        metadata = json.loads((root / 'preview.json').read_text())
        self.assertEqual(metadata['conditioning_visibility'], 'joint_nearest_surface')
        self.assertEqual(metadata['depth_tolerance'], cache.depth_tolerance)

    def test_depth_multiplier_places_import_behind_environment_and_reuses_scene(self):
        scene = self.run_scene('--box', '1', '1', '3', '4', '--target-base', '6', '5',
                               '--object-height', '3')
        with np.load(self.root / 'render_000000.npz') as before:
            self.assertTrue(before['coverage'][0, 0, 1].any())
        self.sam_loader.reset_mock()
        self.load.reset_mock()
        output = self.root / 'behind'
        with redirect_stdout(io.StringIO()):
            mvp.main(['--scene-spec', str(self.root / 'scene.json'),
                      '--object-depth-multiplier', '1.15', '--output-dir', str(output)])
        self.sam_loader.assert_not_called()
        self.load.assert_not_called()
        with np.load(output / 'render_000000.npz') as after:
            self.assertFalse(after['coverage'][0, 0, 1].any())
            self.assertTrue((after['rgb'][0, 0, 1] == -1).all())
        restored = mvp.SceneSpec.load(output / 'scene.json')
        self.assertEqual(restored.sources[0].source_transform, scene.sources[0].source_transform)
        torch.testing.assert_close(restored.cameras()[0], scene.cameras()[0])
        # The same option applies when preparing a fresh import.
        self.root = self.root / 'fresh'
        self.run_scene('--box', '1', '1', '3', '4', '--target-base', '6', '5',
                       '--object-height', '3', '--object-depth-multiplier', '1.15')
        with np.load(self.root / 'render_000000.npz') as after:
            self.assertFalse(after['coverage'][0, 0, 1].any())

    def test_depth_adjustment_preserves_projection_with_nonidentity_camera(self):
        scene = self.run_scene('--box', '1', '1', '3', '4', '--target-base', '6', '5',
                               '--object-height', '3')
        pose = torch.eye(4, dtype=torch.float64)
        pose[:2, :2] = torch.tensor([[0., -1.], [1., 0.]])
        pose[:3, 3] = torch.tensor([.2, .3, .4])
        scene.target_w2cs = pose.repeat(1, 24, 1, 1)
        source = scene.sources[1]
        source.object_transforms = {1: torch.eye(4)}
        source.object_transforms[1][0, 3] = .1

        def projected():
            cache = mvp.build_composite_scene(scene, device='cpu')
            samples = next(s for s in cache._iter_render_inputs() if s.source_id == 'object')
            camera_points = samples.points.double() @ pose[:3, :3].T + pose[:3, 3]
            return camera_points[:, :2] / camera_points[:, 2:], camera_points[:, 2]

        uv_before, z_before = projected()
        mvp.adjust_object_depth(scene, 1.15)
        uv_after, z_after = projected()
        torch.testing.assert_close(uv_after, uv_before, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(z_after, z_before * 1.15, atol=1e-6, rtol=1e-5)

    def test_no_arguments_still_exports_inputs_without_sam(self):
        with redirect_stdout(io.StringIO()):
            mvp.main(['--output-dir', str(self.root)])
        self.sam_loader.assert_not_called()
        self.assertTrue((self.root / 'environment_input.png').exists())
        self.assertTrue((self.root / 'object_input.png').exists())
        self.assertFalse((self.root / 'scene.json').exists())

    def test_invalid_boxes_and_saved_scene_removal_fail_before_loading_inputs(self):
        for args in (
            ['--box', '1', '1', '3', '4', '--object-depth-multiplier', 'nan'],
            ['--scene-spec', 'unused.json', '--object-depth-multiplier', '0'],
            ['--object-depth-multiplier', '1.15'],
            ['--remove-environment-box', '3', '1', '1', '3'],
            ['--remove-environment-box', '1', '1', 'nan', '3'],
            ['--remove-environment-box', '1', '1', '8', '3'],
            ['--scene-spec', 'unused.json', '--remove-environment-box', '1', '1', '3', '3'],
        ):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mvp.main(args)
        self.load.assert_not_called()
        self.sam_loader.assert_not_called()

    def test_empty_removal_mask_is_reported(self):
        self.sam_loader.return_value.predictor.predict.side_effect = lambda **kwargs: (
            np.zeros((1, *self.size), dtype=bool), np.array([0.9]), None,
        )
        with self.assertRaisesRegex(ValueError, 'No mask returned for environment removal boxes'):
            self.run_scene('--remove-environment-box', '1', '1', '3', '3')
        self.assertFalse((self.root / 'scene.json').exists())


if __name__ == '__main__':
    unittest.main()
