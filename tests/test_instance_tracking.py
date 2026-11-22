"""Contract tests for the pinned SAM public video API; no weights required."""

import sys
import hashlib
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import torch

from cosmos_predict1.diffusion.inference.instance_segmentation import (
    InstanceObservation, SAM21_MODELS, SAM21_REVISION, SegmentPrompts, SourceObservation, prepare_source,
)
from cosmos_predict1.diffusion.inference.instance_tracking import SAM21VideoSegmenter
from cosmos_predict1.diffusion.inference.instance_labeling import label_source, load_labeled_source, save_labeled_source


def source(keys=((20, 7), (2, 7), (90, 7))):
    return prepare_source({key: SourceObservation(
        torch.full((3, 4, 6), index / 2), torch.full((4, 6), float('nan')),
        torch.eye(3), torch.eye(4),
    ) for index, key in enumerate(keys)})


def seed(key, obj_id, mask=None, name=None):
    if mask is None:
        mask = torch.ones(4, 6, dtype=torch.bool)
    return InstanceObservation(*key, obj_id, mask, 0.9, name or f'Object {obj_id}')


class VideoPredictor:
    """Returns reversed row order and direction-dependent outputs to detect misuse."""

    def __init__(self):
        self.inputs = {}
        self.calls = []
        self.reset_count = 0

    def init_state(self, video_path, async_loading_frames):
        assert async_loading_frames is False
        files = sorted(Path(video_path).iterdir())
        self.filenames = [p.name for p in files]
        self.pixels = [Image.open(p).getpixel((0, 0)) for p in files]
        return {'num_frames': len(files)}

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.inputs[frame_idx, obj_id] = mask.clone()

    def add_new_points_or_box(self, **kwargs):
        self.prompt_args = kwargs
        self.inputs[kwargs['frame_idx'], kwargs['obj_id']] = torch.ones(4, 6, dtype=torch.bool)

    def reset_state(self, state):
        self.reset_count += 1
        self.inputs.clear()

    def propagate_in_video(self, state, start_frame_idx, reverse):
        self.calls.append((start_frame_idx, reverse))
        offsets = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, state['num_frames'])
        ids = sorted({i for _, i in self.inputs}, reverse=True)
        for offset in offsets:
            logits = []
            for obj_id in ids:
                candidates = [(t, mask) for (t, i), mask in self.inputs.items() if i == obj_id]
                _, mask = min(candidates, key=lambda item: abs(item[0] - offset))
                # A fully occluded object, plus zeros that must be background.
                values = mask.float() if offset != 1 else torch.zeros_like(mask).float()
                logits.append(values[None])
            yield offset, ids, torch.stack(logits)


class VideoTrackingTests(unittest.TestCase):
    def initialized(self):
        predictor = VideoPredictor()
        tracker = SAM21VideoSegmenter(predictor)
        prepared = source()
        tracker.initialize(prepared, source_id='scene', rgb_range=(0, 1))
        return tracker, predictor, prepared

    def test_ordered_export_sparse_keys_and_unchanged_source(self):
        tracker, predictor, prepared = self.initialized()
        self.assertEqual(tracker.frame_keys, ((2, 7), (20, 7), (90, 7)))
        self.assertEqual(predictor.filenames, ['00000000.jpg', '00000001.jpg', '00000002.jpg'])
        self.assertEqual([pixel[0] for pixel in predictor.pixels], [128, 0, 255])
        self.assertIs(tracker.source, prepared)
        self.assertFalse(prepared.observations[2, 7].geometry_mask.any())

    def test_two_tracks_rows_occlusion_ambiguity_and_environment(self):
        tracker, predictor, prepared = self.initialized()
        a = torch.zeros(4, 6, dtype=torch.bool)
        a[:, :3] = True
        tracker.seed_objects((2, 7), [seed((2, 7), 3, a, 'Cup'), seed((2, 7), 8)])
        confirmed = torch.ones(4, 6, dtype=torch.bool)
        result = tracker.propagate(environment_masks={(20, 7): confirmed})
        self.assertIs(result.prepared_source, prepared)
        self.assertEqual(set(result.observations), set(prepared.observations))
        for key, obs in result.observations.items():
            self.assertEqual([i.local_id for i in obs.instances], [3, 8])
            self.assertEqual(obs.instances[0].name, 'Cup')
            self.assertEqual(obs.instances[0].track_id, ('scene', 7, tracker.session_id, 3))
            self.assertTrue(all(i.confidence is None for i in obs.instances))
        torch.testing.assert_close(result.observations[2, 7].instances[0].mask, a)
        torch.testing.assert_close(result.observations[2, 7].ambiguous_mask, a)
        hidden = result.observations[20, 7]
        self.assertFalse(any(i.mask.any() for i in hidden.instances))
        self.assertTrue(hidden.environment_mask.all())
        self.assertFalse(hidden.unassigned_mask.any())
        self.assertFalse(result.observations[90, 7].environment_mask.any())
        self.assertTrue(confirmed.all())

    def test_late_seed_and_correction_replay_preserve_identity_and_snapshot(self):
        tracker, predictor, _ = self.initialized()
        tracker.seed_objects((20, 7), [seed((20, 7), 3)])
        first = tracker.propagate()
        self.assertEqual(predictor.calls, [(1, False), (1, True)])
        old_track = first.observations[2, 7].instances[0].track_id
        tracker.seed_objects((90, 7), [seed((90, 7), 9)])
        self.assertIsNone(tracker.result)
        corrected = torch.zeros(4, 6, dtype=torch.bool)
        corrected[:, 4:] = True
        tracker.correct_object((20, 7), seed((20, 7), 3, corrected, 'Renamed'))
        second = tracker.propagate(boundary_width=1)
        self.assertEqual(predictor.reset_count, 1)
        self.assertEqual(predictor.calls[-2:], [(1, False), (2, True)])
        self.assertEqual(second.observations[2, 7].instances[0].track_id, old_track)
        self.assertEqual(len(second.observations[2, 7].instances), 2)
        self.assertNotEqual(first.revision, second.revision)
        torch.testing.assert_close(second.observations[2, 7].instances[0].mask, corrected)
        self.assertTrue(first.observations[2, 7].instances[0].mask.all())
        self.assertEqual(second.observations[2, 7].instances[0].name, 'Renamed')
        self.assertTrue(second.observations[2, 7].ambiguous_mask.any())

    def test_click_box_seeding_and_prompt_replacement(self):
        tracker, predictor, _ = self.initialized()
        prompt = SegmentPrompts(42, points=torch.tensor([[1., 2.]]), point_labels=torch.tensor([1]),
                                box=torch.tensor([0., 0., 4., 3.]), name='Mug')
        tracker.seed_objects((2, 7), [prompt])
        args = predictor.prompt_args
        self.assertEqual(args['frame_idx'], 0)
        self.assertEqual(args['obj_id'], 42)
        self.assertTrue(args['clear_old_points'])
        self.assertTrue(args['normalize_coords'])
        self.assertNotIn('multimask_output', args)
        tracker.propagate()
        tracker.correct_object((2, 7), replace(prompt, name=None, points=torch.tensor([[2., 2.]])))
        result = tracker.propagate()
        self.assertEqual(result.observations[2, 7].instances[0].name, 'Mug')
        self.assertEqual(predictor.prompt_args['points'].tolist(), [[2., 2.]])

    def test_session_namespaces_and_input_validation(self):
        tracker, predictor, prepared = self.initialized()
        old_session = tracker.session_id
        tracker.initialize(prepared, source_id='scene', rgb_range=(0, 1))
        self.assertNotEqual(tracker.session_id, old_session)
        with self.assertRaisesRegex(RuntimeError, 'seed'):
            tracker.propagate()
        with self.assertRaisesRegex(ValueError, 'camera'):
            tracker.initialize(source(((0, 1), (1, 2))), source_id='x', rgb_range=(0, 1))
        with self.assertRaisesRegex(ValueError, 'existing'):
            tracker.correct_object((2, 7), seed((2, 7), 4))
        with self.assertRaisesRegex(ValueError, 'unknown observation'):
            tracker.seed_objects((99, 7), [seed((99, 7), 4)])
        with self.assertRaisesRegex(ValueError, 'unique'):
            tracker.seed_objects((2, 7), [seed((2, 7), 4)] * 2)
        tracker.seed_objects((2, 7), [seed((2, 7), 4)])
        with self.assertRaisesRegex(ValueError, 'boundary_width'):
            tracker.propagate(boundary_width=-1)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            tracker.propagate(environment_masks={(99, 7): torch.ones(4, 6, dtype=torch.bool)})
        with self.assertRaisesRegex(ValueError, 'Confirmed environment'):
            tracker.propagate(environment_masks={(2, 7): torch.ones(4, 6)})

    def test_malformed_or_incomplete_outputs_fail(self):
        tracker, predictor, _ = self.initialized()
        tracker.seed_objects((2, 7), [seed((2, 7), 4)])
        bad_outputs = [
            [(0, [9], torch.ones(1, 1, 4, 6))],
            [(0, [4], torch.ones(1, 1, 2, 3))],
            [(99, [4], torch.ones(1, 1, 4, 6))], [],
        ]
        for outputs in bad_outputs:
            with patch.object(predictor, 'propagate_in_video', return_value=iter(outputs)):
                with self.assertRaises(ValueError):
                    tracker.propagate()
                self.assertIsNone(tracker.result)

    def test_checkpoint_factory_shared_config(self):
        build = Mock()
        modules = {'sam2': types.ModuleType('sam2'),
                   'sam2.build_sam': types.SimpleNamespace(build_sam2_video_predictor=build)}
        with tempfile.NamedTemporaryFile(suffix='.pt') as checkpoint, patch.dict(sys.modules, modules):
            tracker = SAM21VideoSegmenter.from_checkpoint(checkpoint.name, model_size='small')
            build.assert_called_once_with(SAM21_MODELS['small'][0], checkpoint.name, device='cuda', mode='eval')
            self.assertIs(tracker.predictor, build.return_value)
            self.assertEqual(tracker.provenance.sam_revision, SAM21_REVISION)
            self.assertEqual(tracker.provenance.configuration, SAM21_MODELS['small'][0])
            self.assertEqual(tracker.provenance.checkpoint_sha256, hashlib.sha256(b'').hexdigest())
        with self.assertRaises(ValueError):
            SAM21VideoSegmenter.from_checkpoint('unused', model_size='unknown')
        with self.assertRaises(FileNotFoundError):
            SAM21VideoSegmenter.from_checkpoint('/missing/checkpoint.pt')

    def test_tracking_labeling_reload_and_correction_keep_canonical_id(self):
        tracker, _, prepared = self.initialized()
        # Keep missing depth elsewhere; one frame supplies labeled foreground points.
        prepared.observations[2, 7].depth[:] = 1
        prepared.observations[2, 7].geometry_mask[:] = True
        tracker.seed_objects((2, 7), [seed((2, 7), 42, name='Cup')])
        first = label_source(tracker.propagate(), source_id='scene')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels'
            save_labeled_source(path, first)
            loaded = load_labeled_source(path, prepared)
            corrected = torch.zeros(4, 6, dtype=torch.bool)
            corrected[:, 3:] = True
            tracker.correct_object((2, 7), seed((2, 7), 42, corrected, 'Cup'))
            updated = label_source(tracker.propagate(), source_id='scene', registry=loaded.registry)
            self.assertEqual(updated.registry.track_to_id, first.registry.track_to_id)
            self.assertEqual(len(updated.registry.names), 1)
            self.assertTrue((updated.labels[2, 7][:, 3:] == 1).all())
            self.assertTrue((updated.labels[2, 7][:, :3] == -1).all())
            self.assertTrue((loaded.labels[2, 7] == 1).all())


if __name__ == '__main__':
    unittest.main()
