"""CPU tests for identity, label semantics, and editable persistence."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.instance_labeling import (
    ObjectRegistry, label_source, load_labeled_source, prepared_source_fingerprint, save_labeled_source,
)
from cosmos_predict1.diffusion.inference.instance_segmentation import (
    InstanceProposal, ModelProvenance, PreparedSource, SAM21Segmenter, SegmentedSource, SourceObservation,
    _segmented_observation, file_sha256, prepare_source,
    segment_source,
)


def prepared(keys=((4, 7),), h=4, w=6):
    return prepare_source({key: SourceObservation(torch.zeros(3, h, w), torch.ones(h, w),
                                                 torch.eye(3), torch.eye(4)) for key in keys})


def mask(left=0, right=3, h=4, w=6):
    result = torch.zeros(h, w, dtype=torch.bool)
    result[:, left:right] = True
    return result


def segmented(source, proposals=None, *, environment=None, revision=None):
    observations = {key: _segmented_observation(key, obs, (proposals or {}).get(key, ()),
                                               None if environment is None else environment.get(key))
                    for key, obs in source.observations.items()}
    kwargs = {} if revision is None else dict(revision=revision)
    return SegmentedSource(source, observations, **kwargs)


def track(obj_id, session='session', source_id='scene', view_id=7):
    return (source_id, view_id, session, obj_id)


def proposal(obj_id, visible=None, *, tracked=True, name=None, session='session'):
    return InstanceProposal(mask() if visible is None else visible, None, obj_id, name,
                            track_id=track(obj_id, session) if tracked else None)


class RegistryTests(unittest.TestCase):
    def test_allocate_bind_merge_chain_and_no_id_reuse(self):
        registry = ObjectRegistry('scene')
        a, b, c = registry.allocate('Cup'), registry.allocate(), registry.allocate('Bowl')
        registry.bind_observation((4, 7, 8), a, revision='r1', track_id=track(8))
        registry.bind_observation((9, 7, 8), a, revision='r1', track_id=track(8))
        registry.merge_ids(a, b)
        registry.merge_ids(b, c)
        self.assertEqual(registry.resolve(a), c)
        self.assertEqual(registry.track_to_id[track(8)], c)
        self.assertTrue(all(i == c for i in registry.observation_to_id.values()))
        self.assertEqual(registry.allocate(), 4)
        self.assertEqual(registry.names[c], 'Bowl')

    def test_conflicts_and_invalid_bindings(self):
        registry = ObjectRegistry('scene')
        a, b = registry.allocate(), registry.allocate()
        registry.bind_observation((4, 7, 8), a, revision='r1', track_id=track(8))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            registry.bind_observation((4, 7, 8), b, revision='r1')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            registry.bind_observation((9, 7, 8), b, revision='r2', track_id=track(8))
        with self.assertRaisesRegex(ValueError, 'namespace'):
            registry.bind_observation((4, 7, 8), a, revision='r2', track_id=track(8, source_id='other'))
        for invalid in (-1, 0, True, 2**63):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                registry.resolve(invalid)


class LabelingTests(unittest.TestCase):
    def test_geometry_overlap_boundary_and_confirmed_environment(self):
        source = prepared()
        source.observations[4, 7].geometry_mask[0] = False
        a, b = mask(0, 3), mask(2, 4)
        snapshot = segmented(source, {(4, 7): [proposal(8, a), proposal(3, b)]},
                             environment={(4, 7): torch.ones(4, 6, dtype=torch.bool)})
        snapshot.observations[4, 7].ambiguous_mask[1, 1] = True
        result = label_source(snapshot, source_id='scene')
        labels = result.labels[4, 7]
        # Deterministic allocation sorts local IDs: 3 -> 1 and 8 -> 2.
        self.assertTrue((labels[0] == -1).all())
        self.assertTrue((labels[2:, :2] == 2).all())
        self.assertTrue((labels[:, 2] == -1).all())
        self.assertTrue((labels[1:, 3] == 1).all())
        self.assertTrue((labels[1:, 4:] == 0).all())
        self.assertEqual(labels[1, 1], -1)
        self.assertEqual(labels.dtype, torch.int64)
        self.assertTrue(snapshot.observations[4, 7].instances[0].mask[0].any())
        self.assertTrue(snapshot.observations[4, 7].confirmed_environment_mask.all())
        self.assertIsNone(source.observations[4, 7].labels)

    def test_unassigned_never_becomes_environment(self):
        source = prepared()
        result = label_source(segmented(source, {(4, 7): [proposal(5)]}), source_id='scene')
        self.assertTrue((result.labels[4, 7][:, 3:] == -1).all())
        self.assertTrue((result.labels[4, 7][:, :3] == 1).all())

    def test_image_snapshots_have_fresh_revisions_and_provenance(self):
        source = prepared()
        provenance = ModelProvenance('revision', 'sam-config', 'a' * 64, dict(automatic_mask_options={}))
        segmenter = SAM21Segmenter(None, provenance=provenance)
        a = segment_source(source, segmenter, rgb_range=(0, 1), prompts={(4, 7): []}, boundary_width=1)
        b = segment_source(source, segmenter, rgb_range=(0, 1), prompts={(4, 7): []})
        self.assertNotEqual(a.revision, b.revision)
        self.assertEqual(a.provenance.settings['rgb_range'], (0, 1))
        self.assertEqual(a.provenance.settings['boundary_width'], 1)
        self.assertEqual(a.provenance.settings['mode'], 'image')
        self.assertEqual(provenance.settings, dict(automatic_mask_options={}))

    def test_tracks_survive_occlusion_and_missing_depth(self):
        source = prepared(((4, 7), (15, 7), (90, 7)))
        source.observations[90, 7].geometry_mask[:] = False
        empty = torch.zeros(4, 6, dtype=torch.bool)
        snapshot = segmented(source, {
            (4, 7): [proposal(8)], (15, 7): [proposal(8, empty)], (90, 7): [proposal(8)],
        })
        result = label_source(snapshot, source_id='scene')
        self.assertEqual(len(result.registry.names), 1)
        self.assertEqual(set(result.registry.observation_to_id.values()), {1})
        self.assertTrue((result.labels[15, 7] == -1).all())
        self.assertTrue((result.labels[90, 7] == -1).all())
        self.assertTrue(result.segmented_source.observations[90, 7].instances[0].mask.any())

    def test_correction_preserves_track_identity_and_previous_snapshot(self):
        source = prepared()
        first = label_source(segmented(source, {(4, 7): [proposal(8, name='Cup')]}), source_id='scene')
        corrected = segmented(source, {(4, 7): [proposal(8, mask(3, 6)), proposal(3, mask(0, 1))]})
        second = label_source(corrected, source_id='scene', registry=first.registry)
        self.assertNotEqual(first.segmented_source.revision, corrected.revision)
        self.assertTrue((second.labels[4, 7][:, 3:] == 1).all())
        self.assertEqual(second.registry.track_to_id[track(3)], 2)
        self.assertEqual(len(first.registry.names), 1)
        self.assertTrue((first.labels[4, 7][:, :3] == 1).all())

    def test_images_rediscovery_and_new_sessions_need_explicit_links(self):
        source = prepared()
        first = label_source(segmented(source, {(4, 7): [proposal(3, tracked=False)]}), source_id='scene')
        rediscovered = segmented(source, {(4, 7): [proposal(3, tracked=False)]})
        second = label_source(rediscovered, source_id='scene', registry=first.registry)
        self.assertTrue((second.labels[4, 7][:, :3] == 2).all())
        linked = label_source(rediscovered, source_id='scene', registry=first.registry, identity_links={(4, 7, 3): 1})
        self.assertTrue((linked.labels[4, 7][:, :3] == 1).all())
        video = label_source(segmented(source, {(4, 7): [proposal(3)]}), source_id='scene')
        restarted = segmented(source, {(4, 7): [proposal(3, session='new-session')]})
        new = label_source(restarted, source_id='scene', registry=video.registry)
        self.assertEqual(new.registry.track_to_id[track(3, 'new-session')], 2)
        with self.assertRaisesRegex(ValueError, 'different source'):
            label_source(rediscovered, source_id='other', registry=first.registry)

    def test_independent_image_observations_and_deterministic_order(self):
        source = prepared(((90, 7), (4, 7)))
        objects = {key: [proposal(3, tracked=False)] for key in source.observations}
        snapshot = segmented(source, objects)
        first = label_source(snapshot, source_id='scene')
        reordered = replace(snapshot, observations=dict(reversed(list(snapshot.observations.items()))))
        second = label_source(reordered, source_id='scene')
        self.assertEqual(first.registry.observation_to_id, second.registry.observation_to_id)
        self.assertEqual(len(first.registry.names), 2)

    def test_merge_rebuild_and_same_revision_mutation_rejected(self):
        source = prepared()
        snapshot = segmented(source, {(4, 7): [proposal(3), proposal(8, mask(3, 6))]})
        result = label_source(snapshot, source_id='scene')
        result.registry.merge_ids(2, 1)
        rebuilt = label_source(snapshot, source_id='scene', registry=result.registry)
        self.assertTrue((rebuilt.labels[4, 7] == 1).all())
        self.assertTrue((result.labels[4, 7][:, 3:] == 2).all())
        snapshot.observations[4, 7].instances[0].mask[:, 0] = False
        with self.assertRaisesRegex(ValueError, 'new revision'):
            label_source(snapshot, source_id='scene', registry=rebuilt.registry)

    def test_invalid_or_duplicate_observations(self):
        source = prepared()
        for objects in ([proposal(3), proposal(3)], [proposal(0)]):
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                snapshot = segmented(source, {(4, 7): objects})
                label_source(snapshot, source_id='scene')
        snapshot = segmented(source, {(4, 7): [proposal(3)]})
        obs = snapshot.observations[4, 7]
        invalid = replace(obs.instances[0], mask=obs.instances[0].mask.float())
        snapshot = replace(snapshot, observations={(4, 7): replace(obs, instances=(invalid,))})
        with self.assertRaisesRegex(ValueError, 'boolean'):
            label_source(snapshot, source_id='scene')
        with self.assertRaisesRegex(ValueError, 'exactly'):
            label_source(SegmentedSource(source, {}), source_id='scene')


class PersistenceTests(unittest.TestCase):
    def result(self):
        source = prepared(((4, 7), (90, 7)))
        source.observations[4, 7].geometry_mask[0] = False
        snapshot = segmented(source, {key: [proposal(8)] for key in source.observations},
                             environment={key: torch.ones(4, 6, dtype=torch.bool) for key in source.observations})
        snapshot = replace(snapshot, provenance=ModelProvenance('revision', 'sam-config', 'a' * 64,
                                                               dict(boundary_width=0)))
        return label_source(snapshot, source_id='scene')

    def test_editable_roundtrip_and_all_metadata(self):
        result = self.result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels'
            save_labeled_source(path, result)
            loaded = load_labeled_source(path, result.prepared_source)
            self.assertEqual(loaded.registry, result.registry)
            self.assertEqual(loaded.segmented_source.revision, result.segmented_source.revision)
            self.assertEqual(loaded.segmented_source.provenance, result.segmented_source.provenance)
            self.assertEqual(set(loaded.labels), {(4, 7), (90, 7)})
            for key in result.labels:
                torch.testing.assert_close(loaded.labels[key], result.labels[key])
                before = result.segmented_source.observations[key]
                after = loaded.segmented_source.observations[key]
                self.assertIsNone(after.instances[0].confidence)
                self.assertEqual(before.instances[0].track_id, after.instances[0].track_id)
                for name in ('ambiguous_mask', 'unassigned_mask', 'environment_mask', 'confirmed_environment_mask'):
                    torch.testing.assert_close(getattr(before, name), getattr(after, name))
                torch.testing.assert_close(before.instances[0].mask, after.instances[0].mask)
            # Recover confirmation hidden by an object and invalid geometry.
            self.assertTrue(loaded.segmented_source.observations[4, 7].confirmed_environment_mask.all())
            self.assertTrue((loaded.labels[4, 7][0] == -1).all())
            edited = segmented(loaded.prepared_source, environment={
                key: obs.confirmed_environment_mask for key, obs in loaded.segmented_source.observations.items()
            })
            relabeled = label_source(edited, source_id='scene', registry=loaded.registry)
            self.assertTrue((relabeled.labels[4, 7][1:] == 0).all())
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
                self.assertTrue(all(arrays[name].dtype != object for name in arrays.files))

    def test_variable_sizes_empty_objects_and_large_integer_ids(self):
        first, other = prepared(), prepared(((100, 7),), h=2, w=3)
        combined = PreparedSource({**first.observations, **other.observations})
        registry = ObjectRegistry('scene', next_id=2**60)
        snapshot = segmented(combined, {(4, 7): [proposal(9)]})
        result = label_source(snapshot, source_id='scene', registry=registry)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.npz'
            save_labeled_source(path, result)
            loaded = load_labeled_source(path, combined)
            self.assertEqual(loaded.registry.next_id, 2**60 + 1)
            self.assertEqual(loaded.labels[4, 7][0, 0], 2**60)
            self.assertEqual(loaded.labels[100, 7].shape, (2, 3))
            self.assertTrue((loaded.labels[100, 7] == -1).all())

    def test_source_fingerprint_includes_pixels_geometry_validity_and_preparation(self):
        source = prepared()
        original = prepared_source_fingerprint(source)
        obs = source.observations[4, 7]
        for name in ('rgb', 'depth', 'intrinsics', 'w2c', 'geometry_mask'):
            value = getattr(obs, name).clone()
            if name == 'geometry_mask':
                value[0, 0] = False
            else:
                value.reshape(-1)[0] += 0.25
            changed = replace(source, observations={(4, 7): replace(obs, **{name: value})})
            self.assertNotEqual(prepared_source_fingerprint(changed), original)
        self.assertNotEqual(prepared_source_fingerprint(replace(source, preparation=dict(crop=[0, 0, 4, 6]))), original)
        self.assertEqual(prepared_source_fingerprint(replace(source, observations={
            (4, 7): replace(obs, labels=torch.ones(4, 6, dtype=torch.int64)),
        })), original)
        with tempfile.TemporaryDirectory() as directory:
            result = self.result()
            save_labeled_source(Path(directory) / 'labels', result)
            changed = replace(result.prepared_source, preparation={'different': True})
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_labeled_source(Path(directory) / 'labels', changed)

    def test_reject_corrupt_metadata_arrays_and_stale_merged_labels(self):
        result = self.result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels'
            save_labeled_source(path, result)
            meta_path, array_path = path.with_suffix('.json'), path.with_suffix('.npz')
            original = json.loads(meta_path.read_text())
            for update in (dict(schema_version=99), dict(npz_sha256='wrong')):
                meta_path.write_text(json.dumps({**original, **update}))
                with self.assertRaises(ValueError):
                    load_labeled_source(path, result.prepared_source)
            meta_path.write_text(json.dumps(original))
            with np.load(array_path, allow_pickle=False) as arrays:
                data = {name: arrays[name].copy() for name in arrays.files}
            for invalid in (np.full((4, 6), 999, dtype=np.int64), np.zeros((4, 6), dtype=np.float32),
                            np.zeros((2, 3), dtype=np.int64)):
                data['observation_0_labels'] = invalid
                np.savez_compressed(array_path, **data)
                meta_path.write_text(json.dumps({**original, 'npz_sha256': file_sha256(array_path)}))
                with self.assertRaises(ValueError):
                    load_labeled_source(path, result.prepared_source)
            result.registry.allocate('New')
            result.registry.merge_ids(1, 2)
            with self.assertRaisesRegex(ValueError, 'Stale'):
                save_labeled_source(path, result)

    def test_merge_aliases_and_next_id_survive_reload(self):
        result = self.result()
        result.registry.allocate('Merged')
        result.registry.merge_ids(1, 2)
        result = label_source(result.segmented_source, source_id='scene', registry=result.registry)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels'
            save_labeled_source(path, result)
            loaded = load_labeled_source(path, result.prepared_source)
            self.assertEqual(loaded.registry.resolve(1), 2)
            self.assertEqual(loaded.registry.allocate(), 3)

    def test_quality_scores_remain_optional_and_are_not_identity_probabilities(self):
        source = prepared()
        instance = replace(proposal(8), confidence=1.2, stability_score=0.98)
        result = label_source(segmented(source, {(4, 7): [instance]}), source_id='scene')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels'
            save_labeled_source(path, result)
            loaded = load_labeled_source(path, source)
            observation = loaded.segmented_source.observations[4, 7].instances[0]
            self.assertEqual(observation.confidence, 1.2)
            self.assertEqual(observation.stability_score, 0.98)

    def test_invalid_registry_references_are_rejected(self):
        result = self.result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels'
            save_labeled_source(path, result)
            meta_path = path.with_suffix('.json')
            original = meta_path.read_text()
            for mutation in ('missing-id', 'duplicate', 'reuse', 'cycle', 'missing-binding'):
                metadata = json.loads(original)
                registry = metadata['registry']
                if mutation == 'missing-id':
                    registry['observations'][0][1] = 999
                elif mutation == 'duplicate':
                    registry['names'].append(registry['names'][0])
                elif mutation == 'reuse':
                    registry['next_id'] = 1
                elif mutation == 'cycle':
                    registry['aliases'] = [[2, 3], [3, 2]]
                    registry['next_id'] = 4
                else:
                    registry['observations'] = []
                meta_path.write_text(json.dumps(metadata))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_labeled_source(path, result.prepared_source)

    def test_hash_matches_checkpoint_contents(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            checkpoint.write(b'synthetic checkpoint')
            checkpoint.flush()
            self.assertEqual(file_sha256(checkpoint.name), hashlib.sha256(b'synthetic checkpoint').hexdigest())


if __name__ == '__main__':
    unittest.main()
