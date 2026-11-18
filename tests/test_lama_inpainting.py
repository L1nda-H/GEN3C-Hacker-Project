"""CPU TorchScript adapter, exact unmasked preservation, and cache invalidation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.lama_inpainting import inpaint_background


class ConstantFill(torch.nn.Module):
    def forward(self, image, mask):
        return image * 0 + mask * 0 + .25


def checkpoint(path):
    torch.jit.trace(ConstantFill(), (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8))).save(str(path))


class LamaInpaintingTests(unittest.TestCase):
    def test_cpu_padding_cache_and_exact_unmasked_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.pt'
            checkpoint(model)
            rgb = np.random.default_rng(0).uniform(-1, 1, (7, 11, 3)).astype(np.float32)
            mask = np.zeros((7, 11), bool); mask[2:5, 3:8] = True
            result, metadata = inpaint_background(rgb, mask, model_path=model, cache_dir=root / 'cache')
            np.testing.assert_array_equal(result[~mask], rgb[~mask])
            np.testing.assert_array_equal(result[mask], np.full((15, 3), -.5, dtype=np.float32))
            self.assertEqual(metadata['device'], 'cpu')
            self.assertFalse(metadata['cache_hit'])
            with patch('torch.jit.load', side_effect=AssertionError('Cached RGB should avoid loading model')):
                again, cached = inpaint_background(rgb, mask, model_path=model, cache_dir=root / 'cache')
            self.assertTrue(cached['cache_hit'])
            np.testing.assert_array_equal(result, again)
            changed = mask.copy(); changed[0, 0] = True
            _, other = inpaint_background(rgb, changed, model_path=model, cache_dir=root / 'cache')
            self.assertNotEqual(metadata['cache_key'], other['cache_key'])
            changed_rgb = rgb.copy(); changed_rgb[0, 0] *= .5
            _, other = inpaint_background(changed_rgb, mask, model_path=model, cache_dir=root / 'cache')
            self.assertNotEqual(metadata['cache_key'], other['cache_key'])
            # Cache corruption must not silently replace a reviewed result.
            path = Path(metadata['cache_path'])
            with np.load(path) as arrays:
                stored = json.loads(str(arrays['metadata'].item()))
            np.savez_compressed(path, rgb=np.zeros_like(rgb), metadata=json.dumps(stored))
            with self.assertRaisesRegex(ValueError, 'Invalid LaMa cache'):
                inpaint_background(rgb, mask, model_path=model, cache_dir=root / 'cache')

    def test_no_removal_needs_no_checkpoint_and_missing_weights_are_actionable(self):
        rgb = np.zeros((8, 8, 3), np.float32)
        mask = np.zeros((8, 8), bool)
        with patch('torch.jit.load', side_effect=AssertionError('No model needed')):
            result, metadata = inpaint_background(rgb, mask, model_path='/nonexistent/lama.pt')
        np.testing.assert_array_equal(rgb, result)
        self.assertEqual(metadata['method'], 'none')
        mask[3, 3] = True
        with self.assertRaisesRegex(FileNotFoundError, 'LAMA_SEED_SETUP'):
            inpaint_background(rgb, mask, model_path='/nonexistent/lama.pt')
        with self.assertRaisesRegex(ValueError, 'whole environment'):
            inpaint_background(rgb, np.ones((8, 8), bool))

    def test_checkpoint_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); model = root / 'model.pt'; checkpoint(model)
            rgb = np.zeros((8, 8, 3), np.float32)
            mask = np.eye(8, dtype=bool)
            _, first = inpaint_background(rgb, mask, model_path=model, cache_dir=root / 'cache')
            model.write_bytes(model.read_bytes() + b'version-change')
            with patch('cosmos_predict1.diffusion.inference.lama_inpainting._predict', return_value=rgb):
                _, second = inpaint_background(rgb, mask, model_path=model, cache_dir=root / 'cache')
            self.assertNotEqual(first['cache_key'], second['cache_key'])


if __name__ == '__main__':
    unittest.main()
