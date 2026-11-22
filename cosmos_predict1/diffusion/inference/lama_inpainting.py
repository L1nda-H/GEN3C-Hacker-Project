"""Optional CPU-only LaMa TorchScript inference with a content-addressed RGB cache.

Uses existing torch/numpy only. No original LaMa training code or dependencies,
implicit downloads, CUDA, or synthesized depth. Load only trusted checkpoints.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _predict(rgb, mask, checkpoint):
    """Exported big-lama.pt signature: float image [0,1], binary mask, -> RGB."""
    h, w = mask.shape
    image = np.ascontiguousarray(((rgb + 1) / 2).transpose(2, 0, 1))
    padding = ((0, 0), (0, -h % 8), (0, -w % 8))
    image = torch.from_numpy(np.pad(image, padding, mode='symmetric'))[None]
    mask_tensor = torch.from_numpy(np.pad(mask.astype(np.float32)[None], padding, mode='symmetric'))[None]
    model = torch.jit.load(str(checkpoint), map_location='cpu').eval()
    with torch.inference_mode():
        predicted = model(image, mask_tensor)
    if not isinstance(predicted, torch.Tensor) or predicted.shape != image.shape:
        raise ValueError('LaMa checkpoint must return RGB with the same padded dimensions as its input')
    generated = predicted[0, :, :h, :w].permute(1, 2, 0).cpu().float().numpy()
    if not np.isfinite(generated).all():
        raise ValueError('LaMa returned nonfinite RGB')
    return np.clip(generated, 0, 1) * 2 - 1


def inpaint_background(rgb, mask, *, model_path=None, cache_dir=None):
    """Fill only mask=True pixels in float32 RGB [-1,1]; preserve all others exactly.

    GEN3C_LAMA_MODEL defaults to checkpoints/lama/big-lama.pt, and
    GEN3C_LAMA_CACHE defaults to outputs/lama_seed_cache, relative to repo root.
    Empty masks neither load a checkpoint nor write a cache. Corrupt caches are
    errors rather than silently changing a previously reviewed background.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if (rgb.shape != (*mask.shape, 3) or mask.ndim != 2 or not np.isfinite(rgb).all()
            or (np.abs(rgb) > 1).any()):
        raise ValueError('LaMa needs aligned finite RGB [-1,1] and a 2D removal mask')
    if not mask.any():
        return rgb.copy(), dict(method='none', synthesized_pixels=0)
    if mask.all():
        raise ValueError('Cannot inpaint the whole environment without surrounding context')
    checkpoint = Path(model_path or os.environ.get('GEN3C_LAMA_MODEL', _ROOT / 'checkpoints/lama/big-lama.pt')).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'LaMa checkpoint missing: {checkpoint}. Download big-lama.pt as described in '
                                'docs/LAMA_SEED_SETUP.md or set GEN3C_LAMA_MODEL. No packages need installing.')
    checkpoint_hash = _sha256(checkpoint)
    parameters = dict(version=1, model_sha256=checkpoint_hash, rgb_sha256=_digest(rgb),
                      mask_sha256=_digest(mask), shape=list(rgb.shape), torch_version=str(torch.__version__))
    key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir or os.environ.get('GEN3C_LAMA_CACHE', _ROOT / 'outputs/lama_seed_cache')).expanduser().resolve()
    path = root / f'{key}.npz'
    cache_hit = path.is_file()
    if cache_hit:
        with np.load(path, allow_pickle=False) as arrays:
            result = arrays['rgb'].copy()
            metadata = json.loads(str(arrays['metadata'].item()))
        if (metadata.get('parameters') != parameters or result.shape != rgb.shape
                or result.dtype != np.float32 or not np.isfinite(result).all()
                or (np.abs(result) > 1).any() or metadata.get('result_sha256') != _digest(result)
                or not np.array_equal(result[~mask], rgb[~mask])):
            raise ValueError(f'Invalid LaMa cache: {path}; move this file aside and rerun to regenerate it')
    else:
        generated = _predict(rgb, mask, checkpoint)
        result = rgb.copy()
        result[mask] = generated[mask]
        metadata = dict(parameters=parameters, result_sha256=_digest(result))
        root.mkdir(parents=True, exist_ok=True)
        # Concurrent previews/ranks publish only complete cache files.
        with tempfile.NamedTemporaryFile(dir=root, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(handle, rgb=result, metadata=json.dumps(metadata))
                handle.close()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return result, dict(method='lama_torchscript', device='cpu', model_path=str(checkpoint),
                        model_sha256=checkpoint_hash, cache_key=key, cache_path=str(path),
                        cache_hit=cache_hit, synthesized_pixels=int(mask.sum()),
                        result_sha256=metadata['result_sha256'])
