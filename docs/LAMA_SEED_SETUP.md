# CPU LaMa filling for composite seeds

LaMa automatically fills explicitly removed environment objects before the
image-space seed compositor inserts objects. No pip packages, upgrades, CUDA,
LaMa source checkout, or training dependencies are required. This adapter uses
existing PyTorch, NumPy, Pillow and OpenCV; inference is always on CPU, even
when GEN3C later runs on CUDA. Cache rendering remains CPU-compatible.

The model is an exported Big-LaMa TorchScript checkpoint distributed by
[simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting),
using the same image/mask interface as that project's adapter. The underlying
model is [LaMa](https://github.com/advimman/lama). The export is not the original
repository's `best.ckpt`; those formats are not interchangeable here.

## Set up in your existing local environment

Run from this repository's root, using the Python environment you already use
for composite previews. If that environment is `.venv`, activate it first:

```bash
source .venv/bin/activate
mkdir -p checkpoints/lama
curl -fL --retry 2 \
  https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt \
  -o checkpoints/lama/big-lama.pt
export GEN3C_LAMA_MODEL="$PWD/checkpoints/lama/big-lama.pt"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

The download is approximately 197 MiB. Verify this exact exported checkpoint:

```bash
python - <<'PY'
import hashlib
import os
import torch
path = os.environ['GEN3C_LAMA_MODEL']
digest = hashlib.sha256()
with open(path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == '7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c', 'Unexpected or incomplete checkpoint'
model = torch.jit.load(path, map_location='cpu').eval()
print('LaMa ready on CPU; no additional packages installed')
PY
```

`GEN3C_LAMA_MODEL` is optional when using the default location above; it is useful
for a checkpoint stored elsewhere. Nothing is downloaded implicitly. Missing
weights produce a setup error only when a removal actually needs filling.

## Preview an existing removal scene

For the prepared creature example, which removes the cat from the bed:

```bash
python -m scripts.preview_composite_seed \
  --scene-spec outputs/creature_example/creature/inputs/scene_121.json \
  --output-dir outputs/creature_lama_seed \
  --seed-occlusion-policy allow \
  --compare-rendered
```

Replace the scene path with your own prepared removal scene as needed. Removing
an object means its positive environment instance ID is absent from that
source's `selected_ids` (environment ID 0 remains selected). Preparation with
`scripts.evaluate_real_cache_mvp --remove-environment-box ...` produces this
selection automatically. There is no inpainting-method or seed-strategy flag.

The explicit `allow` policy above is needed when an inserted object overlaps
an inpainted background: that background has unknown depth, so its occlusion
cannot be established from the original depth map. With the default `reject`
policy, such overlaps produce an error. `allow` draws over unknown depth and
records the uncertainty. Inspect these regions before running diffusion.

Inspect the preview files:

- `seed.png`: actual image seed, including pasted foreground objects.
- `seed_environment_removal.png`: original requested removal mask at output size.
- `seed_background_synthesized.png`: LaMa contribution before foreground compositing.
- `seed_synthesized.png`: pixels that still contain LaMa RGB after compositing.
- `seed_uncertainty.png`: object overlaps with uncertain environment depth.
- `seed_environment_depth_valid.png`: excludes the filled background.
- `seed_metadata.json`: model hash, cache key, actual mask size and dilation,
  removed IDs, cache reuse, output hash, and occlusion policy.
- `cache_render_reference/`: optional original cache render, for comparison only.

The masks are also in `seed_diagnostics.npz`. An opaque observed foreground
pixel can cover synthesized background; its final `seed_synthesized` value is
false while `seed_background_synthesized` remains true.

Normal `gen3c_composite` inference uses the same fill automatically. Add
`--seed_occlusion_policy allow` to that CLI only if you have chosen that policy.
Model inference is cached, so preview, CLI preflight, and generation can reuse
one repair without repeatedly loading LaMa.

## Reuse, performance, and geometry

Repaired source RGB is cached under `outputs/lama_seed_cache/`. Optionally set
`GEN3C_LAMA_CACHE` to another writable directory. Keys include source RGB,
removal mask, checkpoint SHA256, adapter version, and PyTorch version. Changing
an input creates a new result. Corrupt caches raise an error instead of
silently replacing a reviewed image. The checkpoint is still needed to verify
its identity when reusing a cache.

LaMa runs at the prepared environment's resolution, usually 240 x 320. The
result is then mapped into the output-resolution seed. It pads to multiples of
8 and crops back afterward. Pixels outside the fill mask remain exactly the
same in the source image. The default fill mask expands removal boundaries by
2 source pixels to cover segmentation fringes, protecting other retained
positive-ID objects. Expansion may replace a small amount of adjacent context;
review the synthesized mask.

On the development CPU, a standalone 240 x 320 test took about 13 seconds plus
8 seconds to load the model, with approximately 661 MiB peak process RAM.
Your laptop and subsequent calls may differ. Larger images cost more memory.
No GEN3C weights are loaded by the preview command.

Source RGB/depth, labels, and point clouds are never changed by inpainting.
Removed object samples remain excluded from warp conditioning. Synthesized RGB
is used only for the seed and is never unprojected into a new point cloud.
LaMa can hallucinate plausible background rather than recover the actual
hidden surface; subsequent camera motion can still expose missing geometry.
The starting-camera and field-of-view restrictions remain unchanged, as does
the restriction on moving environment objects. Resolve ambiguous valid labels
before removal; the adapter does not automatically erase unassigned content.

## Pre-edited sources that already discarded removal IDs

Do not infer object removal from missing depth alone: RGB can remain observed
when depth is invalid. If preparation already removed geometry and discarded
its positive IDs, supply an explicit grayscale mask of those removed pixels
at the **prepared environment source resolution**, not the video resolution:

```bash
python -m scripts.preview_composite_seed \
  --scene-spec path/to/scene.json --output-dir outputs/edited_seed \
  --seed-removal-mask path/to/source_removal_mask.png \
  --seed-occlusion-policy allow
```

Nonzero mask pixels request filling. They must already be excluded from retained
cache geometry; otherwise the builder asks you to deselect the object first.
Full generation uses `--seed_removal_mask`, and Python callers use
`seed_removal_mask` on `generate()` (or `removal_mask` on `build_image_space_seed`).
Direct Python masks must match the source grid. The mask changes the cache key.

The cat example's preparation now saves `inputs/seed_removal_mask.png` for the
camera-to-candle edit; its generation helper forwards that mask automatically.
Regenerate preparation if an older cat example lacks the mask. The creature
example retains canonical removal IDs and needs no extra mask.
