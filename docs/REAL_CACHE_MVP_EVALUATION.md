# Real-image segmentation and composite cache evaluation

Run real SAM segmentation and render 24 frames from the composite 3D cache on
CPU. No A100 or GEN3C diffusion checkpoint is required. Run all commands from
the repository root. CPU segmentation may take time.

## 1. Activate the environment and install SAM

These commands use the repository's existing CPU PyTorch environment.

```bash
cd /home/linda/CS6672/GEN3C-Hacker-Project
source .venv/bin/activate

python -m pip install wheel iopath==0.1.10 huggingface-hub==0.29.2

SAM2_BUILD_CUDA=0 python -m pip install --no-deps --no-build-isolation \
  "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

Disabling the optional CUDA extension follows
[SAM's installation instructions](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md).

## 2. Download the model and two sample sequences

```bash
mkdir -p checkpoints

curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o checkpoints/sam2.1_hiera_tiny.pt

huggingface-cli download nvidia/GEN3C-Testing-Example \
  --repo-type dataset \
  --include "batch_0000/*" "batch_0001/*" \
  --local-dir datasets/gen3c_examples
```

The [GEN3C samples](https://huggingface.co/datasets/nvidia/GEN3C-Testing-Example/tree/main)
provide RGB, depth, geometry validity, and cameras. Geometry validity is separate
from the object mask produced by SAM.

## 3. Run the script to export input previews

```bash
python -m scripts.evaluate_real_cache_mvp
```

This saves the first frame of each sequence at 320 × 240:

- `outputs/real_cache_mvp/environment_input.png`
- `outputs/real_cache_mvp/object_input.png`

The driver is [scripts/evaluate_real_cache_mvp.py](../scripts/evaluate_real_cache_mvp.py).
It treats the destination image as the fixed environment and uses each source
camera as its local coordinate origin.

## 4. Choose the object

Open `outputs/real_cache_mvp/object_input.png` in an image viewer. Find a box
around one recognizable object. Record `X1 Y1 X2 Y2`: top-left and bottom-right
pixel coordinates in this 320 × 240 preview, with the origin at the top-left.

## 5. Segment, compose, and render

Replace the example coordinates below with your box:

```bash
python -m scripts.evaluate_real_cache_mvp --box 80 40 220 200
```

```bash
python -m scripts.evaluate_real_cache_mvp \
  --box 179 92 232 212 \
  --target-base 235 225 \
  --object-height 160 \
  --trajectory forward \
  --camera-travel 0.08 \
  --output-dir outputs/real_cache_mvp_forward
  ```

The script runs SAM, saves labels and prepared sources, places the object near
the destination camera's center of view using a simple depth-based scale and
translation, and renders a small lateral camera move. Placement is a starting
point for visual review; it does not infer floor contact or physical support.

## 6. Review the output

Inspect `outputs/real_cache_mvp/`:

- `object_labels.png`: segmentation overlay.
- `preview.gif`: 24 rendered camera views.
- `rgb_*.png`, `depth_*.png`, `coverage_*.png`: per-frame inspection.
- `render_*.npz`: exact RGB, depth, and coverage buffers.
- `scene.json`: saved composition and camera trajectory.
- `environment.npz`, `object.npz`, and `*_labels.npz/json`: reusable source artifacts.

For the MVP, check the object's silhouette, recognizability, placement, occlusion,
and stability across the camera move. Coverage images show holes in the point
cache. Depth PNGs normalize independently per frame; use NPZ buffers for numeric
depth comparisons.

The existing renderer has been exercised on CPU. Real-checkpoint segmentation
and this complete real-image run still require validation with the downloaded inputs.
