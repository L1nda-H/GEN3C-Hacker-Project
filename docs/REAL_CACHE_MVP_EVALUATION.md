# Real-image segmentation and composite cache evaluation

Run real SAM segmentation and render 24 frames from each selected source cache on
CPU. Preview the separate RGB/mask conditioning buffers, plus a joint-rendered
reference and initial seed. No A100 or GEN3C diffusion checkpoint is required. Run all commands from
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
  --include "batch_0004/*" "batch_0008/*" \
  --local-dir datasets/gen3c_examples
```

The [GEN3C samples](https://huggingface.co/datasets/nvidia/GEN3C-Testing-Example/tree/main)
provide RGB, depth, geometry validity, and cameras. Geometry validity is separate
from the object mask produced by SAM. The script uses `batch_0004` for the
environment and `batch_0008` for the object source.

## 3. Run the script to export input previews

```bash
python -m scripts.evaluate_real_cache_mvp
```

This saves the first frame of each sequence at 320 × 240:

- `outputs/real_cache_mvp/environment_input.png`
- `outputs/real_cache_mvp/object_input.png`

The driver is [scripts/evaluate_real_cache_mvp.py](../scripts/evaluate_real_cache_mvp.py).
It treats the destination image as the fixed environment and uses each source
camera as its local coordinate origin. You can exclude segmented objects from
that environment, import a new object, or combine both operations.

## 4. Choose objects to import or remove

Open `outputs/real_cache_mvp/object_input.png` in an image viewer. Find a box
around one recognizable object. Record `X1 Y1 X2 Y2`: top-left and bottom-right
pixel coordinates in this 320 × 240 preview, with the origin at the top-left.
For removal, choose boxes in `environment_input.png` instead. Box coordinates
must be ordered and within X = 0–319, Y = 0–239. Boxes prompt SAM to find object
silhouettes; they do not erase whole rectangles.

## 5. Segment, compose, and render

Import an object using `--box` (the existing behavior). Replace the example
coordinates below with your box:

```bash
python -m scripts.evaluate_real_cache_mvp --box 80 40 220 200
```

Remove an environment object **without importing anything**:

```bash
python -m scripts.evaluate_real_cache_mvp \
  --remove-environment-box 120 10 305 205 \
  --output-dir outputs/real_cache_mvp_removal
```

Combine removal with import, repeating the removal option for multiple objects:

```bash
python -m scripts.evaluate_real_cache_mvp \
  --remove-environment-box 80 40 160 200 \
  --remove-environment-box 200 60 280 180 \
  --box 179 92 232 212 \
  --target-base 235 225 \
  --object-height 160 \
  --output-dir outputs/real_cache_mvp_replacement
```

Removal segments the specified environment objects and leaves their positive
IDs unselected. Environment ID 0 contains the remaining confirmed pixels.
Removed regions become holes in RGB/depth/coverage; the script does not run
diffusion or reconstruct the hidden background. Inspect `environment_labels.png`
to verify SAM's mask. Empty removal predictions raise an error.

Removal-only runs load only `batch_0004`; they do not require the imported-object
dataset or a placement anchor. Camera travel uses the median valid input
environment depth as its scale. With an import, it continues to use placement
depth; `--target-base` and `--object-height` apply only to imports.

For a larger camera move with an imported object:

```bash
python -m scripts.evaluate_real_cache_mvp \
  --box 179 92 232 212 \
  --target-base 235 225 \
  --object-height 160 \
  --trajectory forward \
  --camera-travel 0.08 \
  --output-dir outputs/real_cache_mvp_forward
```

The script runs SAM, saves labels and prepared sources, places any imported object near
the destination camera's center of view using a simple depth-based scale and
translation, and renders a small lateral camera move by default (or a forward
move with `--trajectory forward`). Placement is a starting
point for visual review; it does not infer floor contact or physical support.

## 6. Review the output

Inspect the selected `--output-dir` (default `outputs/real_cache_mvp/`). Use a
fresh directory for each mode to avoid leftover artifacts from earlier runs:

- `environment_labels.png`: environment and removed-object segmentation overlay.
- `object_labels.png`: imported-object segmentation overlay, when importing.
- `preview.png`: labeled first-frame contact sheet; `preview.gif`: the same
  panels over the camera trajectory. Panels show each source buffer followed by
  the joint geometric reference.
- `buffer_000.gif`, and `buffer_001.gif` when importing: separate source
  animations. Sorted source IDs give buffer 0 = environment and buffer 1 = object.
- `rgb_<frame>_buffer_<index>.png`, `depth_<frame>_buffer_<index>.png`,
  `coverage_<frame>_buffer_<index>.png`: each source's RGB, depth, and validity.
- `render_<frame>.npz`: exact RGB `[1,1,N,3,H,W]`, depth `[1,1,N,H,W]`, and
  coverage `[1,1,N,1,H,W]` buffers. Here `N=1` for removal only, or `N=2` with
  an import; `H=240`, `W=320`.
- `joint_reference_<frame>.png`: joint depth-rendered reference, not a fused
  conditioning buffer or a diffusion prediction.
- `seed.npy`, `seed.png`: joint-rendered initial image-conditioning seed,
  with exact normalized array shape `[1,3,1,H,W]`.
- `preview.json`: ordered source IDs, conditioning strategy, frame count,
  and image dimensions.
- `scene.json`: saved composition and camera trajectory.
- `environment.npz`, optional `object.npz`, and `*_labels.npz/json`: reusable source artifacts.

For the MVP, check the object's silhouette, recognizability, placement, and
stability in its own buffer. Inspect cross-source occlusion in the joint
reference. The environment intentionally remains visible behind objects in its
own buffer; no cross-source depth test is applied to conditioning buffers.
During actual generation, each RGB/mask buffer is encoded independently and
the latents are concatenated for diffusion. This CPU preview does not execute
that model or predict its resolution of occlusion.

Coverage images show each source's holes, including gaps that a background
buffer could otherwise hide. Separate rendering does not itself fill these
gaps. Depth PNGs normalize independently per frame and buffer; use NPZ buffers
for numeric depth comparisons. RGB uses [-1,1], holes use RGB -1/depth 0, and
coverage is binary.

To regenerate previews from saved artifacts without loading SAM or segmenting
again, use a fresh output directory to avoid mixing old preview filenames:

```bash
python -m scripts.evaluate_real_cache_mvp \
  --scene-spec outputs/real_cache_mvp_forward/scene.json \
  --output-dir outputs/real_cache_mvp_forward_buffers
```

This renders every camera in the saved trajectory and uses its saved dimensions
and placement. Original inputs and label overlays remain in the source run's
directory. `--scene-spec` cannot be combined with import or removal boxes;
run segmentation again to change the selection masks. Normal GEN3C generation
also requires enough checkpoint buffer slots
for every selected source; this renderer-only preview does not enforce that
checkpoint limit.

CPU tests cover removal only, import only, combined edits, multiple removals,
and invalid or empty removal predictions. A real SAM 2.1 tiny removal-only run
with the sample environment and box `120 10 305 205` also produced the expected
hole across the 24-frame preview. Inspect segmentation quality for each new
selection; diffusion filling remains a separate validation step.
