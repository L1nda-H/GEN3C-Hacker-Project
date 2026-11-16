# Run composite Cache3D inference on GCP

Run these blocks **in order, in the same Bash terminal on your GPU VM**.
This example downloads all available RGB-D batches, imports the candle into the
environment, and generates a 121-frame video. No training dataset is needed.

The repo reports roughly 43 GB GPU memory with full offloading for ordinary
GEN3C; composite inference may need more and still needs real-GPU validation.

## 0. First-time setup: install Conda and GEN3C dependencies

Skip this section if your `cosmos-predict1` environment is already installed.
If `conda: command not found`, install Conda with
[Miniforge](https://github.com/conda-forge/miniforge#unix-like-platforms-macos-linux--wsl):

```bash
curl -fL \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
  -o /tmp/Miniforge3.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
```

If Miniforge is already installed at `~/miniforge3`, just run the `source`
command. Run that command again when reconnecting if Conda is unavailable.

Create the environment and install inference dependencies, following
[INSTALL.md](../INSTALL.md#inference). GPU drivers alone are insufficient.
Replace the repository path if necessary. These installs can take time.

Install the system OpenGL library required by `opencv-python` on a minimal
Ubuntu/Debian VM. This also fixes `ImportError: libGL.so.1`:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

```bash
cd ~/GEN3C-Hacker-Project
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
export CUDA_HOME="$CONDA_PREFIX"
# Install PyTorch before building SAM; SAM's optional CUDA extension is unused here.
python -m pip install torch==2.6.0 torchvision==0.21.0
SAM2_BUILD_CUDA=0 python -m pip install -r requirements.txt
ln -sf "$CONDA_PREFIX"/lib/python3.10/site-packages/nvidia/*/include/* "$CONDA_PREFIX/include/"
ln -sf "$CONDA_PREFIX"/lib/python3.10/site-packages/nvidia/*/include/* "$CONDA_PREFIX/include/python3.10/"
python -m pip install 'transformer-engine[pytorch]==1.12.0'
if [ ! -d apex ]; then git clone https://github.com/NVIDIA/apex; fi
# Pin Apex: current master uses annotations incompatible with PyTorch 2.6.
git -C apex fetch origin e74a67bba3ee679f778670e17edc21639008ae0a
git -C apex checkout --detach e74a67bba3ee679f778670e17edc21639008ae0a
python -m pip install -v --disable-pip-version-check --no-cache-dir \
  --no-build-isolation --no-deps --force-reinstall \
  --config-settings '--build-option=--cpp_ext' \
  --config-settings '--build-option=--cuda_ext' ./apex
```

MoGe is unnecessary for this composite workflow: the samples already contain
depth and cameras. Skip the MoGe install in the general installation guide.
Current [MoGe dependencies](https://github.com/microsoft/MoGe/blob/main/pyproject.toml)
include NumPy 2 and Gradio, which conflict with this repo's older pins.

If you already installed MoGe and it upgraded Hugging Face Hub or NumPy, repair
the active `cosmos-predict1` environment before continuing:

```bash
conda activate cosmos-predict1
python -m pip uninstall -y moge gradio gradio-client hf-gradio opencv-python-headless
# Both OpenCV packages share cv2; reinstall the repo's package after removal.
python -m pip install --force-reinstall --no-deps \
  huggingface-hub==0.29.2 numpy==1.26.4 opencv-python==4.10.0.84
python -m pip check
python -c 'import cv2, numpy, transformers, huggingface_hub; print(cv2.__version__, numpy.__version__, transformers.__version__, huggingface_hub.__version__)'
```

Resolve any remaining `pip check` conflicts before running inference. This
repair restores the affected repository pins without reinstalling PyTorch or Apex.

## 1. Activate GEN3C and check CUDA

Replace the first path with your cloned repository's location.

```bash
cd ~/GEN3C-Hacker-Project
conda activate cosmos-predict1
export CUDA_HOME="$CONDA_PREFIX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python scripts/test_environment.py
```

Resolve any environment-check errors before continuing. Your checkout must
contain `cosmos_predict1/diffusion/inference/gen3c_composite.py` and
`scripts/evaluate_real_cache_mvp.py`.

If importing `megatron.core` fails inside Transformer Engine with
`ldconfig -p | grep 'libnvrtc'`, NVRTC was not found. Transformer Engine
searches `CUDA_HOME` recursively before checking the system library cache:

```bash
conda activate cosmos-predict1
export CUDA_HOME="$CONDA_PREFIX"
python -c 'import megatron.core; import transformer_engine.pytorch; print("Megatron and Transformer Engine OK")'
```

Keep `CUDA_HOME` exported in the inference terminal. If the same NVRTC lookup
still fails, check whether the environment contains the runtime compiler:

```bash
find "$CONDA_PREFIX" -name 'libnvrtc.so*'
# Only if no library is found; matches cosmos-predict1.yaml's CUDA version:
conda install -y -c conda-forge cuda-nvrtc=12.4
```

Then repeat the import check. NVRTC is CUDA's runtime compiler library;
this traceback does not mean Megatron itself is missing.

If that check instead fails inside Apex with
`normalized_shape has unsupported type list[int]`, rebuild Apex using the
pinned fetch/checkout and pip commands in step 0 (after exporting `CUDA_HOME`).
The pinned source uses `typing.List[int]` and `typing.Tuple`, accepted by the
repo's PyTorch 2.6 schema parser. The reinstall retains the PyTorch version;
it rebuilds Apex's C++/CUDA extensions. Repeat the import check afterward.

## 2. Download inference checkpoints

Log in with a Hugging Face read token; accept any requested model access terms
on Hugging Face if a download returns 401/403. Skip the download if these
checkpoints are already present. This downloads large model files, including T5.

```bash
huggingface-cli login
python scripts/download_gen3c_checkpoints.py --checkpoint_dir checkpoints
```

## 3. Download all sample batches and SAM weights

The [NVIDIA sample dataset](https://huggingface.co/datasets/nvidia/GEN3C-Testing-Example/tree/main)
includes RGB, depth, validity masks, and cameras. It lists 10 batches,
`batch_0000` through `batch_0009`; the wildcard downloads all of them.
The script expects this exact local directory layout. Already downloaded files
are reused.

```bash
huggingface-cli download nvidia/GEN3C-Testing-Example \
  --repo-type dataset \
  --include 'batch_*/*' \
  --local-dir datasets/gen3c_examples

mkdir -p checkpoints
curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o checkpoints/sam2.1_hiera_tiny.pt

# Only if SAM was not installed with requirements.txt:
python -c 'import sam2' || SAM2_BUILD_CUDA=0 python -m pip install \
  --no-deps --no-build-isolation \
  'SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
```

## 4. Segment the candle and save a composite scene

This preparation script runs on CPU even on a GPU VM. Coordinates refer to
its 320 × 240 previews. Use a fresh output directory for each experiment.

```bash
python -m scripts.evaluate_real_cache_mvp \
  --environment-batch batch_0004 --object-batch batch_0008 \
  --box 179 92 232 212 \
  --target-base 235 225 --object-height 160 \
  --output-dir outputs/gcp_composite_inputs
```

Inspect `outputs/gcp_composite_inputs/object_labels.png`, `seed.png`, and
`preview.gif` through your VM's file browser or by downloading them. Adjust
the box or placement and rerun if necessary. This saves all prepared sources
and labels locally, so no files from your laptop are required.

## 5. Convert the preview trajectory for diffusion

The preview has 24 cameras at 240 × 320; diffusion requires at least 121.
This interpolates the script's translation-only path to 121 cameras, sets the
target to 704 × 1280, and scales target intrinsics. Source geometry and labels
stay at their original resolution; the target uses a wider aspect ratio.

```bash
python - <<'PY'
from pathlib import Path
import torch
from cosmos_predict1.diffusion.inference.composite_scene import SceneSpec

root = Path("outputs/gcp_composite_inputs")
scene = SceneSpec.load(root / "scene.json")
poses, intrinsics = scene.cameras()
assert torch.allclose(poses[0, :, :3, :3], poses[0, :1, :3, :3].expand_as(poses[0, :, :3, :3]))
alpha = torch.linspace(0, 1, 121)[:, None]
target = poses[:, :1].repeat(1, 121, 1, 1)
target[0, :, :3, 3] = (1 - alpha) * poses[0, 0, :3, 3] + alpha * poses[0, -1, :3, 3]
h, w = scene.image_size
k = intrinsics[:, :1].repeat(1, 121, 1, 1)
k[..., 0, :] *= 1280 / w
k[..., 1, :] *= 704 / h
k[..., 0, 2] += (1280 / w - 1) / 2
k[..., 1, 2] += (704 / h - 1) / 2
scene.target_w2cs = target
scene.target_intrinsics = k
scene.image_size = (704, 1280)
scene.save(root / "scene_121.json")
print(root / "scene_121.json")
PY
```

## 6. Run single-GPU diffusion

Offloading reduces GPU memory use. This run disables guardrails, keeps the
text encoder enabled for your prompt, and uses no prompt upsampler.

```bash
cd ~/GEN3C-Hacker-Project
conda activate cosmos-predict1
export CUDA_HOME="$CONDA_PREFIX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; import megatron.core; import transformer_engine.pytorch; print("Imports OK")'
set -o pipefail
mkdir -p outputs/gcp_composite
python -m cosmos_predict1.diffusion.inference.gen3c_composite \
  --scene_spec outputs/gcp_composite_inputs/scene_121.json \
  --checkpoint_dir checkpoints \
  --prompt 'A candle standing on a rock in a fantasy mushroom landscape.' \
  --num_video_frames 121 --height 704 --width 1280 \
  --guidance 1 --seed 1 \
  --offload_diffusion_transformer --offload_tokenizer \
  --offload_text_encoder_model --offload_guardrail_models \
  --disable_guardrail \
  --video_save_folder outputs/gcp_composite --video_save_name candle \
  2>&1 | tee outputs/gcp_composite/inference.log
```

Result: `outputs/gcp_composite/candle.mp4`. Inspect `candle_review/seed.png`,
buffer previews, and `generation.json` alongside the video. If seed coverage
fails, inspect those artifacts and fix the scene. If CUDA runs out of memory
with these offload flags, use a GPU with more VRAM; successful `nvidia-smi`
alone does not establish sufficient memory.

To compare a seed stitched from source RGB against the adaptive rendered seed,
first preview it at output resolution (using the pixel-center intrinsics
conversion in step 5):

```bash
OMP_NUM_THREADS=4 python -m scripts.preview_composite_seed \
  --scene-spec outputs/gcp_composite_inputs/scene_121.json \
  --output-dir outputs/candle_seed_preview --compare-rendered
```

Inspect `seed.png` and the comparison in `rendered_seed/`. If uncertain
occlusion is reported, inspect the depth or rerun with
`--seed-occlusion-policy allow` to save a preview with flagged regions in
`seed_uncertainty.png`. This explicitly accepts uncertain depth decisions.
Then add `--seed_strategy image_space` to the inference command (and
`--seed_occlusion_policy allow` only if that policy was inspected and chosen).
Use a new output name/directory for the comparison. The warp buffers stay the
same, so this isolates the seed change. Nonplanar objects may need placement
adjustment when represented as textured planes.

More options: [composite usage](COMPOSITE_GEN3C_USAGE.md) and
[segmentation/placement examples](REAL_CACHE_MVP_EVALUATION.md).
