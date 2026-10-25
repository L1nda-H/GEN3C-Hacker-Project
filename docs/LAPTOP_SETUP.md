# Working on GEN3C from a laptop

The upstream README reports testing on A100/H100 GPUs, with approximately 43 GB
of peak GPU memory even with full offloading. INSTALL.md requires Linux, Python
3.10, CUDA 12.4, Transformer Engine, and Apex for inference. A CPU development
environment cannot run the complete generation pipeline as documented.

## Local environment

This separate environment uses CPU PyTorch and a subset of the upstream
dependencies for camera geometry, depth processing, small cache experiments,
image/video analysis, plotting, notebooks, and tests. It deliberately omits the
diffusion model stack, CUDA extensions, GUI, and checkpoints. It uses headless
OpenCV, so `cv2.imshow` is unavailable; use notebooks or save images instead.
Python 3.12 is used on this laptop; use upstream Python 3.10 for GPU inference.

From the repository root on Linux x86-64:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python -m pip check
python -m scripts.check_cpu_environment
```

Run repository imports from the root using `python -m ...`. Select
`.venv/bin/python` as your editor's interpreter. Start notebooks with
`python -m jupyterlab`; select the environment's Python kernel. No global kernel
registration is needed. The existing `scripts/test_environment.py` checks the
full GPU stack and is not the appropriate check for this environment.

Use explicit `device="cpu"` and float32 in geometry experiments: many functions
default to CUDA. Start with small images (e.g. 64–256 pixels per side) and a few
frames. Importing Warp does not imply GPU availability; the check exercises
PyTorch geometry without the Warp ray-tracing path. Foreground masking and the
complete cache/rendering pipeline are not validated by the local smoke check.

## Moving to a GPU machine

Create a fresh environment there using INSTALL.md and `cosmos-predict1.yaml`;
do not copy this CPU virtual environment or install upstream requirements over
it. Follow the README's Hugging Face authentication and checkpoint steps on
that machine. Keep ViPE in its own environment as directed by the README.

Sources: [GEN3C paper](https://arxiv.org/abs/2503.03751),
[upstream repository](https://github.com/nv-tlabs/GEN3C), and this checkout's
README.md / INSTALL.md.
