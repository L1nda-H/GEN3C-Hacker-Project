"""Run from the repository root: python -m scripts.check_cpu_environment."""

import importlib
import sys


def main():
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "pandas",
        "skimage",
        "imageio",
        "av",
        "imageio_ffmpeg",
        "matplotlib",
        "hydra",
        "einops",
        "warp",
        "huggingface_hub",
        "ipykernel",
        "jupyterlab",
        "pytest",
    ):
        importlib.import_module(name)
        print(f"OK: {name}")

    import torch
    from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
    from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
        unproject_points,
    )

    pose = torch.eye(4)
    intrinsics = torch.eye(3)
    depth = torch.ones(1, 1, 4, 4)
    points = unproject_points(depth, pose[None], intrinsics[None])
    expected = torch.tensor([2.0, 1.0, 1.0])
    torch.testing.assert_close(points[0, 1, 2], expected)
    poses, cameras = generate_camera_trajectory(
        "left",
        pose,
        intrinsics,
        num_frames=5,
        movement_distance=0.1,
        camera_rotation="no_rotation",
        device="cpu",
    )
    assert poses.shape == (1, 5, 4, 4)
    assert cameras.shape == (1, 5, 3, 3)
    assert torch.isfinite(poses).all()
    assert points.device.type == poses.device.type == "cpu"
    torch.testing.assert_close(poses[0, 0], pose)
    print(f"OK: CPU geometry; PyTorch {torch.__version__}")
    print("CPU development environment ready. Full GEN3C inference is not checked.")


if __name__ == "__main__":
    main()
