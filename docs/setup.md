---
icon: lucide/wrench
---

# Setup

## Prerequisites

- Linux (Ubuntu 22.04+)
- NVIDIA GPU with CUDA 12.8
- [Pixi](https://pixi.sh) installed (`curl -fsSL https://pixi.sh/install.sh | bash`)

---

## First-time setup

```bash
pixi run setup
```

This single command runs three things in order:

| Step | What it does |
|---|---|
| `pixi-install` | Installs all conda + PyPI deps into `.pixi/` |
| `clone-grconvnet` | Clones `skumra/robotic-grasping` if not present |
| `clone-jacquard-v2` | Clones `lqh12345/Jacquard_V2` toolkit if not present |
| `install-airbot` | Installs AIRBOT C++ SDK from `sdk/` (requires `sudo`) |

---

## Environment

The pixi environment lives entirely in `.pixi/` — nothing is installed system-wide (except the AIRBOT SDK).

To activate the environment manually:

```bash
pixi shell
```

Key packages:

| Package | Version | Why |
|---|---|---|
| PyTorch | ≥2.7 (CUDA 12.8) | GR-ConvNet inference + training |
| MuJoCo | ≥3.6 | DISCOVERSE simulation backend |
| OpenCV | ≥4.13 | Image processing |
| pyrealsense2 | latest | RealSense D435i camera |
| airbot-py | 5.1.6 | AIRBOT arm control |
| open3d | latest | 3D point cloud + gripper viz |
| wandb | latest | Training tracking + curves |
| mink | latest | IK solver for sim arm |
| zensical | ≥0.0.41 | This documentation site |

---

## Dataset

Download **Jacquard V2** (~59 GB, 12 zip files) from the course OneDrive share.

Unzip all into `data/jacquard/` so the structure looks like:

```
data/jacquard/
├── JacquardV2_Dataset_0/
│   └── 1a1ec1cfe633adcdebbf11b1629fc16a/
│       ├── 0_..._RGB.png
│       ├── 0_..._perfect_depth.tiff
│       └── 0_..._grasps.txt
├── JacquardV2_Dataset_1/
├── ...
└── JacquardV2_Dataset_11/
```

!!! note "Why Jacquard V2?"
    Jacquard V2 is a human-corrected version of the original Jacquard dataset. After 53,026 iterations of human review, 2,884 badly labelled images were removed and 30,292 missing annotations were added. Result: **+7.1% accuracy** across all tested architectures for free.

---

## pixi.toml tasks reference

| Task | Command |
|---|---|
| `pixi run setup` | Full first-time setup |
| `pixi run train` | Train GR-ConvNet on Jacquard V2 |
| `pixi run sim` | Run grasp pipeline in DISCOVERSE (net detector) |
| `pixi run sim-eval` | Evaluate grasp success rate across 10 objects |
| `pixi run deploy` | Run on real AIRBOT arm (NN detector) |
| `pixi run deploy-circle` | Run on real AIRBOT arm (circle detector, no model) |
| `pixi run robot-online` | Check real arm connection |
| `pixi run calib-test` | Verify RealSense camera connects |
| `pixi run calib-record` | Record robot end-effector positions |
| `pixi run calib-label` | Label calibration points in image |
| `pixi run calib-run` | Compute camera-to-robot transform |
