---
icon: lucide/home
---

# ELEC4260 — Multi-Object Grasping

**Competition 2** · AIRBOT Play arm · RGB-D camera · 10 objects · GR-ConvNet v2

---

## What this workspace does

A full pick-and-place pipeline that:

1. Looks at the scene with an RGB-D camera
2. Runs GR-ConvNet v2 to predict the best grasp pose for each object (outputs Q, W, Φ maps)
3. Orders the grasps to minimise total arm travel distance
4. Controls the AIRBOT Play arm to pick and drop each object into a bin

No ROS2 — pure Python SDK control from camera to arm.

---

## Scoring

`Final score = Σ sᵢ` where `sᵢ = actual_time` if success within 30s, else `40s` penalty.

Lower is better. Max possible penalty = **400s** (10 × 40s).

!!! warning "One failure = 40s penalty"
    Being slow on 3 objects (~90s total) is still better than one failed grasp.
    Reliability beats speed.

---

## Workspace at a glance

| Folder / File | What it is |
|---|---|
| `DISCOVERSE/` | MuJoCo sim environment with AIRBOT Play model |
| `robotic-grasping/` | GR-ConvNet v2 model, training, evaluation code |
| `Jacquard_V2/` | Refined dataset toolkit (+7.1% accuracy over V1) |
| `calibration/` | Camera-to-robot extrinsic calibration tools |
| `sdk/` | AIRBOT C++ + Python SDK installers |
| `data/jacquard/` | Dataset lives here — ~59 GB, not in git |
| `output/models/` | Trained model checkpoints — not in git |
| `grasp_object.py` | Real robot entry point |
| `grasp_utils.py` | Shared math: coordinate transforms, depth inpainting |
| `GraspDetector.py` | RealSense camera grasp detector base class |
| `GraspDetectorNN.py` | GR-ConvNet wrapper for live inference |
| `pixi.toml` | Environment + task runner — start here |

---

## Quick start

```bash
pixi run setup          # install everything, clone repos, install SDK
pixi run sim            # run full pipeline in simulation
pixi run train          # train GR-ConvNet on Jacquard V2
pixi run deploy         # run on real AIRBOT arm
```

---

## Key win

Grasp **ordering strategy** + **Jacquard V2 training data**.  
GR-ConvNet runs in ~20ms so inference is not the bottleneck — arm travel distance and grasp selection order are.
