# ELEC4260 — Competition 2: Multi-Object Grasping

AIRBOT Play arm · RGB-D camera · 10 objects · GR-ConvNet v2 + Jacquard V2

---

## Workspace Structure

```
grconvnet_ws/
│
├── DISCOVERSE/                          MuJoCo sim + AIRBOT Play (course-provided)
│   ├── discoverse/
│   │   ├── airbot_play/                 Fast IK solver (AirbotPlayFIK)
│   │   ├── envs/
│   │   │   └── airbot_play_base.py      AirbotPlayBase, AirbotPlayGrasp, AirbotPlayCfg
│   │   ├── examples/
│   │   │   ├── robots/                  Scripted arm motion demos
│   │   │   ├── tasks_airbot_play/       Pick/place task demos (jujube, block, etc.)
│   │   │   └── test_airbot_play/
│   │   │       └── p3_utils.py          GraspDetector, CircleGraspDetector, IK wrappers
│   │   ├── gaussian_renderer/           3D Gaussian Splatting renderer (optional)
│   │   ├── task_base/                   Task base classes
│   │   └── utils/                       PID controller, state machine
│   ├── models/
│   │   ├── mjcf/
│   │   │   ├── airbot_play_object.xml   competition scene (graspable objects)
│   │   │   ├── airbot_play_circle.xml   circle detector scene
│   │   │   └── airbot_play_floor.xml    bare floor scene
│   │   └── urdf/
│   │       └── airbot_play_v3_gripper_fixed.urdf   for IK solver
│   ├── scripts_p3/
│   │   ├── grasp_pipeline.py            ← main sim pipeline entry point
│   │   └── grasp_evaluation.py          grasp success rate evaluation
│   └── arm_airbot.py                    scripted mink IK motion demo
│
├── robotic-grasping/                    GR-ConvNet v2 (skumra/robotic-grasping)
│   ├── inference/
│   │   ├── models/
│   │   │   ├── grconvnet3.py            ← the model we use (GR-ConvNet v2)
│   │   │   ├── grasp_model.py           base class + ResidualBlock
│   │   │   ├── grconvnet.py             original (unused)
│   │   │   ├── grconvnet2.py            multi-dropout variant (unused)
│   │   │   └── grconvnet4.py            inverted variant (unused)
│   │   ├── grasp_generator.py           inference + grasp candidate extraction
│   │   └── post_process.py              Q-map peak detection, angle decode
│   ├── utils/
│   │   ├── data/
│   │   │   ├── jacquard_data.py         Jacquard dataset loader
│   │   │   ├── cornell_data.py          Cornell dataset loader
│   │   │   └── grasp_data.py            base dataset class
│   │   └── dataset_processing/
│   │       ├── grasp.py                 GraspRectangle, IOU computation
│   │       ├── evaluation.py            IOU eval metrics
│   │       └── image.py                 DepthImage + augmentation
│   ├── hardware/
│   │   ├── camera.py                    RealSense camera interface
│   │   └── device.py                    GPU/CPU device selection
│   ├── trained-models/                  pre-trained checkpoints (Jacquard + Cornell)
│   ├── train_network.py                 ← training entry point
│   ├── evaluate.py                      evaluation entry point
│   ├── run_offline.py                   offline inference on saved images
│   └── run_grasp_generator.py           visualise grasp candidates
│
├── Jacquard_V2/                         HIL-refined dataset toolkit
│   └── Jacquard_V2/
│       ├── train.py                     V2-specific trainer (reference)
│       ├── eval.py                      V2 evaluation script
│       ├── add_fn_data.py               HIL annotation correction tool
│       ├── models/                      GG-CNN2, MobileNetV2, ResNet, etc.
│       └── utils/
│           └── data/jacquard_data.py    V2-compatible data loader
│
├── calibration/                         Camera-to-robot extrinsic calibration
│   ├── calibration.py                   solve PnP → 4×4 cam2robot matrix
│   ├── label_points.py                  GUI: click calibration markers in image
│   ├── record_robot_points.py           record EEF positions via AIRBOT SDK
│   ├── test_realsense.py                verify RealSense streams
│   ├── calibration.json                 ← output: cam2robot 4×4 transform
│   ├── robot_poses.json                 recorded EEF positions (session A)
│   └── robot_poses_alt.json             recorded EEF positions (session B)
│
├── sdk/                                 AIRBOT SDK installers
│   ├── airbot_cpp-5.1.6-x86_64-noble.deb   C++ SDK (installed by pixi run setup)
│   └── airbot_py-5.1.6-py3-none-any.whl    Python SDK (installed by pixi)
│
├── docs/                                Documentation (zensical site)
│   ├── index.md                         overview + scoring
│   ├── setup.md                         environment + dataset setup
│   ├── pipeline.md                      full pipeline walkthrough
│   ├── model.md                         GR-ConvNet architecture + training
│   ├── sim.md                           DISCOVERSE sim guide
│   └── real-robot.md                    real robot deployment + calibration
│
├── data/jacquard/                       Jacquard V2 dataset (not in git, ~59 GB)
├── output/models/                       trained model checkpoints (not in git)
│
├── deploy/                              ← real robot deployment scripts
│   ├── grasp_object.py                  entry point  →  pixi run deploy
│   ├── GraspDetector.py                 RealSense camera grasp detector base
│   ├── GraspDetectorNN.py               GR-ConvNet wrapper for live inference
│   ├── grasp_utils.py                   coordinate transforms, depth inpainting
│   ├── record_grasp.py                  record + visualise grasp data (Open3D)
│   └── test_detector_circle.py          smoke-test circle detector on live camera
│
├── scripts/                             ← loose utility scripts
│   └── test_pose.py                     test arm pose estimation
│
├── pixi.toml                            ← start here: environment + all tasks
├── zensical.toml                        docs site config
└── README.md                            this file
```

---

## Setup

```bash
pixi run setup
```

Installs all dependencies, clones robotic-grasping + Jacquard V2, installs AIRBOT SDK.

---

## Dataset

Download **Jacquard V2** (~59 GB, 12 zips) from the course OneDrive share.  
Unzip all into `data/jacquard/` — structure should look like:

```
data/jacquard/
├── JacquardV2_Dataset_0/
│   └── <object_hash>/
│       ├── *_RGB.png
│       ├── *_perfect_depth.tiff
│       └── *_grasps.txt
├── JacquardV2_Dataset_1/
└── ...
```

---

## Pipeline

### 1 — Train

```bash
pixi run train
```

Trains GR-ConvNet v2 (grconvnet3, depth-only) on Jacquard V2.  
Checkpoints saved to `output/models/`. Tracked with W&B.

### 2 — Simulation

```bash
pixi run sim           # full pipeline, network detector
pixi run sim-eval      # evaluate grasp success rate across 10 objects
```

Runs inside DISCOVERSE (MuJoCo). No real robot needed.

Circle detector (no trained model required — good first smoke test):

```bash
XKB_CONFIG_ROOT=/usr/share/X11/xkb python DISCOVERSE/scripts_p3/grasp_pipeline.py --detector circle
```

### 3 — Real Robot

```bash
pixi run robot-online       # check arm connection
pixi run deploy             # full pipeline on real AIRBOT
pixi run deploy-circle      # circle detector only (no model needed)
```

### Calibration

```bash
pixi run calib-test         # verify RealSense connects
pixi run calib-record       # record robot end-effector positions
pixi run calib-label        # label calibration points in image
pixi run calib-run          # compute camera-to-robot transform
```

---

## Scoring

`Final = sum(s_i)` where `s_i = actual_time` if grasp succeeds within 30s, else `40s penalty`.  
Lower is better. Max penalty = 400s (10 objects × 40s).

**Key win**: grasp ordering by proximity to bin drop position — minimises total arm travel.

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | GR-ConvNet v2 (`grconvnet3`) — residual encoder-decoder |
| Training data | Jacquard V2 — 51,601 human-corrected images |
| Simulation | DISCOVERSE (MuJoCo + AIRBOT Play) |
| Camera | Intel RealSense D435i |
| Real robot | AIRBOT Play arm — Python SDK |
| Environment | Pixi + conda-forge, CUDA 12.8, PyTorch 2.7 |
