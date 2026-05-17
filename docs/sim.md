---
icon: lucide/monitor
---

# Simulation — DISCOVERSE

## What it is

DISCOVERSE is a MuJoCo-based simulation environment with the AIRBOT Play arm model built in. It provides:

- Simulated overhead RGB-D camera
- AIRBOT Play arm with full kinematics
- Pre-built scenes with graspable objects
- Python API that mirrors the real robot interface

**Location**: `DISCOVERSE/` (course-provided — do not update from GitHub, the upstream repo has been restructured and will break the pipeline)

---

## DISCOVERSE folder structure

```
DISCOVERSE/
├── discoverse/
│   ├── airbot_play/
│   │   └── airbot_play_fik.py    fast IK solver (AirbotPlayFIK)
│   ├── envs/
│   │   ├── airbot_play_base.py   AirbotPlayBase, AirbotPlayGrasp, AirbotPlayCfg
│   │   └── simulator.py          MuJoCo simulator base class
│   ├── examples/
│   │   ├── tasks_airbot_play/    scripted task demos (pick, place, drawer etc.)
│   │   ├── robots/               arm motion examples (scripted, not teleop)
│   │   └── test_airbot_play/
│   │       └── p3_utils.py       GraspDetector, CircleGraspDetector, IK wrappers
│   ├── motion_planning/          trajectory planning utilities
│   ├── gaussian_renderer/        3DGS renderer (optional, GPU-heavy)
│   └── utils/
│       └── controllor.py         PID controller utility
├── models/
│   ├── mjcf/
│   │   ├── airbot_play_object.xml   scene with graspable objects (competition)
│   │   ├── airbot_play_circle.xml   scene with circular objects (circle detector)
│   │   ├── airbot_play_floor.xml    bare floor scene
│   │   └── airbot_play_simple.xml   minimal scene for IK testing
│   └── urdf/
│       └── airbot_play_v3_gripper_fixed.urdf   URDF for IK solver
├── scripts_p3/
│   ├── grasp_pipeline.py     main pipeline entry point
│   └── grasp_evaluation.py   evaluate grasp success rate across trials
├── mink/                     IK library (git submodule)
└── arm_airbot.py             scripted arm motion demo using mink IK
```

---

## Running the sim

### First smoke test (no model needed)

Uses the circle detector — just OpenCV circle detection, no trained model required. Good to verify the environment works end to end.

```bash
XKB_CONFIG_ROOT=/usr/share/X11/xkb python DISCOVERSE/scripts_p3/grasp_pipeline.py --detector circle
```

### Full pipeline (trained model required)

```bash
pixi run sim
```

### Evaluation across 10 objects

```bash
pixi run sim-eval
# or with custom options:
python DISCOVERSE/scripts_p3/grasp_evaluation.py --trials 10 --device cuda
```

---

## Key classes

### `AirbotPlayCfg`

Config dataclass. Key fields:

```python
cfg = AirbotPlayCfg()
cfg.mjcf_file_path = "mjcf/airbot_play_object.xml"  # which scene to load
cfg.obs_rgb_cam_id = [0]    # which cameras give RGB
cfg.obs_depth_cam_id = [0]  # which cameras give depth
cfg.render_set = {"fps": 30, "width": 640, "height": 480}
```

### `AirbotPlayBase` / `AirbotPlayGrasp`

The MuJoCo environment class. Step interface:

```python
obs, pri_obs, reward, terminal, info = env.step(action)
# action: 7-element array [joint1..joint6, gripper]
# pri_obs["img"][0]  → RGB image (H×W×3)
# pri_obs["dep"][0]  → depth image (H×W)
```

### `AirbotPlayFIK`

Fast inverse kinematics. Takes a target Cartesian pose and returns joint angles.

```python
arm_fik = AirbotPlayFIK(urdf_path)
joint_angles = arm_fik.properIK(translation, rotation_matrix, current_joints)
```

### `p3_utils.py` — key functions

| Function | What it does |
|---|---|
| `GraspDetector(device)` | Loads GR-ConvNet, runs inference on depth image |
| `CircleGraspDetector()` | Detects circles with OpenCV, returns grasp |
| `transfer_grasps(grasps, cam_rot, cam_pos)` | Camera → world frame transform |
| `get_end_effector_trans(grasp3d)` | Converts grasp to EEF target pose |
| `inverse_kinematics(env, fik, pos, mat)` | Wraps AirbotPlayFIK |
| `smooth_motion(env, target, max_step, threshold)` | Interpolated arm motion |

---

## MuJoCo viewer controls

When the sim window opens:

| Action | Control |
|---|---|
| Rotate view | Left click + drag |
| Pan view | Right click + drag |
| Zoom | Scroll wheel |
| Pause / unpause | `Space` |
| Reset | `Backspace` |

There is no keyboard teleop — the arm is controlled programmatically by the pipeline.

---

## `XKB_CONFIG_ROOT` note

MuJoCo's viewer on Ubuntu sometimes fails to find keyboard config. The prefix:

```bash
XKB_CONFIG_ROOT=/usr/share/X11/xkb python ...
```

fixes this. It's already baked into `pixi run sim`.
