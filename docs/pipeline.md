---
icon: lucide/workflow
---

# Pipeline

## Overview

```
[RGB-D camera]
      ↓
[GR-ConvNet v2 inference]   ~20ms · grconvnet3 · depth-only
      ↓
[Grasp selection + ordering]   cluster by object, order by bin proximity
      ↓
[AIRBOT Python SDK]   direct arm control
      ↓
[Place in bin]
```

No ROS2. Pure Python from camera to arm.

---

## Grasp representation

GR-ConvNet outputs three pixel-aligned maps over the depth image:

| Map | Symbol | Meaning |
|---|---|---|
| Quality map | Q | Confidence that a grasp centred here succeeds |
| Width map | W | How wide to open the gripper |
| Angle map | Φ | Gripper rotation (cos/sin decomposed to avoid ±π/2 discontinuity) |

A grasp candidate = `(centre_pixel, angle, width)` → projected to 3D using camera intrinsics + depth value.

---

## Coordinate transforms

```
camera frame  →  robot base frame
```

Done via the 4×4 `cam2robot` matrix loaded from `calibration/calibration.json`.

Key functions in `grasp_utils.py`:

| Function | What it does |
|---|---|
| `transfer_grasp(gripper, tf_mat)` | Applies a 4×4 transform to a grasp pose |
| `get_end_effector_trans(gripper, cam2robot)` | Full camera→robot transform, returns `(R, pos, width, depth)` |
| `get_pre_grasp(target_mat, target_pos)` | 4cm above grasp along approach axis |
| `get_post_grasp(target_mat, target_pos, depth)` | Post-grasp retract position |
| `fill_hole(img_arr)` | Depth inpainting for real camera noise (stub — implement for sim2real) |

---

## Sim pipeline (`DISCOVERSE/scripts_p3/grasp_pipeline.py`)

Runs entirely in MuJoCo. Two detector modes:

=== "Circle detector (smoke test)"

    No trained model needed. Detects circular objects with OpenCV.

    ```bash
    XKB_CONFIG_ROOT=/usr/share/X11/xkb python DISCOVERSE/scripts_p3/grasp_pipeline.py --detector circle
    ```

=== "Network detector (competition)"

    Uses GR-ConvNet v2. Requires a trained model checkpoint.

    ```bash
    pixi run sim
    # or
    XKB_CONFIG_ROOT=/usr/share/X11/xkb python DISCOVERSE/scripts_p3/grasp_pipeline.py --detector net
    ```

### What `grasp_pipeline.py` does step by step

1. Creates MuJoCo environment (`AirbotPlayBase` or `AirbotPlayGrasp`)
2. Resets arm to home position
3. Captures RGB + depth frame from simulated overhead camera
4. Runs selected detector → 2D grasp → 3D grasp (via depth projection)
5. Transforms grasp from camera frame to robot base frame
6. Executes: pre-grasp → open gripper → grasp → close gripper → lift → place in bin

---

## Real robot pipeline (`grasp_object.py`)

Same logic as sim but:

- Camera = Intel RealSense D435i (via `pyrealsense2`)
- Arm control = AIRBOT Python SDK (`airbot_py`)
- `cam2robot` loaded from `calibration/calibration.json`

```python
# arm connection
AIRBOT_IP   = "192.168.209.101"
AIRBOT_PORT = 50051
```

The `RobotInterface` class wraps the SDK:

| Method | What it does |
|---|---|
| `reset()` | Move to initial home pose |
| `idle_grasp()` | Move to pre-pre-grasp staging position |
| `move_to(pose, end)` | Cartesian move + optional gripper width |
| `open_gripper()` | Fully open (width=1) |
| `close_gripper(width)` | Close to specified width |

---

## Grasp ordering (key win)

!!! tip "This is where you gain seconds on the scoreboard"

Don't just pick the globally highest-confidence grasp. Instead:

1. **Cluster** grasp candidates by object (spatial clustering)
2. **Order** objects by proximity to the bin drop position
3. Execute in that order → minimises total arm travel

The baseline (highest Q first) ignores arm travel entirely and is significantly slower across 10 objects.
