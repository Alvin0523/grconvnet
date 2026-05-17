---
icon: lucide/bot
---

# Real Robot Deployment

## Hardware

| Component | Spec |
|---|---|
| Arm | AIRBOT Play — 6 DOF + gripper |
| Camera | Intel RealSense D435i — RGB-D 640×480 @ 30fps |
| Connection | Ethernet · IP `192.168.209.101` · port `50051` |

---

## Entry point

```bash
pixi run deploy          # NN grasp detector (needs trained model)
pixi run deploy-circle   # circle detector (no model, good first test)
```

Both call `grasp_object.py`.

---

## `grasp_object.py`

Real robot equivalent of the DISCOVERSE `grasp_pipeline.py`. The logic is identical — only the camera and arm interfaces differ.

```
RealSense D435i → pyrealsense2
                     ↓
              depth image
                     ↓
         GraspDetector / GraspDetectorNN
                     ↓
              3D grasp (camera frame)
                     ↓
         cam2robot transform (calibration.json)
                     ↓
              3D grasp (robot base frame)
                     ↓
         RobotInterface (AIRBOT SDK)
                     ↓
         pre-grasp → grasp → post-grasp → bin
```

### Check arm connection first

```bash
pixi run robot-online
```

Prints `Online` if the arm responds. If it fails, check Ethernet connection and that the arm is powered on.

---

## Detectors

### Circle detector (`GraspDetector.py` + `GraspDetectorCircle`)

Finds circular objects using OpenCV Hough circles. No model needed.

```python
from GraspDetector import GraspDetectorCircle
detector = GraspDetectorCircle(profile=profile, mat=intrinsics_matrix)
grasp3d = detector.predict_grasp_3d(color_image, depth_image)
```

### NN detector (`GraspDetectorNN.py`)

Wraps GR-ConvNet. Loads a checkpoint and runs inference on the depth image.

```python
from GraspDetectorNN import GraspDetectorNN
detector = GraspDetectorNN(model_path="output/models/.../epoch_XX_iou_0.XX", device="cuda")
grasp3d = detector.predict_grasp_3d(color_image, depth_image, id=999)
```

!!! note "Model path"
    Update `model_path` in `grasp_object.py` to point to your latest trained checkpoint in `output/models/`.

---

## Calibration

Calibration gives the 4×4 `cam2robot` transform matrix that converts grasp poses from camera space to robot base space.

### Workflow

```bash
pixi run calib-test      # 1. verify RealSense connects and streams
pixi run calib-record    # 2. move arm to calibration positions, record EEF poses
pixi run calib-label     # 3. open captured image, click on calibration markers
pixi run calib-run       # 4. solve for transform → writes calibration/calibration.json
```

### Files

| File | What it is |
|---|---|
| `calibration/calibration.json` | Output — 4×4 cam2robot transform matrix |
| `calibration/robot_poses.json` | Recorded robot end-effector positions during calib |
| `calibration/calibration.py` | Solves PnP from point correspondences |
| `calibration/label_points.py` | GUI to click calibration points in image |
| `calibration/record_robot_points.py` | Records EEF positions via AIRBOT SDK |
| `calibration/test_realsense.py` | Streams camera to verify connection |
| `robot_poses.json` | Separate calibration session poses (different from `calibration/`) |

---

## `grasp_utils.py` — shared math

Used by both `grasp_object.py` (real) and `p3_utils.py` (sim).

| Function | What it does |
|---|---|
| `get_transformation(R, t)` | Build 4×4 matrix from R + t |
| `mat2quat(mat)` / `quat2mat(quat)` | Rotation conversions |
| `transfer_grasp(gripper, tf_mat)` | Apply 4×4 transform to a grasp |
| `get_end_effector_trans(gripper, cam2robot)` | Full cam→robot, returns `(R, pos, width, depth)` |
| `get_pre_grasp(R, pos)` | 4cm above grasp along approach axis |
| `get_post_grasp(R, pos, depth)` | Retract position post-grasp |
| `fill_hole(img)` | Depth inpainting stub — fills zero pixels for sim2real |
| `read_extrinsics(filename)` | Load `calibration.json` → 4×4 matrix |
| `save_grasp / load_grasp` | Persist grasp dicts to JSON |
| `create_parallel_gripper(R, t, ...)` | Open3D gripper geometry for visualisation |

---

## sim2real gap

The main issue going from sim to real: **depth image noise**.

`fill_hole()` in `grasp_utils.py` is currently a stub (`return img_arr`). Before real robot sessions, implement depth inpainting here — e.g. using `scipy.ndimage.distance_transform_edt` to fill zero pixels with nearest-neighbour values. This dramatically improves GR-ConvNet accuracy on real depth images.

---

## Competition day checklist

- [ ] Trained model checkpoint in `output/models/`
- [ ] `model_path` updated in `grasp_object.py`
- [ ] `calibration/calibration.json` is fresh (recalibrate if camera moved)
- [ ] `pixi run robot-online` confirms arm connection
- [ ] `pixi run calib-test` confirms camera streams
- [ ] `pixi run deploy-circle` test run succeeds
- [ ] Emergency stop within reach at all times
