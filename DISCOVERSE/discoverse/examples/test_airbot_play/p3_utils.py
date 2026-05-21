#!/usr/bin/env python3

"""Code adapted from https://github.com/mattcorsaro1/mj_pc."""

import math
import numpy as np

from PIL import Image as PIL_Image

import open3d as o3d
import cv2
import json
from transforms3d.quaternions import mat2quat
from discoverse.utils import step_func

"""Convert a quaternion into a 3x3 NumPy rotation matrix.

Args:
    quat: Quaternion in (w, x, y, z) order.

Returns:
    A 3x3 NumPy rotation matrix.
"""

def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # The snippet below can generate a 3x3 rotation matrix, but we avoid it
    # to keep this file independent of MuJoCo.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This implementation is adapted from SciPy source code:
    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""Convert a flat list (length 9) into a 3x3 NumPy rotation matrix.

Args:
    rot_mat_arr: Rotation matrix values in row-major order.

Returns:
    A 3x3 NumPy rotation matrix.
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""Build a 4x4 transform matrix from position and rotation.

Args:
    pos: Position vector of length 3.
    rot_mat: 3x3 NumPy rotation matrix.

Returns:
    A 4x4 homogeneous transformation matrix.
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""Create an Open3D camera intrinsic object from a NumPy intrinsic matrix.

Args:
    cam_mat: 3x3 NumPy camera intrinsic matrix.
    width: Image width in pixels.
    height: Image height in pixels.

Returns:
    Open3D pinhole camera intrinsic object.
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def get_transformation(R, t):
    """Construct a 4x4 homogeneous transform from rotation and translation."""
    trans = np.zeros((4,4))
    trans[:3,:3] = R
    trans[:3,3] = t
    trans[3,3] = 1
    return trans


def load_grasp(file_name):
    """Load grasp candidates from a JSON file."""
    with open(file_name, 'r') as f:
        grasps = json.load(f)

    grippers = []
    for grasp in grasps:
        t = np.array(grasp['t'])
        R = np.array(grasp['R']).reshape((3,3))
        width = grasp['width']
        depth = grasp['depth']
        gripper = [R, t, width, depth]
        grippers.append(gripper)
    print("%d grasps are loaded!"%len(grippers))
    return grippers


def transfer_grasps(grippers, object_mat, object_pos):
    """Transform grasp poses from object frame into world frame in-place."""
    object_trans = get_transformation(object_mat, object_pos)
    for gripper in grippers:
        R = gripper[0]
        t = gripper[1]
        gripper_trans = get_transformation(R, t)
        world_trans = object_trans @ gripper_trans
        gripper[0] = world_trans[:3,:3]
        gripper[1] = world_trans[:3,3]


def get_pre_grasp(target_mat, target_pos):
    """Compute a pre-grasp position slightly offset from the target pose."""
    vector = np.array([-1,0,0])
    target_vector = (target_mat @ vector.T).T
    pre_pos = target_pos + target_vector * 0.05
    return pre_pos


def get_post_grasp(target_mat, target_pos, depth):
    """Compute a post-grasp position by moving along the local approach axis."""
    vector = np.array([1,0,0])
    target_vector = (target_mat @ vector.T).T
    pre_pos = target_pos + target_vector * depth
    return pre_pos


def filter_grasp(target_mat, target_pos):
    """Return True when the grasp height is above the configured threshold."""
    if target_pos[2] > 0.80:
        return True
    else:
        return False


def filter_grasps(grippers):
    """Filter grasp candidates using the default validity rule."""
    valid_grippers = []
    for gripper in grippers:
        R = gripper[0]
        t = gripper[1]
        if filter_grasp(R, t):
            valid_grippers.append(gripper)
    print("There are %d valid grasps!"%len(valid_grippers))
    return valid_grippers


def query_object(exec_node, obj_name):
    """Query object world position and quaternion from MuJoCo data."""
    quat = exec_node.mj_data.body(obj_name).xquat
    pos = exec_node.mj_data.body(obj_name).xpos
    return pos, quat


def update_target(exec_node, new_mat):
    """Update the MuJoCo mocap target pose from a 4x4 transform."""
    mocap_id = exec_node.mj_model.body("target").mocapid[0]
    exec_node.mj_data.mocap_pos[mocap_id] = new_mat[:3,3]
    exec_node.mj_data.mocap_quat[mocap_id] = mat2quat(new_mat[:3,:3])


def evaluate_error(target, current, threshold=0.01):
    """Check whether the first six joints are within an absolute threshold."""
    error = np.abs(current[:6]-target[:6])
    if np.max(error) < threshold:
        return True
    else:
        return False


def smooth_motion(exec_node, target_control, max_step, threshold=0.01, move_speed=2):
    """Move joints toward a target control vector with smooth per-joint steps."""
    for i in range(max_step):
        action = exec_node.mj_data.qpos[:7].copy()
        if evaluate_error(target_control, action, threshold=threshold):
            print("Have reached the target configuration")
            break
        diff = np.abs(action[:6] - target_control[:6])
        joint_move_ratio = diff / (np.max(diff) + 1e-6)
        for i in range(exec_node.nj-1):
            action[i] = step_func(action[i], target_control[i], move_speed * max(joint_move_ratio[i],0.2) * exec_node.delta_t)
        action[6] = target_control[6]
        obs, pri_obs, rew, ter, info = exec_node.step(action)


def inverse_kinematics(exec_node, arm_fik, target_pos, target_mat):
    """Solve IK for a target pose and return a 7D control command."""
    # Visualize the target robot end-effector pose.
    world = np.zeros((4,4))
    world[0,0] = 1
    world[1,1] = 1
    world[2,2] = 1
    world[3,3] = 1
    world[:3,3] = [0, 0.2, 0.78]
    # End-effector pose in the robot base frame.
    trans = get_transformation(target_mat, target_pos)
    # Compute the end-effector pose in the world frame.
    new_mat = world @ trans
    update_target(exec_node, new_mat)
    # Compute joint targets via inverse kinematics.
    new_control = np.zeros(7)
    new_control[:6] = arm_fik.properIK(target_pos, target_mat, exec_node.mj_data.qpos[:6])
    return new_control


def get_end_effector_trans(gripper):
    """Convert a grasp pose into robot-base end-effector target coordinates."""
    R = gripper[0]
    t = gripper[1]
    width = min((gripper[2]+0.02)/0.08,1)
    depth = gripper[3]
    grasp_trans = get_transformation(R,t)
    world = np.zeros((4,4))
    world[0,0] = 1
    world[1,1] = 1
    world[2,2] = 1
    world[3,3] = 1
    world[:3,3] = [0, 0.2, 0.78]
    inv_R = np.zeros((4,4))
    inv_R[3,3] = 1
    inv_R[:3,:3] = world[:3,:3].T
    inv_t = np.eye(4)
    inv_t[:3,3] = -world[:3,3]
    target_trans = inv_t @ inv_R @ grasp_trans # in robot base frame
    bias_mat = np.zeros((3,3))
    bias_mat[0,0] = 1
    bias_mat[1,2] = 1
    bias_mat[2,1] = -1
    target_mat = target_trans[:3,:3]
    target_pos = target_trans[:3,3]
    return target_mat, target_pos, width, depth

class CircleGraspDetector(object):
    """Detect circular objects in an RGB image using HoughCircles and create
    visualizations. Optionally convert the detected 2D grasp to a 3D grasp by
    attaching a GraspDetector instance to this object (set via the
    'grasp_detector' attribute or via constructor).

    Methods:
    - detect_and_visualize(img_rgb, dep, save_path=None): returns a dict with
      keys: 'circle', 'grasp2d', 'grasp3d', 'left', 'right', 'combined'
    """
    def __init__(self, mat=None, dp=1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=200):
        """Initialize circle detector parameters and camera intrinsics."""
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        # Optional external grasp detector to convert 2D grasps to 3D.
        self.grasp_detector = None
        if mat is None:
            self.camera_mat = np.array([[617.79730003, 0, 320], [0, 617.79730003, 240], [0, 0, 1]])
        else:
            # Keep a consistent camera intrinsic attribute name.
            self.camera_mat = mat
    
    def grasp_2d_to_3d(self, grasp, dep):
        """Project a 2D grasp from image space into a simple 3D grasp pose."""
        center, angle, width = grasp
        angle = angle + 3 * np.pi/2
        z = dep[center]
        f = self.camera_mat[0,0]
        center_homo = (center[1], center[0], 1)
        center_3d = z * np.linalg.inv(self.camera_mat) @ center_homo
        center_3d = (center_3d[0], -center_3d[1], -center_3d[2])
        mat_3d = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        mat_bias = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        mat_3d = mat_3d @ mat_bias
        width_3d = width * z / f
        return [mat_3d, center_3d, width_3d, 0.02]

    def predict_grasp(self, img_rgb, save_path):
        """Run HoughCircles on img_rgb and return a 2D grasp candidate."""
        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=self.dp, minDist=self.minDist,
                                   param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)

        detected_circle = None
        grasp2d = None

        if circles is not None and len(circles) > 0:
            circles = np.round(circles[0, :]).astype("int")
            # Use the largest detected circle (by radius).
            circles = sorted(circles, key=lambda x: x[2], reverse=True)
            x, y, r = circles[0]
            detected_circle = (int(x), int(y), int(r))
            rect_w = int(r * 2 + 10)
            rect_h = max(4, int(r * 0.8))
            angle = 0.0
            grasp2d = [(int(y), int(x)), float(angle), float(rect_w)]

        self.visialize_grasp(img_rgb, {'circle': detected_circle, 'grasp2d': grasp2d}, save_path=save_path)
        print(f"Saved visualization to {save_path}")


        return grasp2d

    def visialize_grasp(self, img_rgb, detection, save_path=None):
        """Create left/right/combined overlays from detection results."""
        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        left_overlay = orig_bgr.copy()
        right_overlay = orig_bgr.copy()

        detected_circle = detection.get('circle', None)
        grasp2d_try = detection.get('grasp2d', None)

        if detected_circle is not None:
            x, y, r = detected_circle
            cv2.circle(left_overlay, (x, y), r, (0, 255, 0), 2)
            cv2.circle(left_overlay, (x, y), 2, (0, 0, 255), 3)

        if grasp2d_try is not None:
            (x, y), angle, rect_w = grasp2d_try
            rect_h = max(4, int((rect_w/2) * 0.8))
            box = ((float(x), float(y)), (float(rect_w), float(rect_h)), float(angle))
            try:
                box_pts = cv2.boxPoints(box)
                box_pts = np.int32(np.round(box_pts))
                cv2.drawContours(right_overlay, [box_pts], 0, (0, 255, 0), 2)
            except Exception:
                x1 = int(x - rect_w / 2)
                y1 = int(y - rect_h / 2)
                x2 = int(x + rect_w / 2)
                y2 = int(y + rect_h / 2)
                cv2.rectangle(right_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(right_overlay, (int(x), int(y)), 2, (0, 0, 255), 3)

        combined = None
        try:
            combined = np.concatenate((left_overlay, right_overlay), axis=1)
            if save_path is not None:
                cv2.imwrite(save_path, combined)
        except Exception:
            combined = None

        return left_overlay, right_overlay, combined


import torch

from utils.dataset_processing.grasp import detect_grasps
from inference.post_process import post_process_output


class GraspDetector(CircleGraspDetector):
    def __init__(self, device, mat=None):
        """Load the grasp model and set inference parameters."""
        super().__init__(mat=mat)
        model_name = "grconvnet/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/model.pt"
        self.model = torch.load(model_name, weights_only=False).to(device)
        self.device = device
        self.input_size = 300

    def predict_grasp(self, rgb, dep, id=999):
        """Predict one grasp from an RGBD input and map it to original coordinates."""
        top  = (dep.shape[0] - self.input_size) // 2
        left = (dep.shape[1] - self.input_size) // 2

        dep_crop = dep[top:top+self.input_size, left:left+self.input_size]
        rgb_crop = rgb[top:top+self.input_size, left:left+self.input_size]

        # normalise depth: zero-centre, clip to [-1, 1]
        dep_norm = np.clip((dep_crop - dep_crop.mean()), -1, 1).astype(np.float32)

        # normalise RGB: [0,1] then zero-centre, transpose to (3,H,W)
        rgb_norm = rgb_crop.astype(np.float32) / 255.0
        rgb_norm -= rgb_norm.mean()
        rgb_norm = rgb_norm.transpose((2, 0, 1))

        # stack as [depth, R, G, B] matching training order
        x = np.concatenate([dep_norm[None, ...], rgb_norm], axis=0)
        x = torch.tensor(x[None, ...]).to(self.device).float()

        with torch.no_grad():
            pos_img, angle_img, width_img = post_process_output(*self.model(x))
            q   = pos_img[0]
            ang = angle_img[0]
            wid = width_img[0]
            print(f"[GraspDetector] Q max={q.max():.3f} mean={q.mean():.3f}")
            grasps = detect_grasps(q, ang, width_img=wid, no_grasps=1)
            if not grasps:
                print("[GraspDetector] no peak above threshold, using argmax")
                from utils.dataset_processing.grasp import Grasp
                peak = np.unravel_index(np.argmax(q), q.shape)
                grasp = Grasp(peak, ang[peak])
                grasp.length = wid[peak]
                grasp.width  = grasp.length / 2
            else:
                grasp = grasps[0]

        center = (top + grasp.center[0], left + grasp.center[1])
        angle  = grasp.angle
        width  = grasp.length
        return [center, angle, width]

