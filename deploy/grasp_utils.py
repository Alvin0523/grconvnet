import numpy as np
import cv2
import json
import time
import os
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def get_transformation(R, t):
    """
    Creates a 4x4 transformation matrix from a rotation matrix and translation vector.

    Args:
        R (np.ndarray): The 3x3 rotation matrix.
        t (np.ndarray): The 3-element translation vector.

    Returns:
        np.ndarray: The 4x4 transformation matrix.
    """
    trans = np.eye(4)
    trans[:3, :3] = R
    trans[:3, 3] = t
    return trans


def mat2quat(mat):
    """
    Converts a 3x3 rotation matrix to a quaternion.

    Args:
        mat (np.ndarray): The 3x3 rotation matrix.

    Returns:
        np.ndarray: The quaternion (x, y, z, w).
    """
    return R.from_matrix(mat).as_quat()


def quat2mat(quat):
    """
    Converts a quaternion to a 3x3 rotation matrix.

    Args:
        quat (np.ndarray): The quaternion (x, y, z, w).

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    return R.from_quat(quat).as_matrix()


def get_end_effector_trans(gripper, cam2robot):
    """
    Calculates the target pose for the end effector in the robot's base frame.

    Args:
        gripper (list): A list containing the grasp rotation matrix, center, width, and depth.
        cam2robot (np.ndarray): The 4x4 transformation matrix from camera to robot base frame.

    Returns:
        tuple: A tuple containing the target rotation matrix, target position, width, and depth.
    """
    grasp_in_robot = transfer_grasp(gripper, cam2robot)
    width = min((grasp_in_robot[2] + 0.02) / 0.08, 1)  # For airbot
    depth = grasp_in_robot[3]
    return grasp_in_robot[0], grasp_in_robot[1], width, depth


def transfer_grasp(gripper, tf_mat):
    """
    Transfers a grasp pose using a transformation matrix.

    Args:
        gripper (list): A list containing the grasp rotation matrix, center, width, and depth.
        tf_mat (np.ndarray): The 4x4 transformation matrix to apply.

    Returns:
        tuple: A tuple containing the transformed rotation matrix, position, width, and depth.
    """
    R = gripper[0]
    t = gripper[1]
    grasp_trans = get_transformation(R, t)
    target_trans = tf_mat @ grasp_trans  # in robot base frame
    target_mat = target_trans[:3, :3]
    target_pos = target_trans[:3, 3]
    width = gripper[2]
    depth = gripper[3]
    return target_mat, target_pos, width, depth


def get_pre_grasp(target_mat, target_pos):
    """
    Calculates the pre-grasp position, which is slightly above the target position.

    Args:
        target_mat (np.ndarray): The 3x3 target rotation matrix.
        target_pos (np.ndarray): The 3-element target position vector.

    Returns:
        np.ndarray: The 3-element pre-grasp position vector.
    """
    vector = np.array([-1, 0, 0])
    target_vector = (target_mat @ vector.T).T
    pre_pos = target_pos + target_vector * 0.04
    return pre_pos


def get_post_grasp(target_mat, target_pos, depth):
    """
    Calculates the post-grasp position, which is slightly above the target position after grasping.

    Args:
        target_mat (np.ndarray): The 3x3 target rotation matrix.
        target_pos (np.ndarray): The 3-element target position vector.
        depth (float): The depth of the object.

    Returns:
        np.ndarray: The 3-element post-grasp position vector.
    """
    vector = np.array([1, 0, 0])
    target_vector = (target_mat @ vector.T).T
    pre_pos = target_pos + target_vector * depth
    return pre_pos


def fill_hole(img_arr):
    """
    Fills holes (zero values) in an image array using distance transform.

    Args:
        img_arr (np.ndarray): The input image array.

    Returns:
        np.ndarray: The image array with holes filled.
    """
    # TODO: implement the function to overcome sim2real gap for depth images
    return img_arr


def intrinsics_to_numpy(intrinsics):
    """
    Converts RealSense intrinsics to a NumPy array.

    Args:
        intrinsics (rs.intrinsics): The RealSense camera intrinsics.

    Returns:
        np.ndarray: The 3x3 camera intrinsics matrix.
    """
    return np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])


def extrinsics_to_numpy(extrinsics):
    """
    Converts RealSense extrinsics to a NumPy array.

    Args:
        extrinsics (rs.extrinsics): The RealSense camera extrinsics.

    Returns:
        np.ndarray: The 4x4 camera extrinsics matrix.
    """
    ex = np.eye(4)
    ex[:3, :3] = np.array(extrinsics.rotation).reshape((3, 3))
    ex[:3, 3] = np.array(extrinsics.translation)
    return ex


def read_extrinsics(filename="calibration.json"):
    """
    Reads the camera-to-robot extrinsics from a JSON file.

    Args:
        filename (str, optional): The name of the JSON file. Defaults to "calibration.json".

    Returns:
        np.ndarray: The 4x4 camera-to-robot transformation matrix.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    extrinsics = np.array(data['transformation_matrix']).reshape(4, 4)
    print("Camera2Robot Transformation: ", extrinsics)
    return extrinsics


def save_grasp(grasp_dict, filename="grasp_data.json"):
    """
    Save the grasp dictionary to a JSON file.

    Args:
        grasp_dict (dict): Dictionary containing grasp data for objects.
        filename (str, optional): Name of the output JSON file. Defaults to "grasp_data.json".
    """
    serializable_dict = {}
    for obj_name, grasps in grasp_dict.items():
        serializable_grasps = []
        for grasp in grasps:
            serializable_grasp = [
                grasp[0].tolist(),  # Rotation matrix
                grasp[1].tolist(),  # Translation vector
                grasp[2],           # Width
                grasp[3] if len(grasp) > 3 else None  # Optional parameter
            ]
            serializable_grasps.append(serializable_grasp)
        serializable_dict[obj_name] = serializable_grasps
    
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, indent=4)
    
    print(f"Grasp data saved to {os.path.abspath(filename)}")


def load_grasp(filename):
    """
    Load the grasp dictionary from a JSON file.

    Args:
        filename (str): Name of the input JSON file.

    Returns:
        dict: Dictionary containing grasp data for objects with numpy arrays.
    """
    with open(filename, 'r') as f:
        loaded_dict = json.load(f)
    
    grasp_dict = {}
    for obj_name, grasps in loaded_dict.items():
        loaded_grasps = []
        for grasp in grasps:
            loaded_grasp = [
                np.array(grasp[0]),  # Rotation matrix
                np.array(grasp[1]),   # Translation vector
                grasp[2],           # Width
                grasp[3] if len(grasp) > 3 else None  # Optional parameter
            ]
            loaded_grasps.append(loaded_grasp)
        grasp_dict[obj_name] = loaded_grasps
    
    print(f"Grasp data loaded from {os.path.abspath(filename)}")
    return grasp_dict


def create_parallel_gripper(R, t, width=0.08, depth=0.02, finger_length=0.08, finger_thickness=0.01):
    """
    Create a parallel gripper geometry.

    Args:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3-element translation vector.
        width (float): Distance between fingers when open.
        depth (float): Depth of the gripper.
        finger_length (float): Length of each finger.
        finger_thickness (float): Thickness of fingers.

    Returns:
        list: List of Open3D geometries representing the gripper.
    """
    # Create gripper base (cube)
    app = o3d.geometry.TriangleMesh.create_box(
        width=finger_thickness,
        height=4*finger_thickness,
        depth=finger_thickness
    )
    app.paint_uniform_color([0.2, 0.5, 0.8])
    app.translate([-finger_thickness/2, -4*finger_thickness-0.08, -finger_thickness/2])
    
    base = o3d.geometry.TriangleMesh.create_box(
        width=width-finger_thickness,
        height=finger_thickness,
        depth=finger_thickness
    )
    base.paint_uniform_color([0.2, 0.5, 0.8])
    base.translate([-width/2+finger_thickness-finger_thickness/2, -0.08, -finger_thickness/2])
    
    # Create left finger
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_thickness,
        height=finger_length,
        depth=finger_thickness
    )
    left_finger.paint_uniform_color([0.2, 0.5, 0.8])
    left_finger.translate([-width/2-finger_thickness/2, -0.08, 0-finger_thickness/2])
    
    # Create right finger
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_thickness,
        height=finger_length,
        depth=finger_thickness
    )
    right_finger.paint_uniform_color([0.2, 0.5, 0.8])
    right_finger.translate([width/2-finger_thickness/2, -0.08, 0-finger_thickness/2])

    tf = get_transformation(R, t)
    app.transform(tf)
    base.transform(tf)
    left_finger.transform(tf)
    right_finger.transform(tf)
    
    return [app, base, left_finger, right_finger]