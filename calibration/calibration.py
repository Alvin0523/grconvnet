import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import open3d as o3d
import json

def read_points_camera_3d(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        points_camera = data.get("points_3d", [])
        return points_camera


def read_poses_robot_3d():
    filename = "robot_poses.json"
    poses_robot = json.load(open(filename, 'r'))
    mats_robot = []
    for pose in poses_robot:
        mat = _pose_vec_to_mat(pose)
        mats_robot.append(mat)
    return mats_robot


def _pose_vec_to_mat(pose_vec):
    '''[[x, y, z], [x, y, z, w]] -> 4x4 transformation matrix'''
    translation = pose_vec[0]
    quaternion = pose_vec[1]
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


def HornMethod(points_1, points_2):
    # Step 1: Compute centroids
    C1 = np.mean(points_1, axis=0)
    C2 = np.mean(points_2, axis=0)
    
    # Step 2: Center the points
    P1_centered = points_1 - C1
    P2_centered = points_2 - C2
    
    # Step 3: Compute cross-covariance matrix H
    H = np.dot(P1_centered.T, P2_centered)
    
    # Step 4: Perform SVD
    U, _, Vt = np.linalg.svd(H)
    
    # Step 5: Compute the rotation matrix
    R_mat = np.dot(Vt.T, U.T)
    # Check if we need to fix rotation matrix to ensure det(R) = 1
    if np.linalg.det(R_mat) < 0:
        Vt[-1,:] *= -1
        R_mat = np.dot(Vt.T, U.T)
    
    # Step 6: Compute the translation
    t = C2 - np.dot(R_mat, C1)
    
    # Return rotation and translation
    return R_mat, t


def calibration3D(points_robot, points_camera):
    R, t = HornMethod(points_robot, points_camera)
    t_matrix = np.zeros((4,4))
    t_matrix[:3,:3] = R
    t_matrix[:3,3] = t
    t_matrix[3,3] = 1
    return t_matrix


def evaluate_calibration(transformation_matrix, points_robot, points_camera):
    '''
    Evaluate the calibration by transforming the points_robot to the camera frame using the transformation_matrix.
    Calculate the error between the transformed points and the points_camera.
    '''
    # Convert points to numpy arrays
    points_robot = np.array(points_robot)
    points_camera = np.array(points_camera)

    # Transform the points_robot to the camera frame
    transformed_points_robot = np.ones((points_robot.shape[0], 4))
    transformed_points_robot[:, :3] = points_robot
    transformed_points_robot = transformed_points_robot.T
    transformed_points_robot = transformation_matrix @ transformed_points_robot
    transformed_points_robot = transformed_points_robot[:3, :].T

    # Calculate the error
    error = np.linalg.norm(points_camera - transformed_points_robot, axis=1)
    mean_error = np.mean(error)

    return mean_error

def reprojection_visualization(transformation_matrix, points_robot, points_camera):
    # Convert points to numpy arrays
    points_robot = np.array(points_robot)
    points_camera = np.array(points_camera)

    # Transform the points_robot to the camera frame
    transformed_points_robot = np.ones((points_robot.shape[0], 4))
    transformed_points_robot[:, :3] = points_robot
    transformed_points_robot = transformed_points_robot.T
    transformed_points_robot = transformation_matrix @ transformed_points_robot
    transformed_points_robot = transformed_points_robot[:3, :].T

    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(points_robot)
    pcd_robot.paint_uniform_color([0, 1, 0])

    pcd_camera = o3d.geometry.PointCloud()
    pcd_camera.points = o3d.utility.Vector3dVector(points_camera)
    # color should be red
    pcd_camera.paint_uniform_color([1, 0, 0])

    pcd_transformed_robot = o3d.geometry.PointCloud()
    pcd_transformed_robot.points = o3d.utility.Vector3dVector(transformed_points_robot)
    # color should be blue
    pcd_transformed_robot.paint_uniform_color([0, 0, 1])

    # Visualize the point clouds
    o3d.visualization.draw_geometries([pcd_robot])
    o3d.visualization.draw_geometries([pcd_camera])
    o3d.visualization.draw_geometries([pcd_camera, pcd_transformed_robot])
    
def get_points_robot_3d(poses):
    points_robot = []
    for pose in poses:
        target_vec = pose[:3, 3]
        target_mat = pose[:3, :3]
        offset = np.array([0, 0, 0]) # offset
        end_offset = (target_mat @ offset.T).T
        end_pos = target_vec + end_offset
        points_robot.append(end_pos)
    return points_robot


def save_calibration(t_mat, filename='calibration.json'):
    # Convert the transformation matrix to a dictionary
    calibration_data = {
        'transformation_matrix': t_mat.reshape(-1).tolist()
    }

    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    

if __name__ == "__main__":
    filename = 'points.json'
    points_camera = read_points_camera_3d(filename)
    points_robot = get_points_robot_3d(read_poses_robot_3d())
    print("Robot")
    for point in points_robot:
        print(point)
    print("Camera")
    for point in points_camera:
        print(point)

    # Perform hand-eye calibration
    transformation_matrix = calibration3D(points_robot, points_camera)
    mean_error = evaluate_calibration(transformation_matrix, points_robot, points_camera)
    print("Transformation Matrix from Robot Base to Camera Frame:")
    print(transformation_matrix)
    print("Mean Error:", mean_error)
    ###
    cam2robot = calibration3D(points_camera, points_robot)
    mean_error = evaluate_calibration(cam2robot, points_camera, points_robot)
    print("Transformation Matrix from Camera Frame to Robot Base:")
    print(cam2robot)
    print("Mean Error:", mean_error)

    save_calibration(cam2robot)

    # Visualize the reprojection
    reprojection_visualization(transformation_matrix, points_robot, points_camera)