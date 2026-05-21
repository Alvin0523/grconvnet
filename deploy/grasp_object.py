import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time

from grasp_utils import *

from airbot_py.arm import AIRBOTPlay, SpeedProfile, RobotMode

AIRBOT_IP = "192.168.209.101"
AIRBOT_PORT = 50051

class RobotInterface:
    """
    A simple class to interface with the Airbot robot.
    """
    waypoints = []
    # Initial Pose
    waypoints.append([[0.1638453786497167, 0.003118620159832492, 0.14824917440673882], [0.5797231982914678, 0.4135374835194798, -0.40858724532770846, 0.5709327684085082]])
    # Pre-Pre Grasp
    waypoints.append([[0.3638453786497167, 0.003118620159832492, 0.14824917440673882], [0.5797231982914678, 0.4135374835194798, -0.40858724532770846, 0.5709327684085082]])

    def __init__(self):
        """
        Initializes the RobotInterface, establishing a connection to the Airbot robot.
        """

    def reset(self):
        """
        Resets the robot to its initial pose.

        Returns:
            bool: True if the reset was successful.
        """
        with AIRBOTPlay(url=AIRBOT_IP, port=AIRBOT_PORT) as robot:
            robot.set_speed_profile(SpeedProfile.SLOW)
            robot.move_to_cart_pose(self.waypoints[1])
            time.sleep(2)
            robot.move_to_cart_pose(self.waypoints[0])
            time.sleep(2)
        return True
    
    def idle_grasp(self):
        """
        Moves the robot to the pre-pre-grasp pose.

        Returns:
            bool: True if the movement was successful.
        """
        with AIRBOTPlay(url=AIRBOT_IP, port=AIRBOT_PORT) as robot:
            robot.set_speed_profile(SpeedProfile.SLOW)
            robot.move_to_cart_pose(self.waypoints[1])
            time.sleep(2)
        return True

    def move_to(self, pose, end=None):
        """
        Moves the robot to a specified pose.

        Args:
            pose (list): The target pose [position, quaternion].
            end (float, optional): The end effector state. Defaults to None.
        """
        safe_pose = [[pose[0][0], pose[0][1], max(pose[0][2], 0.008)], pose[1]]
        with AIRBOTPlay(url=AIRBOT_IP, port=AIRBOT_PORT) as robot:
            robot.move_to_cart_pose(safe_pose)
            time.sleep(5)
            if end is not None:
                robot.move_eef_pos([end])
                time.sleep(1)

    def open_gripper(self):
        with AIRBOTPlay(url=AIRBOT_IP, port=AIRBOT_PORT) as robot:
            robot.move_eef_pos([1])
            time.sleep(1)

    def close_gripper(self, width=0.0):
        with AIRBOTPlay(url=AIRBOT_IP, port=AIRBOT_PORT) as robot:
            robot.move_eef_pos([width])
            time.sleep(1)


def run(detector_type):
    assert detector_type in ['circle', 'nn']

    # start camera first so it warms up while robot resets
    pipeline = rs.pipeline()
    config = rs.config()
    cam2robot = read_extrinsics(filename="calibration/calibration.json")
    if detector_type in ['circle', 'nn']:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipeline.start(config)
    time.sleep(3)  # let both streams stabilise

    # start the robot
    api = RobotInterface()
    api.reset()
    # get essential parameters
    profile = pipeline.get_active_profile()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_min = 0.11
    depth_max = 4
    # intrinsics
    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # extrinsics
    depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
    color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))
    # PointCloud
    pc = rs.pointcloud()


    if detector_type == "circle":
        from GraspDetector import GraspDetectorCircle
        detector = GraspDetectorCircle(profile=profile, mat=intrinsics_to_numpy(depth_intrin))
    elif detector_type == "nn":
        from GraspDetectorNN import GraspDetectorNN
        detector = GraspDetectorNN(device="cuda", mat=intrinsics_to_numpy(depth_intrin))

    try:
        counter = 0
        while True:
            counter += 1
            frames = pipeline.wait_for_frames(timeout_ms=15000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_raw = color_frame.get_data()
            depth_raw = depth_frame.get_data()
            if counter >= 50:

                detector.update_raw(color_raw, depth_raw)
                color_image = np.asanyarray(color_raw)
                depth_image = np.asanyarray(depth_raw)
                depth_image = (depth_image.astype(np.float64)/1000.0).astype(np.float32) 
                depth_image = fill_hole(depth_image)

                grasp3d = detector.predict_grasp_3d(color_image, depth_image, id=999)
                print("3D Grasp in camera frame: ", grasp3d)
                if grasp3d is None:
                    print("Fail to generate grasp, please try again!")
                    return
                grasp3d_robot = get_end_effector_trans(grasp3d, cam2robot)
                # grasp3d_robot = transfer_grasp(grasp3d_robot, extrinsics_to_numpy(depth_to_color_extrin))
                print("3D Grasp in robot base: ", grasp3d_robot)
                target_mat, target_pos, width, depth = grasp3d_robot
                width3d = grasp3d_robot[-2]
                pre_pos = get_pre_grasp(target_mat, target_pos)
                post_pos = get_post_grasp(target_mat, target_pos, depth)
                target_quat = mat2quat(target_mat)
                # User check
                print("Time to check grasp!")
                time.sleep(2)
                print("Checking Time end!")
                # Three poses
                pre_pose = [pre_pos.tolist(), target_quat.tolist()]
                target_pose = [target_pos.tolist(), target_quat.tolist()]
                post_pose = [post_pos.tolist(), target_quat.tolist()]
                # Begin Grasp
                print("Grasp Begin!")
                api.idle_grasp()
                print("Move to pre-grasp pose...")
                api.move_to(pre_pose, end=1)
                print(pre_pose)
                print("Move to grasp pose...")
                api.move_to(target_pose, end=width3d)
                print(target_pose)
                print("Move to post-grasp pose...")
                api.move_to(post_pose)
                print(post_pose)
                print("Close gripper...")
                api.close_gripper(width=0.0)
                print("Lift the object...")
                api.move_to(pre_pose)
                print("Open the gripper...")
                api.open_gripper()
                break

    finally:
        pipeline.stop()


if __name__ == "__main__":
    run('nn')
    # run('circle')