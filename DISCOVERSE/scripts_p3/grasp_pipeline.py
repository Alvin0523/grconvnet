import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import quat2mat

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.airbot_play import AirbotPlayFIK
from discoverse.envs.airbot_play_base import AirbotPlayBase, AirbotPlayCfg, AirbotPlayGrasp
from discoverse.examples.test_airbot_play.p3_utils import (
	CircleGraspDetector,
	GraspDetector,
	get_end_effector_trans,
	get_post_grasp,
	get_pre_grasp,
	inverse_kinematics,
	smooth_motion,
	transfer_grasps,
)


def build_env(detector_type):
	"""Create environment and IK solver based on the detector type."""
	cfg = AirbotPlayCfg()
	cfg.obs_rgb_cam_id = [0]
	cfg.obs_depth_cam_id = [0]
	cfg.render_set = {
		"fps": 30,
		"width": 640,
		"height": 480,
	}

	if detector_type == "circle":
		cfg.mjcf_file_path = "mjcf/airbot_play_circle.xml"
		exec_node = AirbotPlayBase(cfg)
	else:
		cfg.mjcf_file_path = "mjcf/airbot_play_object.xml"
		exec_node = AirbotPlayGrasp(cfg)

	robot_urdf = os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
	arm_fik = AirbotPlayFIK(robot_urdf)
	return exec_node, arm_fik


def detect_grasp(detector_type, img, dep, device, save_path, grasp_id):
	"""Run selected grasp detector and return one 3D grasp candidate."""
	if detector_type == "circle":
		detector = CircleGraspDetector()
		grasp2d = detector.predict_grasp(img, save_path)
	else:
		detector = GraspDetector(device)
		grasp2d = detector.predict_grasp(dep, id=grasp_id)

	if grasp2d is None:
		raise RuntimeError("No valid grasp detected.")

	grasp3d = detector.grasp_2d_to_3d(grasp2d, dep)
	return grasp2d, grasp3d


def grasp_execution(exec_node, arm_fik, grasp3d):
	"""Execute the shared pick-and-place sequence for the given grasp pose."""
	camera_pos = exec_node.mj_model.camera("eye_from_top").pos
	camera_quat = exec_node.mj_model.camera("eye_from_top").quat
	transfer_grasps([grasp3d], quat2mat(camera_quat), camera_pos)
	print("Transformed 3D Grasp:", grasp3d)

	target_mat, target_pos, _, depth = get_end_effector_trans(grasp3d)
	pre_pos = get_pre_grasp(target_mat, target_pos)
	post_pos = get_post_grasp(target_mat, target_pos, depth)

	print("Move to pre-grasping pose")
	target_control = inverse_kinematics(exec_node, arm_fik, pre_pos, target_mat)
	target_control[6] = 0
	smooth_motion(exec_node, target_control, max_step=200, threshold=0.01)

	print("Open the gripper")
	target_control = exec_node.mj_data.qpos[:7].copy()
	target_control[6] = 1
	smooth_motion(exec_node, target_control, max_step=30, threshold=0)

	print("Move to post-grasping pose")
	target_control = inverse_kinematics(exec_node, arm_fik, post_pos, target_mat)
	target_control[6] = 1
	smooth_motion(exec_node, target_control, max_step=30, threshold=0.01)

	print("Close the gripper")
	target_control = exec_node.mj_data.qpos[:7].copy()
	target_control[6] = 0.5
	smooth_motion(exec_node, target_control, max_step=20, threshold=0)

	print("Lift the gripper")
	target_control = exec_node.mj_data.qpos[: exec_node.nj].copy()
	target_control[1] = -1.7
	target_control[2] = 2.0
	smooth_motion(exec_node, target_control, max_step=40, threshold=0)

	print("Put the object into a bin")
	put_pose = np.array([0.6, -0.0, 0.30])
	put_quat = Rotation.from_euler("xyz", [180, -40, -90], degrees=True).as_quat()
	put_mat = quat2mat(put_quat)
	target_control = inverse_kinematics(exec_node, arm_fik, put_pose, put_mat)
	target_control[6] = 0.1
	smooth_motion(exec_node, target_control, max_step=100, threshold=0)

	print("Open the gripper")
	target_control = exec_node.mj_data.qpos[:7].copy()
	target_control[6] = 1
	smooth_motion(exec_node, target_control, max_step=20, threshold=0)


def grasp_pipeline(detector_type="circle", device="cuda", save_path=None, grasp_id=123):
	"""Run a full grasp pipeline with either circle or network detector."""
	exec_node, arm_fik = build_env(detector_type)

	exec_node.reset()
	time.sleep(5)

	print("Capture the scene")
	new_control = np.zeros(7)
	_, pri_obs, _, _, _ = exec_node.step(new_control)
	img = pri_obs["img"][0]
	dep = pri_obs["dep"][0]

	if save_path is None:
		save_path = os.path.join(os.getcwd(), "detected_circle_and_grasp.png")

	grasp2d, grasp3d = detect_grasp(detector_type, img, dep, device, save_path, grasp_id)
	print("2D Grasp:", grasp2d)
	print("3D Grasp:", grasp3d)

	grasp_execution(exec_node, arm_fik, grasp3d)


def parse_args():
	"""Parse command-line arguments for detector and runtime options."""
	parser = argparse.ArgumentParser(description="Unified grasp pipeline for circle and network detectors")
	parser.add_argument("--detector", choices=["circle", "net"], default="circle", help="Detector mode")
	parser.add_argument("--device", default="cuda", help="Torch device for net detector")
	parser.add_argument("--save-path", default=None, help="Visualization output path for circle detector")
	parser.add_argument("--grasp-id", type=int, default=123, help="Visualization id used by net detector")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	grasp_pipeline(
		detector_type=args.detector,
		device=args.device,
		save_path=args.save_path,
		grasp_id=args.grasp_id,
	)
