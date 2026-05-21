import argparse
import random
import time
import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from discoverse.envs.airbot_play_base import AirbotPlayCfg, AirbotPlayGrasp
from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ASSERT_DIR
from transforms3d.quaternions import quat2mat


from discoverse.examples.test_airbot_play.p3_utils import (
    GraspDetector,
    transfer_grasps,
    get_end_effector_trans,
    get_pre_grasp,
    get_post_grasp,
    inverse_kinematics,
    smooth_motion,
)


def _set_global_seed(seed):
    """Set global RNG seeds for reproducible behavior across runs."""
    random.seed(seed)
    np.random.seed(seed)


def run_evaluation(trials=10, lift_threshold=0.05, device="cuda", seed=12345):
    """Run grasp evaluation with reproducible random object poses."""
    _set_global_seed(seed)

    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/airbot_play_object.xml"
    cfg.obs_rgb_cam_id = [0]
    cfg.obs_depth_cam_id = [0]
    cfg.render_set = {
        "fps": 30,
        "width": 640,
        "height": 480,
    }

    env = AirbotPlayGrasp(cfg)
    env.enable_ramdom_pose()
    robot_urdf = os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    arm_fik = AirbotPlayFIK(robot_urdf)

    # Initialize detector once
    try:
        detector = GraspDetector(device)
    except Exception:
        detector = GraspDetector("cpu")

    successes = 0

    for t in range(trials):
        print(f"Trial {t+1}/{trials}")
        # AirbotPlayGrasp.resetState uses np.random.uniform for object pose.
        # Reset the RNG state before each reset so each trial pose is reproducible.
        np.random.seed(seed + t)
        # Reset environment and get initial observation
        env.reset()
        # small wait for any rendering / state updates
        time.sleep(0.2)
        new_control = np.zeros(env.nj)
        obs, pri_obs, rew, ter, info = env.step(new_control)

        # get object initial z position
        body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, env.body_name)
        if body_id < 0:
            print("Warning: object body not found in model; marking trial as failure")
            continue
        initial_z = float(env.mj_data.xpos[body_id][2])

        img = pri_obs.get('img', [None])[0]
        dep = pri_obs.get('dep', [None])[0]
        if dep is None or img is None:
            print("Observation missing; marking trial as failure")
            continue

        # detect grasp in RGBD image
        try:
            grasp2d = detector.predict_grasp(img, dep)
            grasp3d = detector.grasp_2d_to_3d(grasp2d, dep)
        except Exception as e:
            print("Grasp detection failed")
            continue

        # transform grasp from camera to world
        cam = env.mj_model.camera("eye_from_top")
        camera_pos = cam.pos
        camera_quat = cam.quat
        transfer_grasps([grasp3d], quat2mat(camera_quat), camera_pos)

        # compute end-effector transform and poses
        try:
            target_mat, target_pos, width, depth = get_end_effector_trans(grasp3d)
            pre_pos = get_pre_grasp(target_mat, target_pos)
            post_pos = get_post_grasp(target_mat, target_pos, depth)
        except Exception as e:
            print("Failed to compute grasp transforms:", e)
            continue

        # move to pre-grasp
        try:
            target_control = inverse_kinematics(env, arm_fik, pre_pos, target_mat)
            target_control[6] = 0.0  # neutral gripper while moving
            smooth_motion(env, target_control, max_step=200, threshold=0.01)

            # open gripper
            target_control = env.mj_data.qpos[:env.nj].copy()
            target_control[6] = 1.0
            smooth_motion(env, target_control, max_step=30, threshold=0)

            # move to grasp (post) pose
            target_control = inverse_kinematics(env, arm_fik, post_pos, target_mat)
            target_control[6] = 1.0
            smooth_motion(env, target_control, max_step=30, threshold=0.01)

            # close gripper to grasp
            target_control = env.mj_data.qpos[:env.nj].copy()
            target_control[6] = 0.1
            smooth_motion(env, target_control, max_step=20, threshold=0)

            # lift
            target_control = env.mj_data.qpos[:env.nj].copy()
            target_control[1] = -1.7
            target_control[2] = 2.0
            smooth_motion(env, target_control, max_step=40, threshold=0)

            # allow physics to settle
            for _ in range(10):
                obs, pri_obs, rew, ter, info = env.step(env.mj_data.qpos[:env.nj])
                time.sleep(0.02)

            current_z = float(env.mj_data.xpos[body_id][2])
            dz = current_z - initial_z
            print(f"Initial z: {initial_z:.3f}, current z: {current_z:.3f}, dz: {dz:.3f}")
            if dz > lift_threshold:
                print("Success: object lifted")
                successes += 1
            else:
                print("Failure: object not sufficiently lifted")
        except Exception as e:
            print("Execution failed during motion:", e)
            continue

    success_rate = successes / float(trials)
    print(f"\nEvaluation completed: {successes}/{trials} successes. Success rate: {success_rate:.2f}")
    return success_rate


def parse_args():
    """Parse command-line arguments for evaluation settings."""
    parser = argparse.ArgumentParser(description="Evaluate grasp success with reproducible random object poses")
    parser.add_argument("--trials", type=int, default=10, help="Number of evaluation trials")
    parser.add_argument("--lift-threshold", type=float, default=0.05, help="Minimum lift height for success")
    parser.add_argument("--device", default="cuda", help="Inference device for grasp detector")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed controlling object initial pose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(
        trials=args.trials,
        lift_threshold=args.lift_threshold,
        device=args.device,
        seed=args.seed,
    )
