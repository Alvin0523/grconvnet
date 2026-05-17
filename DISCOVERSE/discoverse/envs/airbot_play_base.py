import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

from discoverse.airbot_play import AirbotPlayFIK
from transforms3d.quaternions import quat2mat
import os

class AirbotPlayCfg(BaseConfig):
    mjcf_file_path = "mjcf/airbot_play_floor.xml"
    decimation     = 4
    timestep       = 0.005
    sync           = True
    headless       = False
    init_key       = "home"
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    obs_rgb_cam_id  = None
    rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict = {
        "arm_base"  : "airbot_play/arm_base.ply",
        "link1"     : "airbot_play/link1.ply",
        "link2"     : "airbot_play/link2.ply",
        "link3"     : "airbot_play/link3.ply",
        "link4"     : "airbot_play/link4.ply",
        "link5"     : "airbot_play/link5.ply",
        "link6"     : "airbot_play/link6.ply",
        "left"      : "airbot_play/left.ply",
        "right"     : "airbot_play/right.ply",
    }
    

class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        self.nj = 7
        super().__init__(config)

    def post_load_mjcf(self):
        try:
            self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.nj]
            self.init_joint_ctrl = self.mj_model.key(self.config.init_key).ctrl[:self.nj]
        except KeyError as e:
            self.init_joint_pose = np.zeros(self.nj)
            self.init_joint_ctrl = np.zeros(self.nj)

        self.sensor_joint_qpos = self.mj_data.sensordata[:self.nj]
        self.sensor_joint_qvel = self.mj_data.sensordata[self.nj:2*self.nj]
        self.sensor_joint_force = self.mj_data.sensordata[2*self.nj:3*self.nj]
        self.sensor_endpoint_posi_local = self.mj_data.sensordata[3*self.nj:3*self.nj+3]
        self.sensor_endpoint_quat_local = self.mj_data.sensordata[3*self.nj+3:3*self.nj+7]
        self.sensor_endpoint_linear_vel_local = self.mj_data.sensordata[3*self.nj+7:3*self.nj+10]
        self.sensor_endpoint_gyro = self.mj_data.sensordata[3*self.nj+10:3*self.nj+13]
        self.sensor_endpoint_acc = self.mj_data.sensordata[3*self.nj+13:3*self.nj+16]

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        print("    arm .qpos  = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("    arm .qvel  = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))
        print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))
        print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))

        print("    sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("    sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0
        self.mj_data.ctrl[:self.nj] = np.clip(action[:self.nj], self.mj_model.actuator_ctrlrange[:self.nj,0], self.mj_model.actuator_ctrlrange[:self.nj,1])
    
    def get_joint_range(self):
        return self.mj_model.actuator_ctrlrange[:self.nj,0], self.mj_model.actuator_ctrlrange[:self.nj,1]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time" : self.mj_data.time,
            "jq"   : self.sensor_joint_qpos.tolist(),
            "jv"   : self.sensor_joint_qvel.tolist(),
            "jf"   : self.sensor_joint_force.tolist(),
            "ep"   : self.sensor_endpoint_posi_local.tolist(),
            "eq"   : self.sensor_endpoint_quat_local.tolist(),
            "img"  : self.img_rgb_obs_s,
            "dep"  : self.img_depth_obs_s,
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None
    

class AirbotPlayGrasp(AirbotPlayBase):
    def __init__(self, config):
        super().__init__(config)
        self.body_name = 'banana'
        self.random_pose = False


    def set_object(self, object_name):
        self.body_name = object_name
    
    def enable_ramdom_pose(self):
        self.random_pose = True
    
    def disable_random_pose(self):
        self.random_pose = False
    
    def resetState(self):
        super().resetState()
        self.body_name = 'banana'
        # randomize around center (0.0, -0.2, 0.8)
        center = np.array([0.0, -0.2, 0.8])
        if self.random_pose:
            xy_offset = np.random.uniform(-0.1, 0.1, size=2)
            yaw = np.random.uniform(-np.pi, np.pi)
        else:
            xy_offset = np.zeros(2)
            yaw = 0.0
        pos = center.copy()
        pos[0] += xy_offset[0]
        pos[1] += xy_offset[1]
        # rotation restricted to plane -> yaw only
        quat_xyzw = Rotation.from_euler('xyz', [0, 0, yaw]).as_quat()  # returns x,y,z,w
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=self.mj_data.qpos.dtype)  # w,x,y,z
        # set free-body qpos (pos then quat) for the 'banana' body and clear velocities/forces
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if body_id >= 0:
            jnt_id = int(self.mj_model.body_jntadr[body_id])
            if jnt_id >= 0:
                qpos_adr = int(self.mj_model.jnt_qposadr[jnt_id])
                self.mj_data.qpos[qpos_adr:qpos_adr+3] = pos.astype(self.mj_data.qpos.dtype)
                self.mj_data.qpos[qpos_adr+3:qpos_adr+7] = quat
                dof_adr = int(self.mj_model.jnt_dofadr[jnt_id])
                # zero linear/angular velocities and applied forces
                self.mj_data.qvel[dof_adr:dof_adr+6] = np.zeros(6, dtype=self.mj_data.qvel.dtype)
                self.mj_data.qfrc_applied[dof_adr:dof_adr+6] = np.zeros(6, dtype=self.mj_data.qfrc_applied.dtype)
                mujoco.mj_forward(self.mj_model, self.mj_data)


if __name__ == "__main__":
    from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
    cfg = AirbotPlayCfg()
    exec_node = AirbotPlayBase(cfg)
    arm_fik = AirbotPlayFIK(os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))

    obs = exec_node.reset()
    action = exec_node.init_joint_pose[:exec_node.nj]
    joint_min = [-3.14, -2.96, -0.087, -2.96, -1.74, -3.14, 0]
    joint_max = [2.09, 0.17, 3.14, 2.96, 1.74, 3.14, 1]
    action = (np.array(joint_min) + np.array(joint_max))/2
    #########
    target_pos = [-0.3000810163019497, -0.6668083923836137, 0.1721906568689643] 
    target_quat = [0.17360054965890626, 0.5146627231806045, -0.8282357194895089, 0.13787938021381413]
    #########
    target_mat = quat2mat(target_quat)
    while exec_node.running:
        new_control = np.zeros(7)
        new_control[:6] = arm_fik.properIK(target_pos, target_mat, exec_node.mj_data.qpos[:6])
        obs, pri_obs, rew, ter, info = exec_node.step(new_control)
        print("Position: ", pri_obs['ep'])
        print("Rotation: ", pri_obs['eq'])
