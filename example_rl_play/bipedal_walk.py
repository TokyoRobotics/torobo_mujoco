import math
import numpy as np
import mujoco, mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import time
from rl_play_base import RLPlayBase


class BipedalWalkControl(RLPlayBase):

    def __init__(self):
        super().__init__(policy_path="policies/bipedal_walk_policy.pt",
                         use_cpu=False,
                         num_dofs=12,
                         num_actions=12,
                         num_single_obs=50,
                         num_frame_stack=1,
                         sim_duration=60,
                         sim_dt=0.005,
                         sim_decimation=4,
                         default_qpos=np.array([
                            0.3911798987199639, 0.3911798987199639,
                            -0.0012423084323551653,0.0012423084323551653,
                            -0.0005505230847620535, 0.0005505230847620535,
                            0.792139781767314, 0.792139781767314,
                            -0.401659587652539,  -0.401659587652539,
                            0.0004349101512974649, -0.0004349101512974649]),
                         isaac_joint_names_order = ["left_leg/joint_1", "right_leg/joint_1",
                                                    "left_leg/joint_2", "right_leg/joint_2",
                                                    "left_leg/joint_3", "right_leg/joint_3",
                                                    "left_leg/joint_4", "right_leg/joint_4",
                                                    "left_leg/joint_5", "right_leg/joint_5",
                                                    "left_leg/joint_6", "right_leg/joint_6"],
                         mujoco_joint_names_order = ["left_leg/joint_" + str(i) for i in range(1, 7)] + ["right_leg/joint_" + str(i) for i in range(1, 7)])

    def get_obs(self, data):
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        linear_v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # converted to base link frame
        angular_v = data.sensor('angular-velocity').data.astype(np.double)
        gravity_vec = r.apply(np.array([0., 0., -1]), inverse=True).astype(np.double)
        return (q, dq, linear_v, angular_v, gravity_vec)

    def play(self):
        vx = 0.6 # 0 ~ 0.8
        vy = 0.0 # fixed
        dyaw = 0.0 # -1.0 ~ 1.0

        model = mujoco.MjModel.from_xml_path("../robots/leg_v1/leg_v1.xml")
        model.opt.timestep = self.sim_dt

        data = mujoco.MjData(model)
        # set initial positions
        data.qpos[-self.num_actions:] = self.default_qpos[self.isaac2mujoco_indices]
        data.ctrl = self.default_qpos

        viewer = mujoco.viewer.launch_passive(model, data)

        count_lowlevel = 0
        for _ in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Simulating..."):
            step_start = time.time()

            q, dq, linear_v, angular_v, gravity_vec = self.get_obs(data)
            # first 7 elements are the free joints
            q = q[-self.num_actions:]
            # first 6 elements are the free joints
            dq = dq[-self.num_actions:]

            if count_lowlevel % self.sim_decimation == 0:
                obs = np.zeros([1, self.num_single_obs], dtype=np.float32)

                # imu
                obs[0, 0:3] = linear_v
                obs[0, 3:6] = angular_v
                obs[0, 6:9] = gravity_vec
                # command
                obs[0, 9] = vx
                obs[0, 10] = vy
                obs[0, 11] = dyaw
                # joint state
                obs[0, 12:24] = q[self.mujoco2isaac_indices] - self.default_qpos
                obs[0, 24:36] = dq[self.mujoco2isaac_indices]
                obs[0, 36:48] = self.action
                # gait phase
                # 1: stance, 0: swing
                phase = math.sin(2 * math.pi * count_lowlevel * self.sim_dt  / 1.0)
                obs[0, 48] = phase >= 0
                obs[0, 49] = phase < 0
                # double foot support phase
                if abs(phase) < 0.1:
                    obs[0, 48] = 1
                    obs[0, 49] = 1

                # append new obs and discard the oldest one
                self.obs_deq.append(obs)
                self.obs_deq.popleft()

                for i in range(self.num_frame_stack):
                    self.policy_input[0, i * self.num_single_obs : (i + 1) * self.num_single_obs] = self.obs_deq[i][0, :]
                self.action = self.policy(torch.tensor(self.policy_input))[0].detach().numpy()

                self.target_q = self.action.copy()

            data.ctrl = self.target_q * 0.5 + self.default_qpos
            mujoco.mj_step(model, data)

            viewer.sync()
            count_lowlevel += 1

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        viewer.close()

if __name__ == '__main__':
    BipedalWalkControl().play()
