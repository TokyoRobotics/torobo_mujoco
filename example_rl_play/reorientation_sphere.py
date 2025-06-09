import math
import numpy as np
import mujoco, mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import time
from rl_play_base import RLPlayBase

class ReorientationSphereControl(RLPlayBase):

    def __init__(self):
        super().__init__(policy_path="policies/reorientation_sphere_policy.pt",
                         use_cpu=True,
                         num_dofs=16,
                         num_actions=12,
                         num_single_obs=32,
                         num_frame_stack=4,
                         sim_duration=60,
                         sim_dt=0.0083,
                         sim_decimation=4,
                         default_qpos=np.array([0, 0.01, 0.01, 1.42, 0, 0, 0.93, 0.8, 1.5, 0.8, -0.3, 0.46, 1.1, 0.46, -0.3, 0.632]),
                         soft_joint_pos_limit_factor = 1.0,
                         lower_pos_limit = [-0.10472, -0.10472, -0.10472,
                                            0, 0, 0,
                                            -0.13962, -0.13962, 0,
                                            -0.13962, -0.314159, -0.13962,
                                            -0.13962, -0.13962, -0.314159, -0.13962],
                         upper_pos_limit = [0.10472, 0.10472, 0.10472,
                                            1.5009, 1.5009, 1.5009,
                                            0.9250, 1.8849, 1.5009,
                                            1.8849, 1.53588, 1.8849,
                                            1.8849, 1.8849, 1.53588, 1.8849],
                         isaac_joint_names_order = ['hand_first_finger_base_joint', 'hand_second_finger_base_joint', 'hand_third_finger_base_joint',
                                                    'hand_thumb_joint_1', 'hand_first_finger_joint_1', 'hand_second_finger_joint_1',
                                                    'hand_thumb_joint_2', 'hand_first_finger_joint_2', 'hand_third_finger_joint_1',
                                                    'hand_second_finger_joint_2', 'hand_thumb_joint_3', 'hand_first_finger_joint_3',
                                                    'hand_third_finger_joint_2', 'hand_second_finger_joint_3', 'hand_thumb_joint_4', 'hand_third_finger_joint_3'],
                         mujoco_joint_names_order = ['hand_first_finger_base_joint', 'hand_first_finger_joint_1', 'hand_first_finger_joint_2', 'hand_first_finger_joint_3',
                                                    'hand_second_finger_base_joint', 'hand_second_finger_joint_1', 'hand_second_finger_joint_2', 'hand_second_finger_joint_3',
                                                    'hand_third_finger_base_joint', 'hand_third_finger_joint_1', 'hand_third_finger_joint_2', 'hand_third_finger_joint_3',
                                                    'hand_thumb_joint_1', 'hand_thumb_joint_2', 'hand_thumb_joint_3', 'hand_thumb_joint_4'],
                        actuated_joint_names_order = ['hand_first_finger_base_joint', 'hand_second_finger_base_joint',
                                                    'hand_thumb_joint_1', 'hand_first_finger_joint_1', 'hand_second_finger_joint_1',
                                                    'hand_thumb_joint_2', 'hand_first_finger_joint_2', 'hand_third_finger_joint_1',
                                                    'hand_second_finger_joint_2', 'hand_thumb_joint_3', 'hand_third_finger_joint_2', 'hand_thumb_joint_4'])
        self.success_tolerance = 0.2 # radians

    def scale(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def play(self):
        model = mujoco.MjModel.from_xml_path("../robots/hand_v4/hand_v4_with_sphere.xml")
        model.opt.timestep = self.sim_dt

        data = mujoco.MjData(model)

        viewer = mujoco.viewer.launch_passive(model, data)

        prev_target_q = np.zeros((self.num_actions), dtype=np.double)
        prev_action = np.zeros((self.num_actions), dtype=np.double)

        target_quat = [0, 0, 0, 1] # x, y, z, w

        count_lowlevel = 0
        for _ in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Simulating..."):
            step_start = time.time()
            # obtain an observation
            qpos = data.qpos.astype(np.double)

            # first 4 elements are the target object orientation (quaternion)
            # next 3 elements are the object position
            # next 4 elements are the object orientation (quaternion)
            q = qpos[-self.num_dofs:]
            obj_quat = qpos[7:11] # w, x, y, z

            # calculate the quaternion difference from the object to the target
            quat_diff = (R.from_quat(obj_quat[[1, 2, 3, 0]]) * R.from_quat(target_quat).inv()).as_quat()
            # convert quaternion difference to rotation angle difference
            rot_dist = 2 * np.arcsin(min(np.linalg.norm(quat_diff[0:3]), 1.0))
            # if the rotation is close to the target, update the target randomly
            if rot_dist <= self.success_tolerance:
                print("target updated")
                target_quat = (R.random() * R.from_quat(target_quat)).as_quat()
                data.qpos[0:4] = target_quat[[3, 0, 1, 2]] # w, x, y, z

            if count_lowlevel % self.sim_decimation == 0:
                obs = np.zeros([1, self.num_single_obs], dtype=np.float32)

                obs[0, 0:16] = self.scale(q[self.mujoco2isaac_indices], self.lower_pos_limit, self.upper_pos_limit)
                obs[0, 16:28] = self.target_q
                obs[0, 28:32] = quat_diff[[3, 0, 1, 2]] # w, x, y, z

                self.obs_deq.append(obs)
                self.obs_deq.popleft()

                for i in range(self.num_frame_stack):
                    self.policy_input[0, i * self.num_single_obs : (i + 1) * self.num_single_obs] = self.obs_deq[-(i + 1)][0, :]
                self.action = self.policy(torch.tensor(self.policy_input))[0].detach().numpy()

                ema_actions = 0.8 * self.action + 0.2 * prev_action
                self.target_q = prev_target_q + ema_actions

                self.target_q = np.clip(self.target_q, self.actuated_lower_pos_limit, self.actuated_upper_pos_limit)

                prev_action[:] = self.action.copy()
                prev_target_q[:] = self.target_q.copy()


            data.ctrl = self.target_q

            mujoco.mj_step(model, data)

            viewer.sync()
            count_lowlevel += 1

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        viewer.close()


if __name__ == '__main__':
    ReorientationSphereControl().play()
