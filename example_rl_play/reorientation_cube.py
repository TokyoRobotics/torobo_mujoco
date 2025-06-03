import math
import numpy as np
import mujoco, mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import time
from rl_play_base import RLPlayBase

class ReorientationCubeControl(RLPlayBase):

    def __init__(self):
        super().__init__(policy_path="policies/reorientation_cube_policy.pt",
                         use_cpu=True,
                         num_dofs=16,
                         num_actions=12,
                         num_single_obs=46,
                         num_frame_stack=2,
                         sim_duration=60,
                         sim_dt=0.01,
                         sim_decimation=4,
                         default_qpos=np.zeros(16, dtype=np.double),
                         isaac_joint_names_order = ['hand_first_finger_base_joint', 'hand_second_finger_base_joint', 'hand_third_finger_base_joint',
                                                    'hand_thumb_joint_1', 'hand_first_finger_joint_1', 'hand_second_finger_joint_1',
                                                    'hand_thumb_joint_2', 'hand_first_finger_joint_2', 'hand_third_finger_joint_1',
                                                    'hand_second_finger_joint_2', 'hand_thumb_joint_3', 'hand_first_finger_joint_3',
                                                    'hand_third_finger_joint_2', 'hand_second_finger_joint_3', 'hand_thumb_joint_4', 'hand_third_finger_joint_3'],
                         mujoco_joint_names_order = ['hand_first_finger_base_joint', 'hand_first_finger_joint_1', 'hand_first_finger_joint_2', 'hand_first_finger_joint_3',
                                                    'hand_second_finger_base_joint', 'hand_second_finger_joint_1', 'hand_second_finger_joint_2', 'hand_second_finger_joint_3',
                                                    'hand_third_finger_base_joint', 'hand_third_finger_joint_1', 'hand_third_finger_joint_2', 'hand_third_finger_joint_3',
                                                    'hand_thumb_joint_1', 'hand_thumb_joint_2', 'hand_thumb_joint_3', 'hand_thumb_joint_4'])
        self.success_tolerance = 0.3 # radians

    def scale(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def play(self):
        model = mujoco.MjModel.from_xml_path("../robots/hand_v4/hand_v4_with_cube.xml")
        model.opt.timestep = self.sim_dt

        data = mujoco.MjData(model)

        viewer = mujoco.viewer.launch_passive(model, data)

        lower_pos_limit_dict = {'hand_first_finger_base_joint': -0.10472, 'hand_second_finger_base_joint': -0.10472, 'hand_third_finger_base_joint': -0.10472,
                                'hand_thumb_joint_1': 0, 'hand_first_finger_joint_1': 0, 'hand_second_finger_joint_1': 0,
                                'hand_thumb_joint_2': -0.13962, 'hand_first_finger_joint_2': -0.13962, 'hand_third_finger_joint_1': 0,
                                'hand_second_finger_joint_2': -0.13962, 'hand_thumb_joint_3': -0.314159, 'hand_first_finger_joint_3': -0.13962,
                                'hand_third_finger_joint_2': -0.13962, 'hand_second_finger_joint_3': -0.13962, 'hand_thumb_joint_4': -0.314159, 'hand_third_finger_joint_3': -0.13962}
        upper_pos_limit_dict = {'hand_first_finger_base_joint': 0.10472, 'hand_second_finger_base_joint': 0.10472, 'hand_third_finger_base_joint': 0.10472,
                                'hand_thumb_joint_1': 1.5009, 'hand_first_finger_joint_1': 1.5009, 'hand_second_finger_joint_1': 1.5009,
                                'hand_thumb_joint_2': 0.9250, 'hand_first_finger_joint_2': 1.8849, 'hand_third_finger_joint_1': 1.5009,
                                'hand_second_finger_joint_2': 1.8849, 'hand_thumb_joint_3': 1.53588, 'hand_first_finger_joint_3': 1.8849,
                                'hand_third_finger_joint_2': 1.8849, 'hand_second_finger_joint_3': 1.8849, 'hand_thumb_joint_4': 1.53588, 'hand_third_finger_joint_3': 1.8849}

        soft_joint_pos_limit_factor = 0.9

        lower_pos_limit = np.array(list(lower_pos_limit_dict.values()))
        upper_pos_limit = np.array(list(upper_pos_limit_dict.values()))
        mean_pos_limit = 0.5 * (lower_pos_limit + upper_pos_limit)
        lower_pos_limit = mean_pos_limit - (mean_pos_limit - lower_pos_limit) * soft_joint_pos_limit_factor
        upper_pos_limit = mean_pos_limit + (upper_pos_limit - mean_pos_limit) * soft_joint_pos_limit_factor

        # must be controlled in the order of the following indices
        actuated_joint_names = ['hand_first_finger_base_joint', 'hand_second_finger_base_joint',
                                'hand_thumb_joint_1', 'hand_first_finger_joint_1', 'hand_second_finger_joint_1',
                                'hand_thumb_joint_2', 'hand_first_finger_joint_2', 'hand_third_finger_joint_1',
                                'hand_second_finger_joint_2', 'hand_thumb_joint_3', 'hand_third_finger_joint_2', 'hand_thumb_joint_4']
        actuated_lower_pos_limit = np.array([lower_pos_limit_dict[joint_name] for joint_name in actuated_joint_names])
        actuated_upper_pos_limit = np.array([upper_pos_limit_dict[joint_name] for joint_name in actuated_joint_names])
        mean_actuated_pos_limit = 0.5 * (actuated_lower_pos_limit + actuated_upper_pos_limit)
        actuated_lower_pos_limit = mean_actuated_pos_limit - (mean_actuated_pos_limit - actuated_lower_pos_limit) * soft_joint_pos_limit_factor
        actuated_upper_pos_limit = mean_actuated_pos_limit + (actuated_upper_pos_limit - mean_actuated_pos_limit) * soft_joint_pos_limit_factor

        prev_target_q = np.zeros((self.num_actions), dtype=np.double)
        prev_action = np.zeros((self.num_actions), dtype=np.double)
        prev_joint_pos = np.zeros((self.num_dofs), dtype=np.double)
        prev_object_pos = np.zeros((3), dtype=np.double)
        prev_object_quat = np.zeros((4), dtype=np.double)
        prev_target_pos = np.zeros((7), dtype=np.double)
        prev_target_quat_diff = np.zeros((4), dtype=np.double)
        prev_last_processed_action = np.zeros((self.num_actions), dtype=np.double)

        target_quat = [0, 0, 0, 1] # x, y, z, w

        count_lowlevel = 0
        make_quat_unique = False
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
                joint_pos = self.scale(q[self.mujoco2isaac_indices], lower_pos_limit, upper_pos_limit)
                last_processed_action = prev_target_q.copy()
                target_quat_diff = quat_diff[[3, 0, 1, 2]]
                object_pos = qpos[4:7]
                object_quat = qpos[7:11]
                target_pos = np.array([0.13, -0.02, 0.49])
                if make_quat_unique:
                    if target_quat_diff[0] < 0:
                        target_quat_diff *= -1
                    if object_quat[0] < 0:
                        object_quat *= -1
                    if target_quat[3] < 0:
                        target_quat *= -1
                target_pose = np.concatenate([target_pos, target_quat[[3, 0, 1, 2]]])


                self.policy_input[0, :] = np.concatenate([prev_joint_pos, joint_pos,
                                                          prev_object_pos, object_pos,
                                                          prev_object_quat, object_quat,
                                                          prev_target_pos, target_pose,
                                                          prev_target_quat_diff, target_quat_diff,
                                                          prev_last_processed_action, last_processed_action])
                self.action = self.policy(torch.tensor(self.policy_input))[0].detach().numpy()
                self.action = np.clip(self.action, -1, 1)
                self.target_q = 0.1 * self.action + prev_target_q

                self.target_q = np.clip(self.target_q, actuated_lower_pos_limit, actuated_upper_pos_limit)

                prev_action[:] = self.action.copy()
                prev_target_q[:] = self.target_q.copy()
                prev_joint_pos[:] = joint_pos.copy()
                prev_object_pos[:] = object_pos.copy()
                prev_object_quat[:] = object_quat.copy()
                prev_target_pos[:] = target_pose.copy()
                prev_target_quat_diff[:] = target_quat_diff.copy()
                prev_last_processed_action[:] = last_processed_action.copy()

            data.ctrl = self.target_q

            mujoco.mj_step(model, data)

            viewer.sync()
            count_lowlevel += 1

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        viewer.close()


if __name__ == '__main__':
    ReorientationCubeControl().play()
