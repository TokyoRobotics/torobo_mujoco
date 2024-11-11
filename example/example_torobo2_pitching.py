#! /usr/bin/env python
# -*- coding: utf-8 -*-
import mujoco
import mujoco.viewer
import time
import numpy as np
import yaml

# optimized parameters
best_traj_times = {
    "torso/joint_1": {
        "traj_start_time": 0.6993413596540958,
        "traj_end_time": 2.480151272802761
    },
    "torso/joint_2": {
        "traj_start_time": 0.9735932089311297,
        "traj_end_time": 2.020937722234514
    },
    "torso/joint_3": {
        "traj_start_time": 0.824714567249242,
        "traj_end_time": 1.797312886027256
    },
    "right_arm/joint_1": {
        "traj_start_time": 0.8873775506065673,
        "traj_end_time": 1.718165392622821
    },
    "right_arm/joint_2": {
        "traj_start_time": 0.6724670989090791,
        "traj_end_time": 1.687491756717675
    },
    "right_arm/joint_3": {
        "traj_start_time": 0.7820299179815685,
        "traj_end_time": 1.784713234984677
    },
    "right_arm/joint_4": {
        "traj_start_time": 1.115492141463552,
        "traj_end_time": 1.530133297167985
    },
    "right_arm/joint_5": {
        "traj_start_time": 0.5968317565184726,
        "traj_end_time": 2.047444837664937
    },
    "right_arm/joint_6": {
        "traj_start_time": 0.4726989151309045,
        "traj_end_time": 2.386450661187625
    },
    "right_arm/joint_7": {
        "traj_start_time": 0.5103433301095578,
        "traj_end_time": 2.302622318455541
    }
}

class ThrowTrajectoryPlanner:
    def __init__(self, total_duration, total_steps):
        # trajectory parameters
        self._total_duration = total_duration
        self._total_steps = total_steps
        self._throw_animation_scale = 0.93
        self._preliminary_action_time = 0.5

        # joint parameters
        self._head_joint_names = ["head/joint_" + str(i+1) for i in range(3)]
        self._left_arm_joint_names = ["left_arm/joint_" + str(i+1) for i in range(7)]
        self._throw_joint_names = ["torso/joint_" + str(i+1) for i in range(3)] + \
                                  ["right_arm/joint_" + str(i+1) for i in range(7)]

        # joint positions
        self._init_joint_pos = [0, 0, 1.57079633, 0, -0.26179939, 0, 1.57079633, 1.57079633, 0,  0]
        self._head_init_joint_pos = [0, 0, 0]
        self._left_arm_init_joint_pos = [0, -1.57, 0, 0, 0, 0, 0]
        self._left_arm_end_joint_pos = [-0.57595865, -0.52359878, 0, 2.05948852, 0, 0,  0]
        # right_arm/joint_5, 6, 7 are fixed to stabilize release timing
        self._throw_init_joint_pos = [
            0.0, 0.6, -0.216, -0.18, -1.86, 1.6, 0.0, 0.523599, -0.3, 0.0]
        self._throw_end_joint_pos = [
            0.0, 0.6, 2.5, 2.024, -0.244346, -1.2, 1.1, 0.523599, -0.3, 0.0]

    def calc_traj_pos_vel_acc(self, t, ts, tf, init_pos, end_pos):
        # calculate position, velocity, and acceleration from fifth order polynomial which passes through
        # initial and end position with zero velocity and acceleration
        duration = tf - ts
        diff_pos = end_pos - init_pos
        pos = 6 * diff_pos * (t - ts)**5 / duration**5 - 15 * diff_pos * (
            t - ts)**4 / duration**4 + 10 * diff_pos * (t - ts)**3 / duration**3 + init_pos
        vel = 30 * diff_pos * (t - ts)**4 / duration**5 - 60 * diff_pos * (
            t - ts)**3 / duration**4 + 30 * diff_pos * (t - ts)**2 / duration**3
        acc = 120 * diff_pos * (t - ts)**3 / duration**5 - 180 * diff_pos * (
            t - ts)**2 / duration**4 + 60 * diff_pos * (t - ts) / duration**3
        return pos, vel, acc

    def create_traj_start_end_time(self, best_traj_times):
        # start and end time for each joint trajectory is scaled by throw_animation_scale
        self._ts_dict = dict()
        self._tf_dict = dict()
        min_traj_start_time = min([best_traj_times[joint]["traj_start_time"] for joint in self._throw_joint_names])
        for joint in self._throw_joint_names:
            traj_start_time = min_traj_start_time + (best_traj_times[joint]["traj_start_time"] - min_traj_start_time) / self._throw_animation_scale
            self._ts_dict.update({joint : self._preliminary_action_time + traj_start_time})
            self._tf_dict.update({joint : self._preliminary_action_time + traj_start_time + \
                                    (best_traj_times[joint]["traj_end_time"] - best_traj_times[joint]["traj_start_time"]) / self._throw_animation_scale})

    def create_head_trajectory(self):
        # head/joint_1 moves to match torso/joint_3
        ts = self._ts_dict["torso/joint_3"]
        tf = self._tf_dict["torso/joint_3"]
        torso_j3_init_joint_pos = self._throw_init_joint_pos[2]
        torso_j3_end_joint_pos = self._throw_end_joint_pos[2]

        head_positions = self._head_init_joint_pos
        head_velocities = [0.0 for _ in range(len(self._head_init_joint_pos))]
        head_accelerations = [0.0 for _ in range(len(self._head_init_joint_pos))]

        head_positions_list = []

        for i in range(self._total_steps):
            t = (i + 1) * self._total_duration / float(self._total_steps)

            if t < ts:
                head_positions[0], head_velocities[0], head_accelerations[0] = self.calc_traj_pos_vel_acc(
                    t, 0, ts, 0, -1 * torso_j3_init_joint_pos + np.pi/2)
            if t >= ts and t <= tf:
                head_positions[0], head_velocities[0], head_accelerations[0] = self.calc_traj_pos_vel_acc(
                    t, ts, tf, -1 * torso_j3_init_joint_pos + np.pi/2, -1 * torso_j3_end_joint_pos + np.pi/2)

            head_positions_list.append(head_positions.copy())
        return head_positions_list

    def create_left_arm_trajectory(self):
        # left arm moves to match torso/joint_3
        ts = self._ts_dict["torso/joint_3"]
        tf = self._tf_dict["torso/joint_3"]

        left_arm_positions = self._left_arm_init_joint_pos.copy()
        left_arm_velocities = [0.0 for _ in range(len(self._left_arm_init_joint_pos))]
        left_arm_accelerations = [0.0 for _ in range(len(self._left_arm_init_joint_pos))]

        left_arm_positions_list = []

        for i in range(self._total_steps):
            t = (i + 1) * self._total_duration / float(self._total_steps)

            for j in range(len(self._left_arm_joint_names)):
                # preliminary motion before throwing
                if t < ts:
                    left_arm_positions[j], left_arm_velocities[j], left_arm_accelerations[j] = self.calc_traj_pos_vel_acc(
                        t, 0, ts, 0, self._left_arm_init_joint_pos[j])

                # motion while throwing
                if t >= ts and t <= tf:
                    left_arm_positions[j], left_arm_velocities[j], left_arm_accelerations[j] = self.calc_traj_pos_vel_acc(
                        t, ts, tf, self._left_arm_init_joint_pos[j], self._left_arm_end_joint_pos[j])

            left_arm_positions_list.append(left_arm_positions.copy())
        return left_arm_positions_list

    def create_throw_trajectory(self):
        throw_positions = self._throw_init_joint_pos.copy()
        throw_velocities = [0.0 for _ in range(len(self._throw_joint_names))]
        throw_accelerations = [0.0 for _ in range(len(self._throw_joint_names))]

        throw_positions_list = []

        for i in range(self._total_steps):
            t = (i + 1) * self._total_duration / float(self._total_steps)

            for j in range(len(self._throw_joint_names)):
                joint_name = self._throw_joint_names[j]
                traj_start_time = self._ts_dict[joint_name]
                traj_end_time = self._tf_dict[joint_name]
                # preliminary motion before throwing
                if t < traj_start_time:
                    throw_positions[j], throw_velocities[j], throw_accelerations[j] = self.calc_traj_pos_vel_acc(
                        t, 0, traj_start_time, self._init_joint_pos[j], self._throw_init_joint_pos[j])

                # throwing motion
                if t > traj_start_time and t < traj_end_time:
                    throw_positions[j], throw_velocities[j], throw_accelerations[j] = self.calc_traj_pos_vel_acc(
                        t, traj_start_time,traj_end_time, self._throw_init_joint_pos[j], self._throw_end_joint_pos[j])

            throw_positions_list.append(throw_positions.copy())
        return throw_positions_list


def main():
    base_joint_num = 3
    total_duration = 4
    total_steps = 2500
    timestep = total_duration / total_steps

    throw_trajectory_planner = ThrowTrajectoryPlanner(total_duration, total_steps)
    throw_trajectory_planner.create_traj_start_end_time(best_traj_times)

    throw_jt = throw_trajectory_planner.create_throw_trajectory()
    head_jt = throw_trajectory_planner.create_head_trajectory()
    left_arm_jt = throw_trajectory_planner.create_left_arm_trajectory()

    torso_init_joint_pos = [0, 0, 1.57079633]
    right_arm_init_joint_pos =  [0, -0.26179939, 0, 1.57079633, 1.57079633, 0,  0]
    right_gripper_init_joint_pos = 0.031

    m = mujoco.MjModel.from_xml_path("../robots/torobo2/torobo2_right_gripper_pitching.xml")
    d = mujoco.MjData(m)

    # set torso initial position
    d.qpos[10:13] = torso_init_joint_pos
    d.ctrl[3:6] = torso_init_joint_pos
    # set right arm initial position
    d.qpos[20:27] = right_arm_init_joint_pos
    d.ctrl[13:20] = right_arm_init_joint_pos
    # set right gripper initial position
    d.qpos[27] = right_gripper_init_joint_pos
    d.ctrl[20] = right_gripper_init_joint_pos

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        while viewer.is_running():
            step_start = time.time()

            # close gripper to grasp ball
            if time.time() - start > 0.5 and time.time() - start < 1:
                d.ctrl[20] = 0.022

            # execute throwing motion
            if time.time() - start >= 2:
                if len(throw_jt) == 0:
                    continue
                head_pos = head_jt.pop(0)
                d.ctrl[base_joint_num+18:base_joint_num+21] = head_pos

                left_arm_pos = left_arm_jt.pop(0)
                d.ctrl[base_joint_num+3:base_joint_num+10] = left_arm_pos

                throw_pos = throw_jt.pop(0)
                d.ctrl[base_joint_num:base_joint_num+3] = throw_pos[:3] # torso
                d.ctrl[base_joint_num+10:base_joint_num+17] = throw_pos[3:] # right arm

            # open gripper to release ball
            if time.time() - start >= 4.05:
                d.ctrl[20] = 0.031

            mujoco.mj_step(m, d)
            viewer.sync()

            time_until_next_step = timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == '__main__':
    main()
