import numpy as np
from collections import deque
import torch
import dataclasses
from typing import Optional

@dataclasses.dataclass
class RLPlayBase:
    policy_path: str
    use_cpu: bool
    num_dofs: int
    num_actions: int
    num_single_obs: int
    num_frame_stack: int
    sim_duration: float
    sim_dt: float
    sim_decimation: int
    default_qpos: np.ndarray
    isaac_joint_names_order: list
    mujoco_joint_names_order: list
    # optinal arguments
    soft_joint_pos_limit_factor: Optional[float] = None
    lower_pos_limit: Optional[list] = None
    upper_pos_limit: Optional[list] = None
    actuated_joint_names_order: Optional[list] = None

    def __post_init__(self):
        if self.use_cpu:
            self.policy = torch.jit.load(self.policy_path, map_location=torch.device('cpu'))
        else:
            self.policy = torch.jit.load(self.policy_path)
        self.policy_input = np.zeros([1, self.num_single_obs * self.num_frame_stack], dtype=np.float32)
        self.obs_deq = deque()
        for _ in range(self.num_frame_stack):
            self.obs_deq.append(np.zeros([1, self.num_single_obs], dtype=np.double))
        self.target_q = np.zeros((self.num_actions), dtype=np.double)
        self.action = np.zeros((self.num_actions), dtype=np.double)
        # reorder the observation in mujoco to match the order of isaac joint names
        self.mujoco2isaac_indices = [self.mujoco_joint_names_order.index(joint_name) for joint_name in self.isaac_joint_names_order]
        self.isaac2mujoco_indices = [self.isaac_joint_names_order.index(joint_name) for joint_name in self.mujoco_joint_names_order]

        if self.soft_joint_pos_limit_factor is not None:
            lower_pos_limit = np.array(self.lower_pos_limit)
            upper_pos_limit = np.array(self.upper_pos_limit)
            mean_pos_limit = 0.5 * (lower_pos_limit + upper_pos_limit)
            self.lower_pos_limit = mean_pos_limit - (mean_pos_limit - lower_pos_limit) * self.soft_joint_pos_limit_factor
            self.upper_pos_limit = mean_pos_limit + (upper_pos_limit - mean_pos_limit) * self.soft_joint_pos_limit_factor

            actuated_lower_pos_limit = np.array([self.lower_pos_limit[self.isaac_joint_names_order.index(joint_name)] for joint_name in self.actuated_joint_names_order])
            actuated_upper_pos_limit = np.array([self.upper_pos_limit[self.isaac_joint_names_order.index(joint_name)] for joint_name in self.actuated_joint_names_order])
            actuated_mean_pos_limit = 0.5 * (actuated_lower_pos_limit + actuated_upper_pos_limit)
            self.actuated_lower_pos_limit = actuated_mean_pos_limit - (actuated_mean_pos_limit - actuated_lower_pos_limit) * self.soft_joint_pos_limit_factor
            self.actuated_upper_pos_limit = actuated_mean_pos_limit + (actuated_upper_pos_limit - actuated_mean_pos_limit) * self.soft_joint_pos_limit_factor