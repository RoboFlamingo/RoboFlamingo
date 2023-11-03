import logging

import numpy as np
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
import torch
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def world_to_tcp_frame(action, robot_obs):
    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.view(b, s*f, -1)
            robot_obs = robot_obs.view(b, s*f, -1)
        b, s, _ = action.shape
        world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ").float().view(-1, 3, 3)
        tcp_T_world = torch.inverse(world_T_tcp)
        pos_w_rel = action[..., :3].view(-1, 3, 1)
        pos_tcp_rel = tcp_T_world @ pos_w_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_w_rel = action[..., 3:6] * 0.01
        world_T_tcp_new = (
            euler_angles_to_matrix(robot_obs[..., 3:6] + orn_w_rel, convention="XYZ").float().view(-1, 3, 3)
        )
        tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
        orn_tcp_rel = matrix_to_euler_angles(tcp_new_T_tcp_old, convention="XYZ").float()
        orn_tcp_rel = torch.where(orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel)
        orn_tcp_rel = torch.where(orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel)
        # upscaling again
        orn_tcp_rel *= 100
        action_tcp = torch.cat([pos_tcp_rel.view(b, s, -1), orn_tcp_rel.view(b, s, -1), action[..., -1:]], dim=-1)
        if flag:
            action_tcp = action_tcp.view(b, s, -1, action_tcp.shape[-1])
        assert not torch.any(action_tcp.isnan())
    return action_tcp


def tcp_to_world_frame(action, robot_obs):
    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.view(b, s*f, -1)
            robot_obs = robot_obs.view(b, s*f, -1)
        b, s, _ = action.shape
        world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ").float().view(-1, 3, 3)
        pos_tcp_rel = action[..., :3].view(-1, 3, 1)
        pos_w_rel = world_T_tcp @ pos_tcp_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_tcp_rel = action[..., 3:6] * 0.01
        tcp_new_T_tcp_old = euler_angles_to_matrix(orn_tcp_rel, convention="XYZ").float().view(-1, 3, 3)
        world_T_tcp_new = world_T_tcp @ torch.inverse(tcp_new_T_tcp_old)

        orn_w_new = matrix_to_euler_angles(world_T_tcp_new, convention="XYZ").float()
        if torch.any(orn_w_new.isnan()):
            logger.warning("NaN value in euler angles.")
            orn_w_new = matrix_to_euler_angles(
                quaternion_to_matrix(matrix_to_quaternion(world_T_tcp_new)), convention="XYZ"
            ).float()
        orn_w_rel = orn_w_new - robot_obs[..., 3:6].view(-1, 3)
        orn_w_rel = torch.where(orn_w_rel < -np.pi, orn_w_rel + 2 * np.pi, orn_w_rel)
        orn_w_rel = torch.where(orn_w_rel > np.pi, orn_w_rel - 2 * np.pi, orn_w_rel)
        # upscaling again
        orn_w_rel *= 100
        action_w = torch.cat([pos_w_rel.view(b, s, -1), orn_w_rel.view(b, s, -1), action[..., -1:]], dim=-1)
        if flag:
            action_w = action_w.view(b, s, -1, action_w.shape[-1])
        assert not torch.any(action_w.isnan())
    return action_w

if __name__ == "__main__":
    action = torch.randn((4, 5, 3, 7))
    robot_obs = torch.randn((4, 5, 3, 7))
    print(world_to_tcp_frame(action, robot_obs))