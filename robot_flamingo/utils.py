import logging

import numpy as np
# from pytorch3d.transforms import (
#     euler_angles_to_matrix,
#     matrix_to_euler_angles,
#     matrix_to_quaternion,
#     quaternion_to_matrix,
# )
import torch
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    import torch.nn.functional as F
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)



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