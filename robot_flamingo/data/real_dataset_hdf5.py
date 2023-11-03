"""
Code for loading real data.
This dataset contains language + video + action.

Return: text, image sequence, action sequence, timestep, attention_mask
"""
import os
import json
import math
from turtle import forward
import h5py, random
from tqdm import tqdm

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, dataset
import torchvision.transforms as T

HAND_RGB_KEY = "rgb_0"
STATIC_RGB_KEY = "rgb_1"
GRIPPER_POS_CHANGE_THRESHOLD = 5
GRIPPER_THRESHOLD = 30
GRIPPER_OPEN = 1
GRIPPER_CLOSE = -1
VIVE_GRIPPER_OPEN = 2
VIVE_GRIPPER_CLOSE = 3

# Mode 1
OFFSET_EULER_Z = np.pi / 2
OFFSET_POS = [0.15, -0.75, -0.9]

EXCLUDING_VIDEOS = []


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert(isRotm(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    if x < 0:
        x += (2 * np.pi)
    return np.array([x, y, z])


def quat2rotm(quat):
    """Quaternion to rotation matrix."""
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]
    s = w*w + x*x + y*y + z*z
    rotm = np.array([[1-2*(y*y+z*z)/s, 2*(x*y-z*w)/s,   2*(x*z+y*w)/s  ],
                     [2*(x*y+z*w)/s,   1-2*(x*x+z*z)/s, 2*(y*z-x*w)/s  ],
                     [2*(x*z-y*w)/s,   2*(y*z+x*w)/s,   1-2*(x*x+y*y)/s]])
    return rotm


def get_mat_log(R):
  """Get the log(R) of the rotation matrix R.
  
  Args:
    R (3x3 numpy array): rotation matrix
  Returns:
    w (3, numpy array): log(R)
  """
  theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
  w_hat = (R - R.T) * theta / (2 * np.sin(theta) + 1e-10)  # Skew symmetric matrix
  w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])  # [w1, w2, w3]

  return w


def get_binary_gripper_state_from_gripper_pos_vive_cmd(gripper_pos, vive_cmd):
    """Get binary gripper state from gripper pos & vive cmd."""
    assert gripper_pos.shape == vive_cmd.shape

    # find the changing frame
    n_frames = vive_cmd.shape[0]
    gripper_change = np.zeros(n_frames)
    gripper_change[1:] = (vive_cmd[1:] != vive_cmd[:-1]) # first frame is open
    gripper_close = np.logical_and((gripper_change == 1), (vive_cmd == VIVE_GRIPPER_CLOSE))
    gripper_open = np.logical_and((gripper_change == 1), (vive_cmd == VIVE_GRIPPER_OPEN))
    gripper_close_idx = np.where(gripper_close)
    gripper_close_idx = gripper_close_idx[0]
    gripper_open_idx = np.where(gripper_open)
    gripper_open_idx = gripper_open_idx[0]
    
    # a trajectory can only has a open or close
    assert (gripper_close_idx.shape[0] + gripper_open_idx.shape[0]) == 1
    
    # Open trajectory
    if gripper_open_idx.shape[0] > 0:
        gripper_change_start_idx = gripper_open_idx[0]
    elif gripper_close_idx.shape[0] > 0:
        gripper_change_start_idx = gripper_close_idx[0]
    else:
        raise NotImplementedError
    
    # find the position of the gripper when it has not been changed
    unchanged_gripper_pos_idx = int(0.5 * gripper_change_start_idx)
    unchanged_gripper_pos = gripper_pos[unchanged_gripper_pos_idx]
    unchanged_gripper_vive_cmd = vive_cmd[unchanged_gripper_pos_idx]

    gripper_change_start_idx += 1 # the gripper pos changes a frame behind the vive cmd changes
    for i in range(gripper_change_start_idx, n_frames):
        gripper_change_idx = i
        if np.abs(gripper_pos[gripper_change_idx] - unchanged_gripper_pos) > GRIPPER_POS_CHANGE_THRESHOLD:
            break
    
    if unchanged_gripper_vive_cmd == VIVE_GRIPPER_OPEN:
        gripper_states = np.ones(n_frames) * GRIPPER_OPEN
    elif unchanged_gripper_vive_cmd == VIVE_GRIPPER_CLOSE:
        gripper_states = np.ones(n_frames) * GRIPPER_CLOSE
    else:
        raise NotImplementedError
    
    if gripper_open_idx.shape[0] > 0:
        assert unchanged_gripper_vive_cmd == VIVE_GRIPPER_CLOSE
        gripper_states[gripper_change_idx:] = GRIPPER_OPEN
    elif gripper_close_idx.shape[0] > 0:
        assert unchanged_gripper_vive_cmd == VIVE_GRIPPER_OPEN
        gripper_states[gripper_change_idx:] = GRIPPER_CLOSE
    else:
        raise NotImplementedError
    
    return gripper_states


# TODO: This augmentation does not work now, the image is rotated after augmentation.
# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps_h = 1.0 / (h + 2 * self.pad)
        eps_w = 1.0 / (w + 2 * self.pad)
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w + 2 * self.pad, device=x.device, dtype=x.dtype)[:w]
        arange_h = arange_h.unsqueeze(1).repeat(1, w).unsqueeze(2) # (h, w, 1)
        arange_w = arange_w.unsqueeze(0).repeat(h, 1).unsqueeze(2) # (h, w, 1)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift_h = shift * 2.0 / (h + 2 * self.pad)
        shift_w = shift * 2.0 / (w + 2 * self.pad)

        base_grid[:, :, 0:1] += shift_w
        base_grid[:, :, 1:] += shift_h
        grid = base_grid
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class CubeRandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps_h = 1.0 / (h + 2 * self.pad)
        eps_w = 1.0 / (w + 2 * self.pad)
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w + 2 * self.pad, device=x.device, dtype=x.dtype)[:w]
        arange_h = arange_h.unsqueeze(1).repeat(1, w).unsqueeze(2) # (h, w, 1)
        arange_w = arange_w.unsqueeze(0).repeat(h, 1).unsqueeze(2) # (h, w, 1)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(n, 1, 1, 1)
        shift_h = shift * 2.0 / (h + 2 * self.pad)
        shift_w = shift * 2.0 / (w + 2 * self.pad)

        base_grid[:, :, 0:1] += shift_w
        base_grid[:, :, 1:] += shift_h
        grid = base_grid
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PatchMask(nn.Module):
    def __init__(self, patch_size=16, mask_ratio=0.35):
        super(PatchMask, self).__init__()
        self.patch_size=patch_size
        self.mask_ratio=mask_ratio
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Generate random mask coordinates.
        mask_coords = []
        for i in range(batch_size):
            for j in range(0, height, self.patch_size):
                for k in range(0, width, self.patch_size):
                    if random.random() < self.mask_ratio:
                        mask_coords.append((i, j, k))

        # Mask out the patches.
        masked_x = x.clone()
        for i, j, k in mask_coords:
            masked_x[i, :, j:j + self.patch_size, k:k + self.patch_size] = 0.0
        
        return masked_x


class RealDatasetHDF5(Dataset):
    def __init__(self,
                 data_dir,
                 image_fn,
                 text_fn,
                 seq_len=12,
                 mode='train',
                 action_mode='ee_rel_pose_local',
                 use_data_augmentation=True,
                 text_aug=False):
        """Constructor."""
        super().__init__()
        self.dataset_dir = os.path.join(data_dir, mode)
        self.text_fn = text_fn
        self.image_fn = image_fn
        self.text_aug = text_aug
        with open('enrich_lang_real.json', 'r') as f:
            self.enrich_lang_dict = json.load(f)
        self.seq_len = seq_len
        self.mode = mode
        self.action_mode = action_mode
        self.use_data_augmentation = use_data_augmentation

        if self.action_mode == 'ee_rel_pose':
            self.action_dim = 7 # ee xyz (3) + ee euler (3) + gripper (1)
            self.state_dim = 7
            self.ACTION_POS_SCALE = 50
            self.ACTION_ROT_SCALE = 33
        elif self.action_mode == 'ee_rel_pose_local':
            self.action_dim = 7 # ee xyz (3) + ee euler (3) + gripper (1)
            self.state_dim = 7
            self.ACTION_POS_SCALE = 50
            self.ACTION_ROT_SCALE = 33
        else:
            raise NotImplementedError()
        print(f"ACTION_POS_SCALE: {self.ACTION_POS_SCALE}")
        print(f"ACTION_ROT_SCALE: {self.ACTION_ROT_SCALE}")
        
        # the input to this function is a numpy array
        self.input_size = (224, 224)
        self.clip_mean = (0.485, 0.456, 0.406)
        self.clip_std = (0.229, 0.224, 0.225)

        if self.use_data_augmentation:
            self.static_rgb_preprocess_train = T.Compose([
                T.ColorJitter(
                    brightness=0.05,
                    # contrast=0.05,
                    # hue=0.02
                ),
                # CubeRandomShiftsAug(pad=10), # static rgb (300x400)
                RandomShiftsAug(pad=10), # static rgb (300x400)
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std),
                PatchMask()])
            self.hand_rgb_preprocess_train = T.Compose([
                # CubeRandomShiftsAug(pad=20), # hand rgb (480x640)
                RandomShiftsAug(pad=20), # hand rgb (480x640)
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std),
                PatchMask()])
        else:
            self.static_rgb_preprocess_train = T.Compose([
                T.ColorJitter(
                    brightness=0.05,
                    # contrast=0.05,
                    # hue=0.02
                ),
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            self.hand_rgb_preprocess_train = T.Compose([
                T.ColorJitter(
                    brightness=0.05,
                    # contrast=0.05,
                    # hue=0.02
                ),
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        self.static_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])
        self.hand_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])

        self.offset_rotm = gamma2rotm(OFFSET_EULER_Z)
        self.offset_pos = np.array(OFFSET_POS)
        self.hdf5 = h5py.File(os.path.join(self.dataset_dir, "data.hdf5"))
        self._initialize()
        print(f'{len(self)} trajectories in total')

    def _initialize(self):
        """Generate the sequence index pair."""
        with open(os.path.join(self.dataset_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)
        n_trajs = self.meta["num_trajectories"]
        # n_trajs = 1000
        print(f"number of trajectories: {n_trajs}")
        
        self.seq_tuple = []
        self.robot_states = dict()
        all_texts = []
        for traj_idx in tqdm(range(n_trajs)):
            text = self.meta[str(traj_idx)][0]
            all_texts.append(text)
            n_frames = self.meta[str(traj_idx)][1]
            video_name = self.meta[str(traj_idx)][2]
            hdf5_st = self.meta[str(traj_idx)][3]
            hdf5_ed = self.meta[str(traj_idx)][4]
            assert n_frames == hdf5_ed - hdf5_st
            if (hdf5_ed - hdf5_st) < self.seq_len:
                continue
            if video_name in EXCLUDING_VIDEOS:
                continue

            # load robot status and xform with offset
            traj_robot_status = self.hdf5["robot_status"]["robot_status_0"][hdf5_st:hdf5_ed]
            traj_xyz = traj_robot_status[:, 10:13] # (n, 3)
            traj_xyz = traj_xyz.transpose() # (3, n)
            traj_xyz = (self.offset_rotm @ traj_xyz).transpose() + self.offset_pos
            traj_quat = traj_robot_status[:, 13:17]
            traj_rpy = np.zeros((n_frames, 3))
            for i in range(n_frames):
                traj_rpy[i] = rotm2euler(self.offset_rotm @ quat2rotm(traj_quat[i]))
            traj_state = np.zeros((n_frames, 7)).astype(np.float32)
            traj_state[:, :3] = traj_xyz
            traj_state[:, 3:6] = traj_rpy
            vive_control = self.hdf5["vive_control"]["vive_control_0"][hdf5_st:hdf5_ed]
            vive_gripper_cmd = vive_control[:, 1]
            gripper_pos = traj_robot_status[:, 30]
            gripper_states = get_binary_gripper_state_from_gripper_pos_vive_cmd(gripper_pos, vive_gripper_cmd)
            traj_state[:, -1] = gripper_states
            assert not (traj_idx in self.robot_states)
            self.robot_states[traj_idx] = traj_state

            # create sequence: the last frame will not be in the sequence
            for st in range(0, n_frames - self.seq_len):
                ed = st + self.seq_len
                self.seq_tuple.append([traj_idx, text, st, ed, hdf5_st])
        
        all_texts = list(set(all_texts))
        print(all_texts)
        # exit(0)
    
    def __len__(self):
        return len(self.seq_tuple)

    def __getitem__(self, index):
        curr_tuple = self.seq_tuple[index]
        traj_idx = curr_tuple[0]
        text = curr_tuple[1]
        
        # if ("on the plate" in text) and ("pick" in text):
        #     text = text.replace(" on the plate", "")
        # if ("on the desk" in text) and ("pick" in text):
        #     text = text.replace(" on the desk", "")

        st = curr_tuple[2]
        ed = curr_tuple[3]
        hdf5_st = curr_tuple[4]

        static_rgbs = []
        hand_rgbs = []
        actions = []
        states = []

        tlen = ed - st
        assert tlen == self.seq_len

        for i in range(st, ed):
            # action
            if self.action_mode == 'ee_rel_pose':
                # delta_xyz + detla_rpy + gripper in absolute world coordinates
                # xyz are scaled up by 50 rpy are scaled up by 20 and both are clipped to [-1, 1]
                xyz_action = (self.robot_states[traj_idx][i+1, :3] - self.robot_states[traj_idx][i, :3]) 
                rpy_action = (self.robot_states[traj_idx][i+1, 3:6] - self.robot_states[traj_idx][i, 3:6])
                gripper_action = self.robot_states[traj_idx][i+1, 6]
            elif self.action_mode == 'ee_rel_pose_local':
                # a_trans = rotm_t.T @ (trans_t+1 - trans_t)
                # a_rot = rotm_t.T @ rotm_t+1
                curr_xyz = self.robot_states[traj_idx][i, :3]
                curr_rpy = self.robot_states[traj_idx][i, 3:6]
                curr_rotm = euler2rotm(curr_rpy)
                next_xyz = self.robot_states[traj_idx][i+1, :3]
                next_rpy = self.robot_states[traj_idx][i+1, 3:6]
                next_rotm = euler2rotm(next_rpy)
                xyz_action = np.dot(curr_rotm.T, next_xyz - curr_xyz)
                rel_rotm = curr_rotm.T @ next_rotm
                rpy_action = rotm2euler(rel_rotm)
                for rpy_i in range(len(rpy_action)):
                    while rpy_action[rpy_i] > np.pi:
                        rpy_action[rpy_i] -= (2 * np.pi)
                    while rpy_action[rpy_i] < -np.pi:
                        rpy_action[rpy_i] += (2 * np.pi)
                gripper_action = self.robot_states[traj_idx][i+1, 6]
            else:
                raise NotImplementedError()
            action = np.zeros(7)
            action[:3] = xyz_action * self.ACTION_POS_SCALE
            action[3:6] = rpy_action * self.ACTION_ROT_SCALE
            action[6] = gripper_action
            actions.append(action)
            
            # state
            states.append(self.robot_states[traj_idx][i])

            # static rgb
            static_rgb = self.hdf5["rgb"]["rgb_1"][hdf5_st+i]
            static_rgb = static_rgb[190:700, 250:1050] # mode 1
            static_rgb = Image.fromarray(static_rgb)
            static_rgb = T.ToTensor()(static_rgb.convert("RGB"))
            static_rgbs.append(static_rgb)

            # hand rgb
            hand_rgb = self.hdf5["rgb"]["rgb_0"][hdf5_st+i]
            hand_rgb = Image.fromarray(hand_rgb)
            hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
            hand_rgbs.append(hand_rgb)
        
        # Images
        static_rgbs = torch.stack(static_rgbs, dim=0)
        hand_rgbs = torch.stack(hand_rgbs, dim=0)
        if self.mode == 'train':
            static_rgbs = self.static_rgb_preprocess_train(static_rgbs)
            hand_rgbs = self.hand_rgb_preprocess_train(hand_rgbs)
        else:
            static_rgbs = self.static_rgb_preprocess_val(static_rgbs)
            hand_rgbs = self.hand_rgb_preprocess_val(hand_rgbs)

        # State
        states = np.array(states)
        states = torch.from_numpy(states)

        # Action
        actions = np.array(actions) # (len, act_dim)
        actions = torch.from_numpy(actions)

        # RGB
        _, C, H, W = static_rgbs.shape
        padded_static_rgbs = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        padded_hand_rgbs = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        padded_static_rgbs[:tlen] = static_rgbs
        padded_hand_rgbs[:tlen] = hand_rgbs
        rgb_data = padded_static_rgbs
        hand_rgb_data = padded_hand_rgbs

        # State
        padded_states = torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        padded_states[:tlen] = states
        state_data = padded_states

        # Action
        padded_actions = torch.zeros(self.seq_len, self.action_dim).float() # (len, action_dim)
        padded_actions[:tlen] = actions
        action_data = padded_actions

        # Timestep
        timestep = np.zeros(self.seq_len, dtype=np.int32) # (len)
        timestep[:tlen] = np.arange(st, ed)
        timestep_data = torch.from_numpy(timestep).long()

        # Attention mask (should be all 1 for full dataset)
        attention_mask = np.ones(self.seq_len, dtype=np.int32) # (len)
        attention_mask[tlen:] = 0.0
        assert np.sum(attention_mask) == self.seq_len
        attention_mask_data = torch.from_numpy(attention_mask).long()

        data = dict()
        data['rgb'] = rgb_data # (len, C, H, W)
        data['hand_rgb'] = hand_rgb_data # (len, C, H, W)
        if self.text_aug:
            if text in self.enrich_lang_dict:
                if random.random() > 0.1: # preserve the original text in 0.1 prob
                    text = random.choice(self.enrich_lang_dict[text])
        data['text'] = text
        data['timestep'] = timestep_data # (len,)
        data['state'] = state_data # (len, state_dim)
        data['action'] = action_data # (len, action_dim)
        data['attention_mask'] = attention_mask_data # (len,)

        return data
    
    def visualize_action(self):
        """Visualize the distribution of actions."""
        with open(os.path.join(self.dataset_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)
        n_trajs = self.meta["num_trajectories"]
        xyz_actions = []
        rpy_actions = []
        xyz_states = []
        rpy_states = []
        for traj_idx in range(n_trajs):
            temp_robot_states = self.robot_states[traj_idx]
            n_frames = self.meta[str(traj_idx)][1]
            for i in range(0, n_frames):
                xyz_states.append(temp_robot_states[i, :3])
                rpy_states.append(temp_robot_states[i, 3:6])
            for i in range(1, n_frames):
                xyz_action = temp_robot_states[i, :3] - temp_robot_states[i-1, :3]
                rpy_action = temp_robot_states[i, 3:6] - temp_robot_states[i-1, 3:6]
                xyz_actions.append(xyz_action)
                rpy_actions.append(rpy_action)
        print(f"number of actions: {len(xyz_actions)}")
        xyz_actions = np.array(xyz_actions)
        rpy_actions = np.array(rpy_actions)
        xyz_states = np.array(xyz_states)
        rpy_states = np.array(rpy_states)
        a_labels = ['a_x', 'a_y', 'a_z']
        for i in range(len(a_labels)):
            plt.figure()
            plt.hist(xyz_actions[:, i], bins=512, label=a_labels[i], alpha=0.5)
            plt.legend(loc='upper right')
            plt.savefig(f"./data_stats/{a_labels[i]}.png")
        a_labels = ['a_roll', 'a_pitch', 'a_yaw']
        for i in range(len(a_labels)):
            plt.figure()
            plt.hist(rpy_actions[:, i], bins=512, label=a_labels[i], alpha=0.5)
            plt.legend(loc='upper right')
            plt.savefig(f"./data_stats/{a_labels[i]}.png")
        s_labels = ['s_x', 's_y', 's_z']
        for i in range(len(s_labels)):
            plt.figure()
            plt.hist(xyz_states[:, i], bins=512, label=s_labels[i], alpha=0.5)
            plt.legend(loc='upper right')
            plt.savefig(f"./data_stats/{s_labels[i]}.png")
        s_labels = ['s_roll', 's_pitch', 's_yaw']
        for i in range(len(s_labels)):
            plt.figure()
            plt.hist(rpy_states[:, i], bins=512, label=s_labels[i], alpha=0.5)
            plt.legend(loc='upper right')
            plt.savefig(f"./data_stats/{s_labels[i]}.png")

        abs_xyz_actions = np.abs(xyz_actions)
        abs_rpy_actions = np.abs(rpy_actions)
        x_action_max = np.max(abs_xyz_actions[:, 0])
        y_action_max = np.max(abs_xyz_actions[:, 1])
        z_action_max = np.max(abs_xyz_actions[:, 2])
        x_action_min = np.min(abs_xyz_actions[:, 0])
        y_action_min = np.min(abs_xyz_actions[:, 1])
        z_action_min = np.min(abs_xyz_actions[:, 2])
        x_action_mean = np.mean(abs_xyz_actions[:, 0])
        y_action_mean = np.mean(abs_xyz_actions[:, 1])
        z_action_mean = np.mean(abs_xyz_actions[:, 2])

        print(f"xyz_action max: {x_action_max:.3f}, {y_action_max:.3f}, {z_action_max:.3f}")
        print(f"xyz_action min: {x_action_min:.3f}, {y_action_min:.3f}, {z_action_min:.3f}")
        print(f"xyz_action mean: {x_action_mean:.3f}, {y_action_mean:.3f}, {z_action_mean:.3f}")

        er_action_max = np.max(abs_rpy_actions[:, 0])
        ep_action_max = np.max(abs_rpy_actions[:, 1])
        ey_action_max = np.max(abs_rpy_actions[:, 2])
        er_action_min = np.min(abs_rpy_actions[:, 0])
        ep_action_min = np.min(abs_rpy_actions[:, 1])
        ey_action_min = np.min(abs_rpy_actions[:, 2])
        er_action_mean = np.mean(abs_rpy_actions[:, 0])
        ep_action_mean = np.mean(abs_rpy_actions[:, 1])
        ey_action_mean = np.mean(abs_rpy_actions[:, 2])

        print(f"rpy_action max: {er_action_max:.3f}, {ep_action_max:.3f}, {ey_action_max:.3f}")
        print(f"rpy_action min: {er_action_min:.3f}, {ep_action_min:.3f}, {ey_action_min:.3f}")
        print(f"rpy_action mean: {er_action_mean:.3f}, {ep_action_mean:.3f}, {ey_action_mean:.3f}")

    def visualize_episode_len(self):
        """Visualize the length distribution of episodes."""
        pass
    
    def collator(self, sample):
        image_tensors = torch.stack([s['rgb'] for s in sample], dim=0)
        gripper_tensors = torch.stack([s['hand_rgb'] for s in sample], dim=0)
        action_tensors = torch.stack([s['action'] for s in sample], dim=0)
        state_tensors = torch.stack([s['state'] for s in sample], dim=0)
        robot_obs = state_tensors.clone()
        text = [s['text'] for s in sample]
        # print(text)
        text_tensors, attention_mask = self.text_fn(text)
        # print(text_tensors, attention_mask)
        return image_tensors, (text_tensors, attention_mask), action_tensors, gripper_tensors, state_tensors, robot_obs


if __name__ == "__main__":
    # import clip
    from transformers import AutoTokenizer
    import functools
    
    def preprocess_text_calvin(sample, tokenizer):
        tokenizer.padding_side = "right"
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
        text = tokenizer(
            sample,
            max_length=32,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
        return text["input_ids"], text["attention_mask"]

    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/robotics/lxh/mpt-1b-redpajama-200b-dolly", local_files_only=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    seq_len = 12
    data_dir = "/mnt/bn/robotics-data-hl/real_data/mode1_data_pick_place_001_1023/"
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    DS = RealDatasetHDF5(
        data_dir,
        None,
        text_fn=preprocess_text_fn,
        seq_len=seq_len,
        mode='train',
        action_mode='ee_rel_pose_local',
        use_data_augmentation=True)
    # dataloader = DataLoader(dataset=DS, batch_size=2, num_workers=2, collate_fn=DS.collator)
    # for i, data in enumerate(dataloader):
        
    #     image_tensors, (text_tensors, attention_mask), action_tensors, gripper_tensors, state_tensors, robot_obs = data
    #     print(image_tensors.shape)
    #     print(text_tensors.shape, attention_mask.shape)
    #     print(action_tensors.shape)
    #     print(state_tensors.shape)
    
    # exit(0)
    # # Visualize action distribution
    # DS.visualize_action()
    
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])

    # data_loader_val = torch.utils.data.DataLoader(
    #     DS, 
    #     batch_size=1, 
    #     num_workers=32, 
    #     drop_last=True)
    # for i, data in enumerate(data_loader_val):
    #     print(f"--{i}--")


    # Visualize data
    np.set_printoptions(3, suppress=True)
    ds_ids = np.arange(len(DS))
    np.random.shuffle(ds_ids)
    all_texts = []
    for i in tqdm(ds_ids):
        data = DS[i]
        text = data['text']
        all_texts.append(text)
        continue
        rgb = data['rgb']
        hand_rgb = data['hand_rgb']
        timestep = data['timestep']
        state = data['state']
        action = data['action']
        attention_mask = data['attention_mask']
        
        # video_name = data['video_name']
        print(f"{text}")

        # state
        state = state.numpy()
        gripper_state = state[:, -1].tolist()
        # if (-1 in gripper_state) and (1 in gripper_state):
        if True:
            print("State")
            for k in range(seq_len):
                print(f"{k}: {state[k, :]}")

            # action
            print("Action")
            action = action.numpy()
            for k in range(seq_len):
                print(f"{k}: {action[k, :]}")

            # RGB
            fig, ax = plt.subplots(seq_len // 4 + 1, 4)
            for k in range(seq_len):
                row = k // 4
                col = k % 4
                temp_rgb = rgb[k].permute(1, 2, 0).numpy() # (224, 224, 3)
                temp_rgb = temp_rgb * rgb_std + rgb_mean
                temp_rgb = np.clip(temp_rgb, 0, 1)
                ax[row, col].imshow(temp_rgb)
            plt.savefig("rgb.png", dpi=300)

            # hand RGB
            fig, ax = plt.subplots(seq_len // 4 + 1, 4)
            for k in range(seq_len):
                row = k // 4
                col = k % 4
                temp_rgb = hand_rgb[k].permute(1, 2, 0).numpy() # (224, 224, 3)
                temp_rgb = temp_rgb * rgb_std + rgb_mean
                temp_rgb = np.clip(temp_rgb, 0, 1)
                ax[row, col].imshow(temp_rgb)
            plt.savefig("hand_rgb.png", dpi=300)

            import pdb; pdb.set_trace()
    all_texts = list(set(all_texts))
    print(all_texts)