import ast
from cgitb import text
import functools
import io
import json
import logging
import math
import os
import random
import sys
import tarfile
from dataclasses import dataclass
from multiprocessing import Value
import zipfile

import braceexpand
import torch
import torchvision
import webdataset as wds
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

from calvin_agent.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)
import numpy as np
from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset
from robot_flamingo.data.real_dataset_hdf5 import RealDatasetHDF5

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


import logging
from pathlib import Path
from typing import Dict, Tuple, Union

from calvin_agent.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    # process_language,
    # process_rgb,
    process_state,
)
import numpy as np
from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset
from robot_flamingo.data.vl_dataset import CaptionDataset, VQADataset

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)


def get_validation_window_size(
    idx: int, min_window_size: int, max_window_size: int
) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


from typing import Any, Dict, List, Tuple, Callable
from itertools import chain
from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern
import pickle
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x


class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        # print('ws {}, min_ws {}, max_ws {}'.format(self.window_size, self.max_window_size, self.min_window_size))
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
       
        with open('/mnt/bn/robotics/lxh/robot-flamingo/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        assert (
            "validation" in self.abs_datasets_dir.as_posix()
            or "training" in self.abs_datasets_dir.as_posix()
        )
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx
        
        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)
        
        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list
        # print(pad_size, len(new_list))
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info


class DebugDataset(Dataset):
    def __init__(self, **kwargs: Any,):
        super().__init__()
    def __len__(self) -> int:
        return 10000
    def __getitem__(self, index):
        window_size = 8
        rgb = torch.randn(window_size, 3, 200, 200)
        gripper = torch.randn(window_size, 84, 84)
        state = torch.randn(window_size, 15)


class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if (start_idx, end_idx) not in partial_st_ed_list:
                    continue
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def collater(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        text_tensors, attention_mask = self.text_fn(stacked_language)
        
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
        
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)

            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
        
        return image_tensors, (text_tensors, attention_mask), action_tensors, gripper_tensors, state_tensors, robot_obs


class CalvinDataset(Dataset):
    """Naive implementation of dataset to store
    calvin debug dataset, may be changed to WDS for the full dataset
    """

    def __init__(self, image_fn, text_fn, dataset_path, is_train=True) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.image_fn = image_fn
        self.text_fn = text_fn

        tag = "training" if is_train else "validation"
        self.file_prefix = f"{self.dataset_path}/{tag}"
        self.anns = np.load(
            f"{self.file_prefix}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
        ).item()

    def __len__(self):
        return len(self.anns["info"]["indx"])

    def __getitem__(self, index):
        task = self.anns["language"]["task"][index]
        text = self.anns["language"]["ann"][index]
        st, ed = self.anns["info"]["indx"][index]
        # CJ: randomly sample a datapoint in the episode
        frame = random.randint(st, ed)
        frame = np.load(
            f"{self.file_prefix}/episode_{frame:07d}.npz"
        )  # , allow_pickle=True (lazy load)
        rgb_static = Image.fromarray(frame["rgb_static"])
        rgb_gripper = Image.fromarray(frame["rgb_gripper"])
        actions = np.array(frame["rel_actions"])
        
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        return rgb_static, text, actions

    def collater(self, sample):
        images = [s[0] for s in sample]
        texts = [s[1] for s in sample]
        actions = [s[2] for s in sample]

        image_tensors = self.image_fn(images)
        text_tensors = self.text_fn(texts)
        action_tensors = torch.FloatTensor(np.stack(actions))
        return image_tensors, text_tensors, action_tensors


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image


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


def preprocess_interleaved(sample, tokenizer, clip_processor, sim_threshold):
    info = json.loads(sample[0])
    tar_file_obj = io.BytesIO(sample[1])
    image_tar = tarfile.open(fileobj=tar_file_obj)
    sentences = info["text_list"]

    images, image_idxs = [], []
    for image_path, sim in zip(info["image_info"], info["similarity_matrix"]):
        # pick one image per sentence
        if info["image_info"][image_path]["matched_text_index"] in image_idxs:
            continue
        rawbytes = image_tar.extractfile(
            os.path.join(image_tar.getnames()[0], image_path)
        ).read()

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue
        if sim[info["image_info"][image_path]["matched_text_index"]] < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        image_idxs.append(info["image_info"][image_path]["matched_text_index"])

    if len(images) == 0:
        raise ValueError("No images in sample")

    # filter out images that are exact duplicates
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), MAX_NUM_IMAGES))
    images_tensors = images_tensors[keep_ixs]
    image_idxs = [image_idxs[ix] for ix in keep_ixs]

    # pad to 5 images
    if len(images_tensors) < MAX_NUM_IMAGES:
        zero_padding = torch.zeros(
            (MAX_NUM_IMAGES - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # add in <image> and <eoc> tokens
    # eoc after sentence = "sentence loss"
    for ix in image_idxs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"

    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )

    if num_images == 0:
        raise ValueError("No images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def get_coco_dataset(args, image_processor, tokenizer, epoch=0):
    coco_data_dir = "path/to/coco/train2014"
    coco_ann = "path/to/coco/annotations/captions_train2014.json"
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    coco_dataset = CaptionDataset(coco_data_dir, coco_ann, preprocess_text_fn, image_processor)
    
    sampler = DistributedSampler(
        coco_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    
    dataloader = DataLoader(
        coco_dataset,
        batch_size=args.batch_size_vl,
        pin_memory=False,
        num_workers=args.workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=coco_dataset.collator,
        drop_last=True
    )
    
    return dataloader


def get_vqa_dataset(args, image_processor, tokenizer, epoch=0):
    vqa_data_dir = "path/to/vqav2/train2014"
    vqa_questions = "path/to/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
    vqa_ann = "path/to/vqav2/v2_mscoco_train2014_annotations.json"
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    vqa_dataset = VQADataset(vqa_data_dir, vqa_questions, vqa_ann, preprocess_text_fn, image_processor)
    
    sampler = DistributedSampler(
        vqa_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    
    dataloader = DataLoader(
        vqa_dataset,
        batch_size=args.batch_size_vl,
        pin_memory=False,
        num_workers=args.workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=vqa_dataset.collator,
        drop_last=True
    )
    
    return dataloader


def get_calvin_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset

    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    calvin_dataset = DiskCalvinDataset(
        datasets_dir=Path(dataset_path) / "training",
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data
    )

    round_fn = math.floor if floor else math.ceil

    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size_calvin * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    # the batch_size and num_workers are per-GPU !
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size_calvin,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collater,
        drop_last=True
    )
    # dataloader = DataLoader(calvin_dataset, batch_size=args.batch_size_calvin)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)


def get_real_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset

    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    calvin_dataset = RealDatasetHDF5(
        image_fn=preprocess_image_fn,
        # data_dir="/mnt/bn/robotics-data-hl/real_data/mode1_data_pick_place_001_0912/",
        data_dir="/mnt/bn/robotics-data-hl/real_data/mode1_data_pick_place_001_1023/",
        text_fn=preprocess_text_fn,
        seq_len=args.window_size,
        text_aug=args.text_aug
    )

    round_fn = math.floor if floor else math.ceil

    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size_calvin * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    # the batch_size and num_workers are per-GPU !
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size_calvin,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    # dataloader = DataLoader(calvin_dataset, batch_size=args.batch_size_calvin)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)


def get_calvin_dataset_debug(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset

    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    calvin_dataset = DebugDataset(
        datasets_dir=Path(dataset_path) / "training",
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons
    )

    round_fn = math.floor if floor else math.ceil

    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size_calvin * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    # the batch_size and num_workers are per-GPU !
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size_calvin,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        drop_last=True
    )
    # dataloader = DataLoader(calvin_dataset, batch_size=args.batch_size_calvin)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)


def get_dataset_fn(dataset_type):
    if dataset_type == "calvin":
        return get_calvin_dataset
    elif dataset_type == 'debug':
        return get_calvin_dataset_debug
    elif dataset_type == "real":
        return get_real_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    return get_dataset_fn(dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    )

def load_partial_traj_data():
    with open('partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data