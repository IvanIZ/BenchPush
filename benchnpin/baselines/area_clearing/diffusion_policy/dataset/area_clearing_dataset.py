import torch
import copy
import numpy as np
from einops import rearrange
from pathlib import Path
from benchnpin.baselines.area_clearing.diffusion_policy.dataset.base_dataset import *
from benchnpin.baselines.area_clearing.diffusion_policy.utils.pytorch_util import dict_apply
from benchnpin.baselines.area_clearing.diffusion_policy.utils.replay_buffer import ReplayBuffer
from benchnpin.baselines.area_clearing.diffusion_policy.utils.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.normalizer import LinearNormalizer
from benchnpin.baselines.area_clearing.diffusion_policy.utils.normalize_util import get_image_range_normalizer


class AreaClearingDataset(BaseImageDataset):
    """
    Read zarr data and put in form for training
    """
    def __init__(self,
        zarr_path, 
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None
        ):
    
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            # 'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        # image = np.moveaxis(sample['img'],-1,1)/255
        image = rearrange(sample['img'].astype(np.float32) / 255.0, "t h w c -> t c h w")

        data = {
            'obs': {
                'image': image, # T, 4, 224, 224
                # 'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 1
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('data/area_clearing/replay.zarr')
    dataset = AreaClearingDataset(zarr_path, horizon=4)
    first = dataset[0]
    print(first['obs']['image'].shape, first['action'].shape)
    print(1)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)


if __name__ == "__main__":
    test()
