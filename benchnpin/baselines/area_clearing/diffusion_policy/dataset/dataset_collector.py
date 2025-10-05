import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import zarr
from numcodecs import Blosc


import torch
from benchnpin.baselines.area_clearing.ppo.policy import AreaClearingPPO


class DatasetCollector:
    """
    Using a baseline trained policy, collect data for training the diffusion model and save as a Zarr
    N_total is total number of samples over all of the episodes
    - 'img' : (N_total, H, W, C=4)  #NOTE: C is 4
    - 'action' : (N_total, 1)  
    - 'episode_ends' : (N_episodes, )
    
    NOTE: no state/poses included
    """
    def __init__(self, 
                 policy_name=None,
                 policy_weights_path=None,
                 cfg=None):
        if policy_name:
            policy = AreaClearingPPO(model_name=policy_name,
                                     model_path=policy_weights_path, cfg=cfg)
        else:
            policy = AreaClearingPPO(model_path=policy_weights_path, cfg=cfg)
            
        self.policy = policy
        
    def data_collection(self, num_eps=1000):
        return self.policy.transitions(num_eps=num_eps)
        
    def save_to_zarr(self,
                     out_path : str,
                     num_eps=1000, 
                     batch_eps=200
                     ):
        """
        Collect in chunks and append to a single Zarr:
          - create empty, growable datasets on first batch
          - append each batch and offset episode_ends by the number of steps already written
        """
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        store = zarr.DirectoryStore(str(out_p))
        root = zarr.group(store=store, overwrite=True)
        data_grp = root.require_group('data')
        extra_grp = root.require_group('extra')
        
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        
        n_steps_written = 0
        n_eps_written = 0
        img_shape_cached = None
        action_dim_cached = None
        img_ds = None
        
        for start in range(0, num_eps, batch_eps):
            n_eps = min(batch_eps, num_eps - start)  # if not divisble: num_eps / batch_eps
            
            img, action, episode_ends, rewards = self.data_collection(num_eps=n_eps)
            H, W, C = img.shape[1:]
            action_dim = action.shape[1]
            
            if img_ds is None:
                # create datasets
                # choose chunk sizes (assuming at least 10 steps are taken per eps)
                chunks_img_len = min(10 * batch_eps, img.shape[0])
                chunks_act_len = min(10 * batch_eps, action.shape[0])
                chunks_rew_len = min(10 * batch_eps, rewards.shape[0])
                chunks_ends = len(episode_ends)
                
                img_ds = data_grp.create_dataset(
                    'img',
                    shape=(0, H, W, C),
                    maxshape=(None, H, W, C),
                    chunks=(chunks_img_len, H, W, C),
                    dtype='u1',
                    compressor=compressor,
                )
                act_ds = data_grp.create_dataset(
                    'action',
                    shape=(0, action_dim),
                    maxshape=(None, action_dim),
                    chunks=(chunks_act_len, action_dim),
                    dtype='f4',
                    compressor=compressor,
                )
                ends_ds = extra_grp.create_dataset(
                    'episode_ends',
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(chunks_ends,),
                    dtype='i8',
                    compressor=compressor,
                )
                rew_ds = extra_grp.create_dataset(
                    'rewards_per_step',
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(chunks_rew_len,),
                    dtype='f4',
                    compressor=compressor,
                )
                
                root.attrs['img_shape'] = (H, W, C)
                root.attrs['action_dim'] = int(action_dim)
                img_shape_cached = (H, W, C)
                action_dim_cached = action_dim
            else:
                # same handles
                img_ds = data_grp['img']
                act_ds = data_grp['action']
                ends_ds = extra_grp['episode_ends']
                rew_ds = extra_grp['rewards_per_step']
                
                # check shape is still fine 
                assert img_ds.shape[1:] == img_shape_cached, f"Error - Image shape changed: {img.shape[1:]}"
                assert act_ds.shape[1] == action_dim_cached, f"Error - Action shape changed: {act_ds.shape[1]}"
                
            # concatenate current batch of steps w/ the prev ones
            B = img.shape[0]
            prev_N = img_ds.shape[0]
            
            img_ds.resize(prev_N + B, *img_shape_cached)
            img_ds[prev_N:prev_N + B, ...] = img.astype('u1', copy=False)
                
            act_ds.resize(prev_N + B, action_dim_cached)
            act_ds[prev_N:prev_N + B, ...] = action.astype('f4', copy=False)
            
            rew_ds.resize(prev_N + B)
            rew_ds[prev_N:prev_N + B] = rewards.astype('f4', copy=False)

            E = episode_ends.shape[0]
            prev_E = ends_ds.shape[0]
            ends_ds.resize(prev_E + E)
            # offset episode_ends by num of steps already written
            ends_ds[prev_E:, ...] = episode_ends.astype('i8', copy=False) + prev_N
            
            n_steps_written += B
            n_eps_written += E
            
        root.attrs['n_total'] = int(n_steps_written)
        root.attrs['n_episodes'] = int(n_eps_written)
                
        print(f"Saved {n_steps_written} steps from {n_eps_written} episodes to {out_path}")
        return n_steps_written, root.attrs['img_shape'], (root.attrs['action_dim'],)



def main():
    # policy_weights_path = "benchnpin/baselines/area_clearing/ppo/model"
    policy_weights_path = None
    out_path = "data/area_clearing/replay.zarr"
    collector = DatasetCollector(policy_weights_path=policy_weights_path)
    collector.save_to_zarr(out_path=out_path, num_eps=2000, batch_eps=100)

    
def test():
    # policy_weights_path = "benchnpin/baselines/area_clearing/ppo/model"
    policy_weights_path = None
    out_path = "data_test/area_clearing/replay.zarr"
    collector = DatasetCollector(policy_weights_path=policy_weights_path)
    collector.save_to_zarr(out_path=out_path, num_eps=20, batch_eps=2)
    
    root = zarr.open_group(out_path, mode='r')
    imgs = root['data']['img']
    actions = root['data']['action']
    eps_ends = root['meta']['episode_ends']
    print(imgs.shape, actions.shape, eps_ends.shape)
    print(f"Img chunks: {imgs.chunks}")


if __name__ == "__main__":
    main()
