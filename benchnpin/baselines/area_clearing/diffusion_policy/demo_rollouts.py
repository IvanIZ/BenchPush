import argparse
import os
import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader

from benchnpin.baselines.area_clearing.diffusion_policy.dataset.area_clearing_dataset import AreaClearingDataset
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.multi_image_obs_encoder import MultiImageObsEncoder
from benchnpin.baselines.feature_extractors import ResNet18
from benchnpin.baselines.area_clearing.diffusion_policy.policy import AreaClearingDiffusion
from benchnpin.baselines.area_clearing.diffusion_policy.dataset.dataset_collector import DatasetCollector

from collections import deque
from einops import rearrange


def build_shape_meta(sample, obs_key="image"):
    """
    sample: one item from AreaClearingDataset
            {
              'obs': {'image': (T, C, H, W)},
              'action': (T, A)
            }
    use per-step channel/height/width for shape_meta['obs'][obs_key].
    """
    obs = sample["obs"][obs_key]
    act = sample["action"]
    _, C, H, W = obs.shape
    A = act.shape[-1]
    shape_meta = {
        "obs": {
            obs_key: {
                "shape": (C, H, W),
                "type": "rgb"  # 4 channels but treated as image tensor; MultiImageObsEncoder needs this flag
            }
        },
        "action": {
            "shape": (A,)
        }
    }
    return shape_meta



def rollouts(model_path : str, num_eps : int, env_type="area-clearing-v0", config=None):
    # ! Make sure these match the training args
    parser = argparse.ArgumentParser()
    # TODO: finish adding helps 
    parser.add_argument("--zarr_path", type=str, default="data/area_clearing/replay.zarr")
    parser.add_argument("--horizon", type=int, default=16, help="Number of steps predicted during model forward pass")
    parser.add_argument("--n_obs_steps", type=int, default=4, help="# of observation steps used for conditioning (To in the paper)")
    parser.add_argument("--n_action_steps", type=int, default=8, help="# of action steps executed (Ta in the paper)")
    parser.add_argument("--obs_as_global_cond", action="store_true", default=True)
    parser.add_argument("--scheduler_steps", type=int, default=100)
    parser.add_argument("--scheduler_beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--encoder_dim", type=int, default=512, help="ResNet18 feature embedding dim (flattened)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    assert args.n_action_steps <= args.horizon - (args.n_obs_steps - 1), "Need to follow: args.n_action_steps <= args.horizon - (args.n_obs_steps - 1)"
    
    # set seeds
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    
    # config env
    env = gym.make(env_type, config)
    env = env.unwrapped
    
    # load training results
    training_dict = torch.load(model_path)
    
    # datset (just need for building obs encoder)
    dataset = AreaClearingDataset(
        zarr_path=args.zarr_path,
        horizon=args.horizon,
        val_ratio=0.1,
        seed=42
    )
    # look at one sample to infer shapes
    sample = dataset[0]
    shape_meta = build_shape_meta(sample, obs_key="image")

    # Build the observation encoder
    C, H, W = shape_meta["obs"]["image"]["shape"]
    observation_space = spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)
    rgb_backbone = ResNet18(observation_space=observation_space, features_dim=args.encoder_dim)
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_backbone,
        share_rgb_model=True,
        imagenet_norm=False, 
        resize_shape=None,  # keep same dataset resolution
        crop_shape=None,
        random_crop=False
    )

    # scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.scheduler_steps,
        beta_schedule=args.scheduler_beta_schedule
    )

    # policy
    policy = AreaClearingDiffusion(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        obs_encoder=obs_encoder,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=args.scheduler_steps,
        obs_as_global_cond=args.obs_as_global_cond,
        seed=args.seed
    )
    
    # load weights
    policy.model.load_state_dict(training_dict["model"])
    policy.obs_encoder.load_state_dict(training_dict["obs_encoder"])
    policy.normalizer.load_state_dict(training_dict["normalizer"])
    policy.ema_model.load_state_dict(training_dict["ema_model"])
    
    def count_params(module : torch.nn.Module):
        total = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total
    
    print(f"Total number of params (Encoder + UNet): {count_params(policy.model) + count_params(policy.obs_encoder)}")
    
    # put policy in eval mode
    policy.model.eval()
    policy.obs_encoder.eval()
    policy.ema_model.eval()
    
    # metric = TaskDrivenMetric(alg_name="PPO", robot_mass=env.cfg.agent.mass)
    rewards_list = []
    
    # need to collect To observations first
    To = policy.n_obs_steps
    obs_buffer = deque(maxlen=To)
    
    for eps in range(num_eps):
        print(f"Episode #{eps + 1}")
        obs, info = env.reset(seed)
        done = truncated = False
        eps_reward = 0.0 
        
        # ! Just append the first obs To times (may need to change this?)
        # ! maybe do random walks To times in the beginning? 
        for _ in range(To):
            obs_buffer.append(obs)
        
        while True:
            obs_stack = np.stack(obs_buffer, axis=0)  # shape: (To, H, W, C)
            obs_stack = torch.from_numpy(obs_stack).to(device=policy.device)
            obs_stack = rearrange(obs_stack, "t h w c -> t c h w")
            obs_stack = obs_stack / 255.0  # preprocessing step done in area_clearing_dataset
            obs_stack = obs_stack[None, ...]  # add batch axis since it's expected 
            
            # NOTE: normalizing is handled within policy, so no need to do it here 
            obs_dict = {"image" : obs_stack}
            
            with torch.no_grad():            
                action_dict = policy.act(obs_dict=obs_dict)
                action = action_dict['action'][0][0]
            
            action = action.detach().cpu().numpy()

            obs, reward, terminate, trunc, info = env.step(action)
            
            eps_reward += reward
            done = terminate
            truncated = trunc
            env.render()
            # add new obs to buffer
            obs_buffer.append(obs)
            
            if done or truncated:
                obs_buffer.clear()
                rewards_list.append(eps_reward)
                break
    
    env.close()
    

if __name__ == "__main__":
    rollouts(model_path="baselines/area_clearing/diffusion_policy/checkpoints/20250925-1707/epoch_0040.pt",
             num_eps=10)
