import argparse
import os
import numpy as np
import torch

from gymnasium import spaces
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader

from benchnpin.baselines.area_clearing.diffusion_policy.dataset.area_clearing_dataset import AreaClearingDataset
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.multi_image_obs_encoder import MultiImageObsEncoder
from benchnpin.baselines.feature_extractors import ResNet18
from benchnpin.baselines.area_clearing.diffusion_policy.policy import AreaClearingDiffusion
from benchnpin.baselines.area_clearing.diffusion_policy.dataset.dataset_collector import DatasetCollector


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


def main():
    parser = argparse.ArgumentParser()
    # TODO: finish adding helps 
    parser.add_argument("--collect_dataset", action="store_true", default=False)
    parser.add_argument("--zarr_path", type=str, default="data/area_clearing/replay.zarr")
    parser.add_argument("--horizon", type=int, default=16, help="Number of steps predicted during model forward pass")
    parser.add_argument("--n_obs_steps", type=int, default=4, help="# of observation steps used for conditioning (To in the paper)")
    parser.add_argument("--n_action_steps", type=int, default=8, help="# of action steps executed (Ta in the paper)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--sample_every", type=int, default=5)
    # parser.add_argument("--rollout_every", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--obs_as_global_cond", action="store_true", default=True)
    parser.add_argument("--scheduler_steps", type=int, default=100)
    parser.add_argument("--scheduler_beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--encoder_dim", type=int, default=512, help="ResNet18 feature embedding dim (flattened)")
    parser.add_argument("--run_dir", type=str, default="baselines/area_clearing/diffusion_policy/runs")
    parser.add_argument("--checkpoint_dir", type=str, default="baselines/area_clearing/diffusion_policy/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    assert args.n_action_steps <= args.horizon - (args.n_obs_steps - 1), "Need to follow: args.n_action_steps <= args.horizon - (args.n_obs_steps - 1)"
    
    # set seeds
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    
    args.run_dir = os.path.abspath(args.run_dir)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # os.makedirs("baselines/area_clearing/diffusion_policy/runs", exist_ok=True)
    # os.makedirs("baselines/area_clearing/diffusion_policy/checkpoints", exist_ok=True)

    if args.collect_dataset:
        policy_weights_path = None
        collector = DatasetCollector(policy_weights_path=policy_weights_path)
        collector.save_to_zarr(out_path=args.zarr_path, num_eps=200, batch_eps=100)

    # datset
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

    # start training 
    policy.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_every=args.val_every,
        sample_every=args.sample_every,
        chkpoint_every=args.checkpoint_every,
        run_dir=args.run_dir,
        chkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()
