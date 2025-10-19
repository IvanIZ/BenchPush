from benchnpin.baselines.base_class import BasePolicy
from benchnpin.baselines.feature_extractors import ResNet18_mod
import benchnpin.environments
import gymnasium as gym
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
from datetime import datetime
from dataclasses import dataclass
# from line_profiler import profile

import argparse
from typing import Dict
from collections import deque
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.normalizer import LinearNormalizer
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.UNet_model import *
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.multi_image_obs_encoder import MultiImageObsEncoder
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.mask_generator import LowdimMaskGenerator
from benchnpin.baselines.area_clearing.diffusion_policy.utils.pytorch_util import dict_apply, optimizer_to
from einops import rearrange, reduce
from benchnpin.baselines.area_clearing.diffusion_policy.dataset.area_clearing_dataset import *
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.ema_model import EMAModel
from benchnpin.baselines.area_clearing.diffusion_policy.policy_components.lr_scheduler import get_scheduler
from benchnpin.common.metrics.task_driven_metric import TaskDrivenMetric

# from benchnpin.common.metrics.task_driven_metric import TaskDrivenMetric
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@dataclass
class EvalConfig:
    zarr_path: Path = Path("data/area_clearing/replay.zarr")
    horizon: int = 16
    n_obs_steps: int = 4
    n_action_steps: int = 8
    obs_as_global_cond: bool = True
    scheduler_steps: int = 100
    scheduler_beta_schedule: str = "squaredcos_cap_v2"
    seed: int = 42


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


class AreaClearingDiffusion(BasePolicy):
    """
    Based on: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py
    """
    def __init__(self, 
                 shape_meta : dict, # * dict for storing shapes
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: MultiImageObsEncoder, 
                 horizon,
                 n_action_steps,
                 n_obs_steps, 
                 rank=None,
                 local_rank=None,
                 num_inference_steps=None, 
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5, 
                 n_groups=8,
                 cond_predict_scale=True,
                 seed=42,
                 **kwargs
                 ):
        super().__init__()
        
        # set device using local rank
        if rank is not None and local_rank is not None:
            self.device = torch.device(f"cuda:{local_rank}")
            self.local_rank = local_rank
            self.rank = rank
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.seed = seed

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        
        # create diffusion model
        # * If obs aren't used for conditioning, model the joint distribution instead 
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        # multi-gpu setup        
        obs_encoder = obs_encoder.to(self.device)
        model = model.to(self.device)
        ema_model = copy.deepcopy(model)
        self.ema_model = ema_model.to(self.device)
        
        if rank is not None and local_rank is not None:
            self.obs_encoder = DDP(obs_encoder, device_ids=[local_rank], output_device=local_rank, 
                                find_unused_parameters=True)
            self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            self.obs_encoder = obs_encoder
            self.model = model
        
        self.g = torch.Generator(device=str(self.device))
        self.g.manual_seed(self.seed)
        
        if self.device.type == 'cuda':
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device.type == 'mps':
            self.amp_dtype = torch.float16
        else:  # cpu
            self.amp_dtype = torch.bfloat16
        
        # NOTE: End-to-end training so include encoder params too 
        params = chain(self.model.parameters(), self.obs_encoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=1.0e-4, betas=(0.95, 0.999), weight_decay=1.0e-3)
        optimizer_to(self.optimizer, self.device)

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps  # num of action steps being outputed 
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        
        # num of reverse process steps to do when denoising back to actions 
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        eval_cfg: EvalConfig,
        trained_DDP: bool = False,
        ):
        """Builds policy and loads weights; returns (policy, shape_meta)."""
        # datset just to infer shapes
        zarr_path = Path(os.path.expanduser(str(eval_cfg.zarr_path)))
        dataset = AreaClearingDataset(
            zarr_path=zarr_path,
            horizon=eval_cfg.horizon,
            val_ratio=0.1,
            seed=42,
        )
        sample = dataset[0]
        shape_meta = build_shape_meta(sample, obs_key="image")

        # obs encoder
        C, H, W = shape_meta["obs"]["image"]["shape"]
        observation_space = gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)
        rgb_backbone = ResNet18_mod(observation_space=observation_space)
        obs_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_backbone,
            share_rgb_model=True,
            use_group_norm=True,
            imagenet_norm=False,
            resize_shape=None,
            crop_shape=None,
            random_crop=False,
        )

        # scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=eval_cfg.scheduler_steps,
            beta_schedule=eval_cfg.scheduler_beta_schedule,
        )

        # policy
        policy = cls(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=eval_cfg.horizon,
            n_action_steps=eval_cfg.n_action_steps,
            n_obs_steps=eval_cfg.n_obs_steps,
            num_inference_steps=eval_cfg.scheduler_steps,
            obs_as_global_cond=eval_cfg.obs_as_global_cond,
            seed=eval_cfg.seed,
        )
        
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        def _strip_module_prefix(sd):
            consume_prefix_in_state_dict_if_present(sd, prefix="module.")
            return sd

        # load weights
        ckpt = torch.load(model_path)
        
        # need to get rid of module. if saved from DDP
        if trained_DDP:
            for key in ["model", "ema_model", "obs_encoder", "normalizer"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    _strip_module_prefix(ckpt[key])
        
        policy.model.load_state_dict(ckpt["model"])
        policy.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        policy.normalizer.load_state_dict(ckpt["normalizer"])
        policy.ema_model.load_state_dict(ckpt["ema_model"])

        return policy, shape_meta

    # ******* Inference ******** 
    @torch.no_grad()
    def conditional_sample(self, 
                           condition_data, condition_mask,
                           model=None,
                           local_cond=None, global_cond=None,
                           generator=None, 
                           **kwargs
                           ):
        """
        NOTE: Pass ema weights in model
        """
        device = self.device
        
        if model is None:
            model = self.model
        else:
            model = model.to(device)
            
        if global_cond is not None:
            global_cond = global_cond.to(device=device)
        if local_cond is not None:
            local_cond = dict_apply(local_cond, lambda x: x.to(device=device))

        scheduler = self.noise_scheduler
        
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=device,
            generator=generator
        )
        
        # set step values
        scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = scheduler.timesteps.to(device)
        
        for t in timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # 2. predict model output
            with torch.autocast(device_type=device.type, dtype=self.amp_dtype):
                model_output = model(trajectory, t,
                                    local_cond=local_cond, 
                                    global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_{t-1}
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator, **kwargs
            ).prev_sample
        
        # make sure conditioning is enforce
        trajectory[condition_mask] = condition_data[condition_mask]
        
        return trajectory
    
    @torch.no_grad()
    def act(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        
        assert 'past_action' not in obs_dict  # not implemented 
        device = self.device
        dtype = torch.float32
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = dict_apply(nobs, lambda x: x.to(device=device, non_blocking=True))
        
        value = next(iter(nobs.values()))
        
        # B, To = value.shape[:2]
        B = value.shape[0]
        T = self.horizon  # * CHECK: Should be To + Ta
        Da = self.action_dim 
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        # handle different ways of passing observation 
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condtion through global feature
            # NOTE: * is used for argument unpacking (shape returns tuple that needs unpacking)
            this_nobs = dict_apply(nobs, lambda x: x[:, :To,...].contiguous().reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action since we condition on obs, not actions 
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through inpainting: 
            # - give the model a mask M that marks which entries are known and which to denoise 
            this_nobs = dict_apply(nobs, lambda x: x[:, :To,...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To, Da:] = nobs_features
            cond_mask[:,:To, Da:] = True
        
        # run sampling
        # NOTE: uses ema model
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            model=self.ema_model,
            local_cond=local_cond,
            global_cond=global_cond,
            generator=self.g,
            **self.kwargs
        )
        
        # unnormalize pred
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        if action_pred.device != device:
            action_pred = action_pred.to(device)
        
        # get action 
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    # ********* training *********
    def train(self, 
              dataset : AreaClearingDataset, 
              num_epochs=1000, 
              batch_size=1, 
              num_workers=4,
              val_every=1,
              sample_every=5,
              chkpoint_every=50,
              run_dir="runs",
              chkpoint_dir="checkpoints"):
        """
        - sample_every : eval policy and look at the action preds
        Based on https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py
        """
        os.makedirs(chkpoint_dir, exist_ok=True)
        
        if self.rank is not None and self.local_rank is not None:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                            num_replicas=dist.get_world_size(),
                                                                            rank=self.rank,
                                                                            drop_last=True)
            train_dataloader = DataLoader(dataset, 
                                        batch_size=batch_size,
                                        sampler=train_sampler,
                                        num_workers=num_workers,
                                        shuffle=False,
                                        pin_memory=True)
        else:
            train_dataloader = DataLoader(dataset, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=False,
                                        pin_memory=True)
            
        normalizer = dataset.get_normalizer()
        
        # config the validation set 
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=False,
                                    pin_memory=True)
        
        self.set_normalizer(normalizer)
        
        # ! they used a last_epoch arg here
        # NOTE: num_training_steps is used to derive the learning rate schedule
        total_train_steps = len(train_dataloader) * num_epochs
        lr_scheduler = get_scheduler(
            "cosine", self.optimizer, num_warmup_steps=int(0.05 * total_train_steps), 
            num_training_steps=total_train_steps,
        )
        
        # config ema copy 
        ema = EMAModel(self.ema_model, power=0.75)
        
        # # config env 
        # # ? Is this needed? 
        # env = gym.make('area-clearing-v0')
        # env = env.unwrapped
     
        # tensorboard logging and chk point dir
        run_name = datetime.now().strftime("%Y%m%d-%H%M")
        writer = SummaryWriter(log_dir=os.path.join(run_dir, run_name))
        chkpoint_run_dir = os.path.join(chkpoint_dir, run_name)
        os.makedirs(chkpoint_run_dir, exist_ok=True)
        
        device = self.device
        print(f"Using device: {device}")
        # self.model.to(device)
        # self.obs_encoder.to(device)
        # self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)
        
        # save batch for sampling 
        train_sampling_batch = None
        global_step = 0
        # best_val = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            self.obs_encoder.train()
            train_losses = []
            show_bar = (epoch % 10 == 0)
            train_sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(train_dataloader):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = batch
                
                # compute loss 
                loss = self.compute_loss(batch)
                loss.backward()
                
                # NOTE: did this to check if params were updating in the encoder, they are
                # def module_checksum(module):
                #     s = 0.0
                #     for p in module.parameters():
                #         with torch.no_grad():
                #             s += p.float().norm().cpu()
                #     return s

                # pre = module_checksum(self.obs_encoder)

                # step optimizer
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                
                # post = module_checksum(self.obs_encoder)
                # print("obs_encoder changed:", not (abs(post - pre) < 1e-12))
                
                # update ema 
                # only need one process to update ema
                if self.rank == 0:
                    ema.step(self.model.module)  # unwrap DDP
                else:
                    ema.step(self.model)
                
                # logs
                loss_val = float(loss.detach().cpu())
                train_losses.append(loss_val)
        
                # per-step logging
                writer.add_scalar("train/batch_loss", loss_val, global_step)
                writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], global_step)

                global_step += 1
            
            # epoch average 
            train_loss = np.mean(train_losses)
            writer.add_scalar("train/epoch_avg_loss", train_loss, epoch)
            
            if self.rank == 0 and show_bar:
                print(f"Epoch {epoch} Average Train loss: {train_loss:.4f}")
            elif self.rank is None and show_bar:
                print(f"Epoch {epoch} Average Train loss: {train_loss:.4f}")
            
            # ****** validation for the epoch ********
            val_loss = None
            self.model.eval()
            self.ema_model.eval()
            self.obs_encoder.eval()
            if (epoch % val_every) == 0:
                with torch.no_grad():
                    val_batch_losses = []
                    val_ema_batch_losses = []
                    
                    for vbatch_idx, vbatch in enumerate(val_dataloader):
                        # ******* sampling metric on a val batch ********
                        if vbatch_idx == 0:
                            obs_dict = vbatch['obs']
                            expected_act = vbatch['action'].to(self.device)
                            
                            pred = self.act(obs_dict)
                            pred_action = pred['action_pred']
                        
                            mse = F.mse_loss(pred_action, expected_act)
                            writer.add_scalar("val/action_mse", float(mse.detach().cpu()), epoch) 
                        
                        vbatch = dict_apply(vbatch, lambda x: x.to(self.device, non_blocking=True))
                        vloss_model = self.compute_loss(vbatch, model=self.model)
                        vloss_ema_model = self.compute_loss(vbatch, model=self.ema_model)
                        val_batch_losses.append(float(vloss_model.detach().cpu()))
                        val_ema_batch_losses.append(float(vloss_ema_model.detach().cpu()))
                    if val_batch_losses:
                        val_loss = np.mean(val_batch_losses, axis=0)
                        val_ema_loss = np.mean(val_ema_batch_losses, axis=0)
                        writer.add_scalar("val/avg_loss", val_loss, epoch)
                        writer.add_scalar("val/avg_loss_ema_model", val_ema_loss, epoch)
                        
                        if self.rank == 0:
                            print(f"Epoch {epoch} Average Val loss: {val_loss:.4f}, Average Val EMA loss: {val_ema_loss:.4f}")
                        elif self.rank is None:
                            print(f"Epoch {epoch} Average Val loss: {val_loss:.4f}, Average Val EMA loss: {val_ema_loss:.4f}")
                            
            # ******* sampling metric on a train batch ********
            if (epoch % sample_every) == 0 and train_sampling_batch:
                with torch.no_grad():
                    obs_dict = train_sampling_batch['obs']
                    # obs_dict = dict_apply(obs_dict, lambda x: x.to(self.device, non_blocking=True))
                    expected_act = train_sampling_batch['action']
                    
                    pred = self.act(obs_dict) 
                    pred_action = pred['action_pred']
                    
                    mse = F.mse_loss(pred_action, expected_act)
                    writer.add_scalar("train/action_mse", float(mse.detach().cpu()), epoch)    
            
            # ****** checkpointing ***********
            if (epoch % chkpoint_every) == 0:
                ckpt_path = os.path.join(chkpoint_run_dir, f"epoch_{epoch:04d}.pt")
                if self.rank == 0:
                    # only save from one process (they share the same params across processes)
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "model": self.model.state_dict(),
                        "ema_model": self.ema_model.state_dict(),
                        "obs_encoder": self.obs_encoder.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "normalizer": self.normalizer.state_dict(),
                    }, ckpt_path)
                
                    dist.barrier() # sync processes
                    
                    # then load this model across all processes (reduces write overhead)
                    acc = torch.accelerator.current_accelerator()
                    map_location = {f'{acc}:0': f'{acc}:{self.local_rank}'}
                    training_dict = torch.load(ckpt_path, map_location=map_location, weights_only=True)
                    self.model.load_state_dict(training_dict["model"])
                    self.obs_encoder.load_state_dict(training_dict["obs_encoder"])
                    self.ema_model.load_state_dict(training_dict["ema_model"])
                    self.optimizer.load_state_dict(training_dict["optimizer"])
                    self.normalizer.load_state_dict(training_dict["normalizer"])
                else:
                    torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": self.model.state_dict(),
                    "ema_model": self.ema_model.state_dict(),
                    "obs_encoder": self.obs_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "normalizer": self.normalizer.state_dict(),
                    }, ckpt_path)
                    

            # # Track best
            # metric = val_loss
            # if metric < best_val:
            #     best_val = metric
            #     best_path = os.path.join(chkpoint_run_dir, "best.pt")
            #     torch.save({
            #         "epoch": epoch,
            #         "global_step": global_step,
            #         "model": self.model.state_dict(),
            #         "ema_model": self.ema_model.state_dict(),
            #         "obs_encoder": self.obs_encoder.state_dict(),
            #         "optimizer": self.optimizer.state_dict(),
            #         "normalizer": self.normalizer.state_dict(),
            #     }, best_path)
        
        writer.close()

    @classmethod
    def evaluate(cls, 
                 model_path: str,
                 num_eps: int, 
                 trained_DDP : bool,
                 env_type="area-clearing-v0", 
                 config=None,
                 eval_cfg=EvalConfig(),
                 ):
        
        assert eval_cfg.n_action_steps <= eval_cfg.horizon - (eval_cfg.n_obs_steps - 1), "Need to follow: n_action_steps <= horizon - (n_obs_steps - 1)"
        
        # set seeds
        np.random.seed(eval_cfg.seed)
        torch.manual_seed(eval_cfg.seed)
        torch.cuda.manual_seed_all(eval_cfg.seed)   
        
        # config env
        env = gym.make(env_type, config)
        env = env.unwrapped
        
        # load policy
        policy, _ = cls.from_checkpoint(model_path, eval_cfg, trained_DDP=trained_DDP)
        
        def count_params(module : torch.nn.Module):
            total = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total
        
        print(f"Total number of params (Encoder + UNet): {count_params(policy.model) + count_params(policy.obs_encoder)}")
        
        # put policy in eval mode
        policy.model.eval()
        policy.obs_encoder.eval()
        policy.ema_model.eval()
        
        metric = TaskDrivenMetric(alg_name="Diffusion_Policy", robot_mass=env.cfg.agent.mass)
        rewards_list = []
        
        # need to collect To observations first
        To = policy.n_obs_steps
        obs_buffer = deque(maxlen=To)
        
        for eps in range(num_eps):
            print(f"Episode #{eps + 1}")
            obs, info = env.reset(eval_cfg.seed)
            metric.reset(info)
            
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
                metric.update(info=info, reward=reward, eps_complete=(terminate or trunc))
                
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
        metric.plot_scores(save_fig_dir=os.path.expanduser("~/BenchNPIN/benchnpin/baselines/area_clearing/diffusion_policy"))

        return metric.success_rates, metric.efficiency_scores, metric.effort_scores, metric.rewards, "Diffusion Policy"
    
    def set_normalizer(self, normalizer : LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
    def compute_loss(self, batch, model=None):
        """
        Computes loss on noise predictions. But can be configured for computing loss on the clean samples
        """
        device = self.device
        
        if model is None:
            model = self.model
        
        # normalize input 
        assert 'valid_mask' not in batch 
        nobs = self.normalizer.normalize(batch['obs'])
        nobs = dict_apply(nobs, lambda x: x.to(device=device, non_blocking=True))

        nactions = self.normalizer['action'].normalize(batch['action'])
        nactions = nactions.to(device=device, non_blocking=True)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        
        # handle different ways of passing obs
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps,...].contiguous().reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate inpainting mask
        condition_mask = self.mask_generator(trajectory.shape).to(device)
        
        # sample noise to add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images based on the noise variance at each time step
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        
        # compute loss mask
        loss_mask = (~condition_mask)
        
        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        # pred the noise residual 
        with torch.autocast(device_type=device.type, dtype=self.amp_dtype):
            pred = model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.to(loss.dtype)
            loss = loss.view(loss.shape[0], -1).mean(dim=1).mean()
        
        return loss


def main():
    import argparse, os
    from pathlib import Path
    
    default_ckpt = os.path.expanduser(
        "~/BenchNPIN/benchnpin/baselines/area_clearing/diffusion_policy/checkpoints/20251007-2229/epoch_0040.pt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=default_ckpt)
    parser.add_argument("--num_eps", type=int, default=200)
    parser.add_argument("--env_type", type=str, default="area-clearing-v0")
    parser.add_argument("--zarr_path", type=str, default="data/area_clearing/replay.zarr")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=4)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--obs_as_global_cond", action="store_true", default=True)
    parser.add_argument("--scheduler_steps", type=int, default=100)
    parser.add_argument("--scheduler_beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_cfg = EvalConfig(
        zarr_path=Path(os.path.expanduser(args.zarr_path)),
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        obs_as_global_cond=args.obs_as_global_cond,
        scheduler_steps=args.scheduler_steps,
        scheduler_beta_schedule=args.scheduler_beta_schedule,
        seed=args.seed,
    )

    AreaClearingDiffusion.evaluate(
        model_path=os.path.expanduser(args.model_path),
        num_eps=args.num_eps,
        trained_DDP=True,
        env_type=args.env_type,
        config=None,
        eval_cfg=eval_cfg,
    )


if __name__ == "__main__":
    main()
