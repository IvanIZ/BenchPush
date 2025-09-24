from benchnpin.baselines.base_class import BasePolicy
from benchnpin.baselines.feature_extractors import ResNet18
import benchnpin.environments
import gymnasium as gym
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from itertools import chain
from datetime import datetime

from typing import Dict
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


class AreaClearingDiffusion(BasePolicy):
    """
    Based on: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py
    """
    def __init__(self, 
                 shape_meta : dict, # * dict for storing shapes
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: MultiImageObsEncoder, # ? End-to-end trained visual encoder
                 horizon,
                 n_action_steps,
                 n_obs_steps, # ? how's this different from horizon 
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
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        self.g = torch.Generator(device=self.device)
        self.g.manual_seed(self.seed)
        
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
                
        self.obs_encoder = obs_encoder.to(self.device)
        self.model = model.to(self.device)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model = self.ema_model.to(self.device)
        
        # NOTE: End-to-end training so include encoder params too 
        params = chain(self.model.parameters(), self.obs_encoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=1.0e-4, betas=(0.95, 0.999), weight_decay=1.0e-6)
        optimizer_to(self.optimizer, self.device)

        self.noise_scheduler = noise_scheduler
        # ? what does this do 
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
            this_nobs = dict_apply(nobs, lambda x: x[:, :To,...].reshape(-1, *x.shape[2:]))
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
        
        train_dataloader = DataLoader(dataset, 
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      shuffle=True,
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
            "cosine", self.optimizer, num_warmup_steps=500, 
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
        best_val = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            self.obs_encoder.train()
            train_losses = []
            
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}", 
                           leave=False, mininterval=1.0) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    
                    # compute loss 
                    loss = self.compute_loss(batch)
                    loss.backward()
                    
                    # step optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                    
                    # update ema 
                    ema.step(self.model)
                    
                    # logs
                    loss_val = float(loss.detach().cpu())
                    train_losses.append(loss_val)
                    tepoch.set_postfix(loss=loss_val, refresh=False)

                    # per-step logging
                    writer.add_scalar("train/batch_loss", loss_val, global_step)
                    writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], global_step)

                    global_step += 1
            
            # epoch average 
            train_loss = np.mean(train_losses)
            writer.add_scalar("train/epoch_avg_loss", train_loss, epoch)
            
            # ****** validation for the epoch ********
            val_loss = None
            self.model.eval()
            self.obs_encoder.eval()
            if (epoch % val_every) == 0:
                with torch.no_grad():
                    val_batch_losses = []
                    with tqdm.tqdm(val_dataloader, desc=f"Val {epoch}", 
                                   leave=False, mininterval=1.0) as vepoch:
                        for vbatch in vepoch:
                            vbatch = dict_apply(vbatch, lambda x: x.to(self.device, non_blocking=True))
                            vloss = self.compute_loss(vbatch)
                            val_batch_losses.append(float(vloss.detach().cpu()))
                    if val_batch_losses:
                        val_loss = np.mean(val_batch_losses)
                        writer.add_scalar("avg_val/loss", val_loss, epoch)

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
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": self.model.state_dict(),
                    "ema_model": self.ema_model.state_dict(),
                    "obs_encoder": self.obs_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "normalizer": self.normalizer.state_dict(),
                }, ckpt_path)

            # Track best
            metric = val_loss
            if metric < best_val:
                best_val = metric
                best_path = os.path.join(chkpoint_run_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": self.model.state_dict(),
                    "ema_model": self.ema_model.state_dict(),
                    "obs_encoder": self.obs_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "normalizer": self.normalizer.state_dict(),
                }, best_path)
        
        writer.close()

    def evaluate(self, policy):
        # TODO: implement this using the benchnpin eval metrics
        pass
    
    def set_normalizer(self, normalizer : LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
    def compute_loss(self, batch):
        """
        Computes loss on noise predictions. But can be configured for computing loss on the clean samples
        """
        device = self.device
        
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
                                   lambda x: x[:, :self.n_obs_steps,...].reshape(-1, *x.shape[2:]))
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
        
        # pred the noise residual 
        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=local_cond, global_cond=global_cond)
        
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.to(loss.dtype)
        loss = loss.view(loss.shape[0], -1).mean(dim=1).mean()
        return loss
