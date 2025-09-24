import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from einops.layers.torch import Rearrange

# based on https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/conv1d_components.py

class Downsample1D(nn.Module):
    """
    Conv layer with stride 2 for downsampling. NOTE channels are not changed here
    
    NOTE to self: With groups=1 means each output channel is a weighted sum of all input 
    channels over the kernel size.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, groups=1)
    
    def forward(self, x):
        return self.conv(x)
    

class Upsample1D(nn.Module):
    """
    Upsamples the length by 2x and keeps the channels the same 
    """
    
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1, groups=1)
        
    def forward(self, x):
        return self.conv(x)
    

class Conv1DBlock(nn.Module):
    """
    Conv1D -> GroupNorm -> Mish
    
    NOTE:
    GroupNorm: splits channels into groups and computes within each group the mean and variance for normalization 
    -- No batch dimension AND no running stats like in BatchNorm
    -- n_groups is number of groups to separate channels into
    Mish: activation that kind of behaves like relu but is smooth and non-monotonic (not consistently increasing or decreasing)
    """
    
    def __init__(self, c_in, c_out, kernel_size, n_groups=8):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2), # IF k is even, L_out = L_in + 1, otherwise L_out = L_in
            nn.GroupNorm(n_groups, c_out),  
            nn.Mish()
        )
    
    def forward(self, x):
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal position embedding for timestep t
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device  # REM: Creating a new tensor so device is needed. Other classes in this file use already existing tensors 
        half_dim = self.dim // 2  # one half for sin, one half for cos
        
        c = math.log(10000) / (half_dim - 1)  # log-space step size so freq are evenly spaced from 1 to 1/10000 on a log scale
        # NOTE: the ratio between inverse frequencies that are a step apart in log space are constant
        
        emb = torch.exp(torch.arange(half_dim, device=device) * -c)  # calc the inverse frequencies
        emb = x[:, None] * emb[None, :]  # pos * inv_freq (with broadcasting)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual block with FiLM conditioning

    * cond_dim: size of the conditioning vector fed into the block (should be the denosing iter + observations)
    """
    
    def __init__(self, c_in, c_out, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Conv1DBlock(c_in, c_out, kernel_size, n_groups),
            Conv1DBlock(c_out, c_out, kernel_size, n_groups)
        ])
        
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        # * double channels since FiLM needs two numbers per output channel to learn scale and bias functions
        cond_channels = c_out * 2 if cond_predict_scale else c_out 
        self.cond_predict_scale = cond_predict_scale
        self.c_out = c_out
        
        self.cond_encoder = nn.Sequential(
            nn.Mish(),  # activation function first since FiLM is applied to the activations 
            nn.Linear(cond_dim, cond_channels),
            Rearrange('b t -> b t 1')  # add a length dimension for broadcasting
        )
        
        # mathmatically equivalent to a linear layer if there's a mismatch in channels
        self.residual_conv = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
  
    def forward(self, x, cond):
        """
        x : [B x c_in x Horizon]
        cond : [B x cond_dim]
        
        returns : [B x c_out x Horizon]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.c_out, 1)  # [B x 2 x c_out x 1]. 1 for broadcasting
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias  # FiLM modulation
        else:
            out += embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)  # Residual connection
        return out 
