from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import math

from ...modules.diffusionmodules.sampling import VideoDDIMSampler, VPSDEDPMPP2MSampler
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS

# import rearrange
from einops import rearrange
import random
from sat import mpu


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="df",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips", 'df']

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss

def fourier_transform(x, balance=None):
    """
    Apply Fourier transform to the input tensor and separate it into low-frequency and high-frequency components.

    Args:
    x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
    balance (torch.Tensor or float, optional): Learnable balance parameter for adjusting the cutoff frequency.

    Returns:
    low_freq (torch.Tensor): Low-frequency components (with real and imaginary parts)
    high_freq (torch.Tensor): High-frequency components (with real and imaginary parts)
    """
    # Perform 2D Real Fourier transform (rfft2 only computes positive frequencies)
    x = x.to(torch.float32)
    fft_x = torch.fft.rfft2(x, dim=(-2, -1))
    
    # Calculate magnitude of frequency components
    magnitude = torch.abs(fft_x)

    # Set cutoff based on balance or default to the 80th percentile of the magnitude for low frequency
    if balance is None:
        # Downsample the magnitude to reduce computation for large tensors
        subsample_size = 10000  # Adjust based on available memory and tensor size
        if magnitude.numel() > subsample_size:
            # Randomly select a subset of values to approximate the quantile
            magnitude_sample = magnitude.flatten()[torch.randint(0, magnitude.numel(), (subsample_size,))]
            cutoff = torch.quantile(magnitude_sample, 0.8)  # 80th percentile for low frequency
        else:
            cutoff = torch.quantile(magnitude, 0.8)  # 80th percentile for low frequency
    else:
        # balance is clamped for safety and used to scale the mean-based cutoff
        cutoff = magnitude.mean() * (1 + 10 * balance)

    # Smooth mask using sigmoid to ensure gradients can pass through
    sharpness = 10  # A parameter to control the sharpness of the transition
    low_freq_mask = torch.sigmoid(sharpness * (cutoff - magnitude))
    
    # High-frequency mask can be derived from low-frequency mask (1 - low_freq_mask)
    high_freq_mask = 1 - low_freq_mask
    
    # Separate low and high frequencies using smooth masks
    low_freq = fft_x * low_freq_mask
    high_freq = fft_x * high_freq_mask

    # Return real and imaginary parts separately
    low_freq = torch.stack([low_freq.real, low_freq.imag], dim=-1)
    high_freq = torch.stack([high_freq.real, high_freq.imag], dim=-1)
    
    return low_freq, high_freq


def extract_frequencies(video: torch.Tensor, balance=None):
    """
    Extract high-frequency and low-frequency components of a video using Fourier transform.

    Args:
    video (torch.Tensor): Input video tensor of shape [batch_size, channels, frames, height, width]

    Returns:
    low_freq (torch.Tensor): Low-frequency components of the video
    high_freq (torch.Tensor): High-frequency components of the video
    """
    # batch_size, channels, frames, _, _ = video.shape
    video = rearrange(video, 'b c t h w -> (b t) c h w')  # Reshape for Fourier transform

    # Apply Fourier transform to each frame
    low_freq, high_freq = fourier_transform(video, balance=balance)

    return low_freq, high_freq

class SRDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch, hq_video=None, decode_first_stage=None):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        # Uncommnet for SR training
        noised_input = torch.cat((noised_input, batch['lq']), dim=2) # [B, T /4, 32, 60, 90]

        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        if self.type == "df":
            # print('idx:', idx)
            return self.get_loss(model_output, input, w, hq_video, idx, decode_first_stage)
        else:
            return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w, video_data=None, timesteps=None, decode_first_stage=None):    # model_output: x_hat_0;   target: x_0
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        elif self.type == "df":
            # v-prediction loss
            loss_v = torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
            with torch.no_grad():
                model_output = model_output.to(torch.bfloat16)
                model_output = model_output.permute(0, 2, 1, 3, 4).contiguous()
                pred_x0 = decode_first_stage(model_output)
            # print('pred_x0:', pred_x0.shape)   # [1, 3, 25, 480, 720]
            # print('video_data:', video_data.shape)  # [1, 3, 25, 480, 720]
            low_freq_pred_x0, high_freq_pred_x0 = extract_frequencies(pred_x0)
            low_freq_x0, high_freq_x0 = extract_frequencies(video_data)

            # timestep-aware loss
            alpha = 2
            ct = (timesteps/999) ** alpha
            loss_low = F.l1_loss(low_freq_pred_x0.float(), low_freq_x0.float(), reduction="mean")
            loss_high = F.l1_loss(high_freq_pred_x0.float(), high_freq_x0.float(), reduction="mean")
            loss_t = 0.01*(ct * loss_low + (1 - ct) * loss_high)

            beta = 1 # 1 is the default setting
            weight_t = 1 - timesteps/999
            loss = loss_v + beta * weight_t * loss_t
            # print('loss_v:', loss_v.mean().item(), 'loss_t:', (beta * weight_t * loss_t).mean().item())
            return loss