from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[float, float], T: int):
        super().__init__()
        self.model = model  # the denoising model being trained
        self.T = T  # number of diffusion steps

        # precompute and store beta, alpha, and their cumulative products
        self.register_buffer("beta_t", torch.linspace(beta[0], beta[1], T, dtype=torch.float32))
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0) # cumulative product of alphas

        # precompute signal and noise rates for each time step
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, labels):
        batch_size = x_0.shape[0]
        device = x_0.device

        # efficiently generate random training steps and noise
        t = torch.randint(0, self.T, (batch_size,), device=device)
        epsilon = torch.randn_like(x_0) # random noise

        # use broadcasting for signal and noise rates
        # apply diffusion to the input images using the signal and noise rates for the selected time steps
        signal_rate_t = self.signal_rate[t][:, None, None, None]
        noise_rate_t = self.noise_rate[t][:, None, None, None]
        x_t = signal_rate_t * x_0 + noise_rate_t * epsilon

        # pass labels to the model
        # pass the noisy images and the labels to the model to predict the noise
        epsilon_theta = self.model(x_t, t, labels)
        return F.mse_loss(epsilon_theta, epsilon)