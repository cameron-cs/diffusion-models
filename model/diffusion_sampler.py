from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class DDIM(nn.Module):
    def __init__(self, model, beta: Tuple[float, float], T: int):
        super().__init__()
        self.model = model  # the trained denoising model
        self.T = T  # number of diffusion steps

        # precompute and register the beta values, alpha values, and their cumulative products
        beta_t = torch.linspace(beta[0], beta[1], T, dtype=torch.float32)
        alpha_t = 1.0 - beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0) # cumulative product of alphas

        self.register_buffer("alpha_t_bar", alpha_t_bar)
        self.register_buffer("one_minus_alpha_t_bar", 1 - alpha_t_bar)

    @torch.no_grad()
    def sample_one_step(self, x_t, t, prev_t, eta, labels):
        alpha_t = self.alpha_t_bar[t]
        alpha_t_prev = self.alpha_t_bar[prev_t]

        # get noise prediction from the model
        epsilon_theta_t = self.model(x_t, torch.tensor([t], device=x_t.device), labels)

        # compute noise scaling factors
        sigma_t = eta * torch.sqrt(self.one_minus_alpha_t_bar[prev_t] * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)

        sqrt_alpha_ratio = torch.sqrt(alpha_t_prev / alpha_t)
        sqrt_diff = torch.sqrt(self.one_minus_alpha_t_bar[prev_t] - sigma_t ** 2) - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)

        # compute the previous time step's sample based on noise predictions and scaling factors
        x_t_minus_one = sqrt_alpha_ratio * x_t + sqrt_diff * epsilon_theta_t + sigma_t * epsilon_t
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, labels, steps=1, method="linear", eta=0.0, only_return_x_0=True, interval=1):
        if method == "linear":
            a = max(1, self.T // steps)
            time_steps = np.arange(self.T - 1, -1, -a)
        elif method == "quadratic":
            time_steps = np.round(np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(int)
        else:
            raise NotImplementedError(f"Sampling method '{method}' is not implemented.")

        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        results = [x_t]
        for i, t in enumerate(time_steps):
            prev_t = time_steps_prev[i]
            # generate the sample for each time step
            x_t = self.sample_one_step(x_t, t, prev_t, eta, labels)

            # collect results at intervals or only return the final sample
            if not only_return_x_0 and (i % interval == 0 or i == len(time_steps) - 1):
                results.append(torch.clip(x_t, -1.0, 1.0))

        return x_t if only_return_x_0 else torch.stack(results, dim=1)