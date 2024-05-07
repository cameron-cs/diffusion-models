import math
import torch
from torch import nn

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2                                            # divide the dimension by 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
                                                               # compute frequency using exponential and log functions
    args = timesteps[:, None].float() * freqs.to(timesteps.device)[None]
                                                               # compute arguments for cosine and sine
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                                                               # concatenate cosine and sine values
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
                                                               # add zeros if dimension is odd
    return embedding

def norm_layer(channels):
    return nn.GroupNorm(32, channels)