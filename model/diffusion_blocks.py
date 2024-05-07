import math
import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod

from model.utils import norm_layer, timestep_embedding


class TimestepBlockABC(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, emb):
        pass


# sequential block for timestep embeddings
class TimestepEmbedSequential(nn.Sequential, TimestepBlockABC):
    def forward(self, x, emb):
        for layer in self:                                     # iterate through layers
            x = layer(x, emb) if isinstance(layer, TimestepBlockABC) else layer(x)
                                                               # Apply timestep embedding or standard forward
        return x


# fused residual block with timestep embeddings
class FusedResidualBlock(TimestepBlockABC):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                                                               # first convolution layer
        self.norm1 = norm_layer(out_channels)                  # first normalization layer
        self.activation = nn.SiLU()                            # SiLU activation function
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                                                               # second convolution layer
        self.norm2 = norm_layer(out_channels)                  # second normalization layer
        self.dropout = nn.Dropout(dropout)                     # dropout layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                        if in_channels != out_channels else nn.Identity()
                                                               # shortcut connection

        self.time_emb = nn.Sequential(
            nn.SiLU(),                                         # SiLU activation
            nn.Linear(time_channels, out_channels)             # linear layer for timestep embedding
        )

    def forward(self, x, t):
        h = self.conv1(x)                                      # apply first convolution
        h = self.norm1(h)                                      # apply normalization
        h = self.activation(h)                                 # apply activation
        h += self.time_emb(t)[:, :, None, None]                # add timestep embedding
        h = self.conv2(h)                                      # apply second convolution
        h = self.norm2(h)                                      # apply second normalization
        h = self.activation(h)                                 # apply activation
        h = self.dropout(h)                                    # apply dropout
        return h + self.shortcut(x)                            # add shortcut and return


# attention block for self-attention mechanism
class OptimizedAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads                            # number of heads for multi-head attention
        self.head_dim = channels // num_heads                 # dimension per head
        self.scale = 1.0 / (self.head_dim ** 0.5)             # scaling factor

        self.norm = norm_layer(channels)                      # normalization layer
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
                                                              # convolution for Q, K, V
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
                                                              # convolution for output projection

    def forward(self, x):
        B, C, H, W = x.shape                                  # extract shape parameters
        x = self.norm(x)                                      # apply normalization
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
                                                              # reshape for multi-head attention
        q, k, v = qkv.unbind(1)                               # split Q, K, V

        q = q.permute(0, 2, 3, 1).reshape(B * self.num_heads, H * W, self.head_dim)
                                                              # reshape Q
        k = k.permute(0, 2, 3, 1).reshape(B * self.num_heads, H * W, self.head_dim)
                                                              # reshape K
        v = v.permute(0, 2, 3, 1).reshape(B * self.num_heads, H * W, self.head_dim)
                                                              # reshape V

        attn = (torch.bmm(q, k.transpose(1, 2)) * self.scale).softmax(dim=-1)
                                                              # compute attention
        h = torch.bmm(attn, v)                                # apply attention to V

        h = h.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
                                                              # reshape output
        return self.proj(h) + x                               # apply projection and add residual connection

# upsample block to upscale feature maps
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv                              # to determine whether to use convolution
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1) if use_conv else None
                                                              # convolution layer if use_conv is True

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # upsample using nearest neighbor interpolation
        return self.conv(x) if self.use_conv else x           # apply convolution if use_conv is True

# downsample block to downscale feature maps
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1) \
                   if use_conv else nn.AvgPool2d(kernel_size=2)
                                                              # convolution or average pooling layer

    def forward(self, x):
        return self.op(x)                                     # apply downsampling
