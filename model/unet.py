import torch
from torch import nn

from model.diffusion_blocks import TimestepEmbedSequential, FusedResidualBlock, OptimizedAttentionBlock, Downsample, \
    Upsample
from model.utils import norm_layer, timestep_embedding


# UNet architecture with conditional generation
class ConditionalUNet(nn.Module):
    def __init__(
            self,
            in_channels=3,                                    # input channels (RGB images)
            model_channels=128,                               # base number of channels
            out_channels=3,                                   # output channels (RGB images)
            num_res_blocks=2,                                 # number of residual blocks per level
            attention_resolutions=(8, 16),                    # resolutions for attention
            dropout=0,                                        # dropout rate
            channel_mult=(1, 2, 2, 2),                        # multipliers for channel counts at each level
            conv_resample=True,                               # whether to use convolution for up/down sampling
            num_heads=4,                                      # number of attention heads
            num_classes=10                                    # number of classes for conditional generation
    ):
        super().__init__()
        self.in_channels = in_channels                        # set input channels
        self.model_channels = model_channels                  # set model channels
        self.out_channels = out_channels                      # set output channels
        self.num_res_blocks = num_res_blocks                  # set number of residual blocks
        self.attention_resolutions = attention_resolutions    # set attention resolutions
        self.dropout = dropout                                # set dropout rate
        self.channel_mult = channel_mult                      # set channel multipliers
        self.conv_resample = conv_resample                    # set conv_resample
        self.num_heads = num_heads                            # set number of heads
        self.num_classes = num_classes                        # set number of classes

        # add an embedding layer for the class labels
        self.class_embedding = nn.Embedding(num_classes, model_channels * 4)

        # create the timestep embedding layers
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4)
        )

        # define the downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
            )
        )
        ch = model_channels                                  # initialize the current number of channels
        ds = 1                                               # initialize downsampling factor

        down_block_chans = [model_channels]                  # keep track of channel counts at each level
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    TimestepEmbedSequential(
                        FusedResidualBlock(ch, mult * model_channels, model_channels * 4, dropout),
                        OptimizedAttentionBlock(mult * model_channels, num_heads=num_heads) if ds in attention_resolutions else nn.Identity()
                    )
                )
                down_block_chans.append(mult * model_channels)
                ch = mult * model_channels
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample)
                    )
                )
                down_block_chans.append(ch)
                ds *= 2

        # define the middle block
        self.middle_block = TimestepEmbedSequential(
            FusedResidualBlock(ch, ch, model_channels * 4, dropout),
            OptimizedAttentionBlock(ch, num_heads=num_heads),
            FusedResidualBlock(ch, ch, model_channels * 4, dropout)
        )

        # define the upsampling blocks
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(TimestepEmbedSequential(
                    FusedResidualBlock(ch + down_block_chans.pop(), model_channels * mult, model_channels * 4, dropout),
                    OptimizedAttentionBlock(model_channels * mult, num_heads=num_heads) if ds in attention_resolutions else nn.Identity(),
                    Upsample(model_channels * mult, conv_resample) if level and i == num_res_blocks else nn.Identity()
                ))
                ch = model_channels * mult
                ds //= 2

        # define the output layers
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, timesteps, labels):
        hs = []                                              # store the intermediate states
        # embed the timestep and the class label
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        class_emb = self.class_embedding(labels)
        emb = emb + class_emb

        for module in self.down_blocks:
            x = module(x, emb)                               # apply downsampling blocks
            hs.append(x)
        x = self.middle_block(x, emb)                        # apply middle block
        for module in self.up_blocks:
            x = module(torch.cat([x, hs.pop()], dim=1), emb) # apply upsampling blocks
        return self.out(x)                                   # apply output layers and return generated