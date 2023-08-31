import math
import torch
import torch.nn as nn
from models.modules.blocks import ConvBlock, LinearBlock, ResBlock


class Discriminator(nn.Module):
    def __init__(
        self,
        disc_downs,
        n_res,
        c_in,
        c,
        norm="none",
        bias=True,
        w_norm="spectral",
        activ="lrelu",
        pad_type="zeros",
    ):
        super(Discriminator, self).__init__()
        self.model = nn.ModuleList()

        self.model.append(
            ConvBlock(
                c_in * 2,
                c,
                norm="none",
                activ=activ,
                bias=bias,
                w_norm=w_norm,
                pad_type=pad_type,
            )
        )
        for i in range(disc_downs):
            self.model.append(
                ConvBlock(
                    c,
                    c * 2,
                    norm=norm,
                    downsample=True,
                    activ=activ,
                    bias=bias,
                    w_norm=w_norm,
                    pad_type=pad_type,
                )
            )
            c *= 2

        for i in range(n_res):
            self.model.append(
                ResBlock(c, c, norm=norm, w_norm=w_norm, activ=activ, pad_type="zeros")
            )
        self.model.append(
            ConvBlock(
                c,
                1,
                norm="none",
                activ="sigmoid",
                bias=bias,
                w_norm=w_norm,
                pad_type=pad_type,
            )
        )

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x