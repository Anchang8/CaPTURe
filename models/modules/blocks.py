import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm


class ConvBlock(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="none",
        activ="relu",
        bias=True,
        upsample=False,
        downsample=False,
        w_norm="none",
        pad_type="zeros",
    ):
        super(ConvBlock, self).__init__()
        if norm == "in":
            self.norm = nn.InstanceNorm2d(C_out)
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(C_out)
        elif norm == "none":
            self.norm = nn.Identity()

        if activ == "relu":
            self.activ = nn.ReLU(inplace=True)
        elif activ == "lrelu":
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activ == "tanh":
            self.activ = nn.Tanh()
        elif activ == "sigmoid":
            self.activ = nn.Sigmoid()
        elif activ == "none":
            self.activ = nn.Identity()
        if upsample:
            kernel_size = 4
            stride = 2
            self.conv = nn.ConvTranspose2d(
                C_in,
                C_out,
                kernel_size,
                stride,
                padding,
                bias=bias,
                padding_mode=pad_type,
            )
        elif downsample:
            stride = 2
            self.conv = nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride,
                padding,
                bias=bias,
                padding_mode=pad_type,
            )
        else:
            self.conv = nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride,
                padding,
                bias=bias,
                padding_mode=pad_type,
            )

        if w_norm == "spectral":
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        norm="none",
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        upsample=False,
        downsample=False,
        w_norm="none",
        activ="relu",
        pad_type="zeros",
    ):
        super(ResBlock, self).__init__()

        self.downsample = downsample
        self.upsample = upsample

        self.conv1 = ConvBlock(
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            norm,
            activ,
            downsample=downsample,
            upsample=upsample,
            w_norm=w_norm,
            pad_type=pad_type,
            bias=bias,
        )
        self.conv2 = ConvBlock(
            C_out,
            C_out,
            kernel_size,
            stride,
            padding,
            norm,
            activ="none",
            w_norm=w_norm,
            pad_type=pad_type,
            bias=bias,
        )

        if C_in != C_out or upsample or downsample:
            if w_norm == "spectral":
                self.skip = spectral_norm((nn.Conv2d(C_in, C_out, 1, 1, 1)))
            elif w_norm == "none":
                self.skip = ConvBlock(C_in, C_out, 1, 1, 0)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)  

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)

        if hasattr(self, "skip"):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        out = (out + x) / math.sqrt(2)
        out = self.lrelu(out)
        return out


class LinearBlock(nn.Module):
    def __init__(
        self, C_in, C_out, bias=True, w_norm="none", norm="none", activ="relu"
    ):
        super(LinearBlock, self).__init__()
        if w_norm == "spectral":
            self.fc = SpectralNorm(nn.Linear(C_in, C_out, bias=bias))
        else:
            self.fc = nn.Linear(C_in, C_out, bias=bias)

        if norm == "bn":
            self.norm = nn.BatchNorm1d(C_out)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(C_out)
        elif norm == "ln":
            self.norm = nn.LayerNorm(C_out)
        elif norm == "none":
            self.norm = nn.Identity()

        if activ == "relu":
            self.activ = nn.ReLU(inplace=True)
        elif activ == "lrelu":
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activ == "tanh":
            self.activ = nn.Tanh()
        elif activ == "sigmoid":
            self.activ = nn.Sigmoid()
        elif activ == "none":
            self.activ = nn.Identity()

    def forward(self, x):
        out = self.fc(x)
        out = self.norm(out)
        out = self.activ(out)
        return out


def spectral_norm(module):
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.data.zero_()
    return nn.utils.spectral_norm(module)
