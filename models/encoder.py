import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import ConvBlock, LinearBlock, ResBlock

from models.modules.modules import AdaIN, AttnModule


class IdentityEncoder(nn.Module):
    def __init__(self, downs, c_in, c, min_ch, in_norm, pad_type="zeros"):
        super(IdentityEncoder, self).__init__()
        if in_norm:
            norm = "in"
        else:
            norm = "none"
        self.conv_layer = ConvBlock(
            c_in, c, activ="lrelu", norm=norm, pad_type=pad_type
        )

        self.res_down = nn.ModuleList()
        for i in range(downs):
            c_out = min(c * 2, min_ch)
            self.res_down.append(
                ResBlock(
                    c,
                    c_out,
                    downsample=True,
                    activ="lrelu",
                    norm=norm,
                    pad_type=pad_type,
                )
            )
            c = min(c * 2, min_ch)
        c_in = c
        self.make_code = nn.ModuleList()
        self.make_code.append(
            ConvBlock(c_in, c_in * 2, downsample=True, norm=norm, pad_type=pad_type),
        )
        self.make_code.append(
            ConvBlock(
                c_in * 2, c_in * 4, downsample=True, norm=norm, pad_type=pad_type
            ),
        )
        c_in *= 4
        self.code_linear_layer = nn.Sequential(
            LinearBlock(c_in, c_in * 2, activ="lrelu")
        )

    def forward(self, x):
        x = self.conv_layer(x)

        features = dict()
        i = 1

        for layer in self.res_down:
            features[i] = x
            i += 1
            x = layer(x)
        for layer in self.make_code:
            features[i] = x
            i += 1
            x = layer(x)
        features[i] = x
        x = x.mean(dim=(2, 3))

        for layer in self.code_linear_layer:
            i += 1
            x = layer(x)
        features[i] = x
        
        # The half of identity features are mean and the other are std.
        samp_mean = x[:, : x.size(1) // 2]
        samp_std = x[:, x.size(1) // 2 :]
        iden_vector = torch.add(torch.randn_like(samp_mean), samp_mean)
        iden_vector = torch.mul(iden_vector, samp_std)
        features[0] = iden_vector
        
        """
        The identity feature's size
        features[idx]:
        1 torch.Size([4, 64, 256, 256])
        2 torch.Size([4, 128, 128, 128])
        3 torch.Size([4, 256, 64, 64])
        4 torch.Size([4, 256, 32, 32])
        5 torch.Size([4, 256, 16, 16]) -> Insert to the attention module.
        6 torch.Size([4, 512, 8, 8])
        7 torch.Size([4, 1024, 4, 4])
        0 torch.Size([4, 1024])
        """
        return features

class PoseEncoder(nn.Module):
    def __init__(
        self,
        downs,
        c_in,
        c,
        min_ch,
        in_norm,
        pad_type,
        res2=False,
        conv_num=0,
        enc_adain_layers=[],
        enc_attn_adain_layers=[],
        enc_attn_layers=[],
    ):
        super(PoseEncoder, self).__init__()
        if enc_adain_layers == []:
            self.adain_layer = [-1]
        else:
            self.adain_layer = enc_adain_layers

        if enc_attn_layers == []:
            self.enc_attn_layers = [-1]
        else:
            self.enc_attn_layers = enc_attn_layers

        self.attnModule = nn.ModuleDict()
        self.enc_attn_adain_layers = enc_attn_adain_layers
        for i in enc_attn_layers:
            if i == 1:
                attnModule_c_in = 64
            elif i == 2:
                attnModule_c_in = 128
            else:
                attnModule_c_in = min_ch
            if i > -1:
                self.attnModule[f"{i}"] = AttnModule(attnModule_c_in)
        if in_norm:
            norm = "in"
        else:
            norm = "none"
        self.conv_layer = ConvBlock(
            c_in, c, activ="lrelu", norm=norm, pad_type=pad_type
        )

        self.res_down = nn.ModuleList()
        for i in range(downs):
            c_out = min(c * 2, min_ch)
            self.res_down.append(
                ResBlock(
                    c,
                    c_out,
                    downsample=True,
                    activ="lrelu",
                    norm=norm,
                    pad_type=pad_type,
                )
            )
            c = min(c * 2, min_ch)

        c_in = c
        self.make_code = nn.ModuleList()
        self.make_code.append(
            ConvBlock(c_in, c_in * 2, downsample=True, norm=norm, pad_type=pad_type)
        )
        self.make_code.append(
            ConvBlock(c_in * 2, c_in * 4, downsample=True, norm=norm, pad_type=pad_type)
        ) 
        if res2:
            self.make_code.append(
                ConvBlock(
                    c_in * 4, c_in * 4, downsample=True, norm=norm, pad_type=pad_type
                )
            )
        else:
            self.make_code.append(
                ConvBlock(
                    c_in * 4, c_in * 4, downsample=False, norm=norm, pad_type=pad_type
                )
            )

    def forward(self, x, iden_feats):
        x = self.conv_layer(x)
        features = dict()
        i = 1
        for layer in self.res_down:
            if i in self.enc_attn_layers:
                x = self.attnModule[f"{i}"](x, iden_feats[i], x)
            if i in self.adain_layer:
                x = AdaIN(iden_feats[i], x)
            features[i] = x
            i += 1
            x = layer(x)

        for layer in self.make_code:
            if i in self.enc_attn_layers:
                x = self.attnModule[f"{i}"](x, iden_feats[i], x)
            if i in self.adain_layer:
                x = AdaIN(iden_feats[i], x)
            features[i] = x
            i += 1
            x = layer(x)
        if i in self.adain_layer:
            x = AdaIN(iden_feats[i], x)

        features[0] = x
        
        """
        The pose feature's size
        features[idx]:
        1 torch.Size([b, 64, 256, 256])
        2 torch.Size([b, 128, 128, 128])
        3 torch.Size([b, 256, 64, 64])
        4 torch.Size([b, 256, 32, 32])
        5 torch.Size([b, 256, 16, 16])
        6 torch.Size([b, 512, 8, 8])
        7 torch.Size([b, 1024, 4, 4])
        8 torch.Size([b, 1024, 1, 1])
        9 torch.Size([b, 2048, 1, 1])
        """
        return features