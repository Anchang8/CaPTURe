import torch
import torch.nn as nn

from models.modules.modules import AdaIN, AttnModule

from .modules.blocks import ConvBlock, ResBlock


class Decoder(nn.Module):
    def __init__(
        self,
        c_in,
        ups,
        min_ch,
        res2=False,
        t=None,
        dec_attn_layers=[],
        dec_adain_layers=[],
    ):
        super(Decoder, self).__init__()

        self.adain_layer = dec_adain_layers
        self.dec_attn_layers = dec_attn_layers
        self.t = t
        self.c_in = c_in
        ##TODO: 지우기
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        
        
        self.attnModule_256 = nn.ModuleDict()
        for i in dec_attn_layers:
            if i == 3:
                attnModule_c_in = 128
            elif i == 4:
                attnModule_c_in = 64
            else:
                attnModule_c_in = min_ch
            self.attnModule_256[f"{i}"] = AttnModule(attnModule_c_in, t=t)

        if res2:
            self.code_to_feat = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBlock(min_ch * 8, min_ch * 4, activ="lrelu"),
            )
        else:
            self.code_to_feat = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBlock(min_ch * 8, min_ch * 4, activ="lrelu"),
                nn.Upsample(scale_factor=2),
            )

        if res2:
            self.upBlock = nn.ModuleList(
                [
                    nn.Upsample(scale_factor=2),
                    ConvBlock(min_ch * 8, min_ch * 8, activ="lrelu"),
                    nn.Upsample(scale_factor=2),
                    ConvBlock(min_ch * 8, min_ch * 4, activ="lrelu"),
                    nn.Upsample(scale_factor=2),
                    ConvBlock(min_ch * 4, min_ch * 2, activ="lrelu"),
                    ConvBlock(min_ch * 2, min_ch * 1, activ="lrelu"),
                ]
            )
        else:
            self.upBlock = nn.ModuleList(
                [
                    nn.Upsample(scale_factor=2),
                    ConvBlock(min_ch * 8, min_ch * 8, activ="lrelu", norm="in"),
                    nn.Upsample(scale_factor=2),  
                    ConvBlock(min_ch * 8, min_ch * 4, activ="lrelu", norm="in"),
                    ConvBlock(min_ch * 4, min_ch * 2, activ="lrelu", norm="in"),
                    ConvBlock(min_ch * 2, min_ch * 1, activ="lrelu", norm="in"),
                ]
            )

        self.model = nn.ModuleList()

        for i in range(ups):
            if c_in == min_ch and i < ups - 2:
                c_out = min_ch
            else:
                c_out = c_in // 2
            self.model.append(ResBlock(c_in, c_out, upsample=True, activ="lrelu"))
            if c_in == min_ch and i < ups - 2:
                pass
            else:
                c_in //= 2
        c_in = c_out
        self.toRGB = ConvBlock(c_in, 3, activ="tanh")

    def forward(self, iden_feats, pose_feats):
        iden = iden_feats[0].unsqueeze(2).unsqueeze(3)
        n = torch.randn_like(iden)
        iden = torch.cat([iden, n], dim=1) # Concatenate the identity vector with gaussian noise.

        iden = self.code_to_feat(iden) # Make the identity vector to the feature which has a resolution.
        x = torch.cat([pose_feats[0], iden], dim=1,)
        for i, layer in enumerate(self.upBlock):
            x = layer(x)
        for i, layer in enumerate(self.model):
            if i in self.dec_attn_layers:
                x = self.attnModule_256[
                    f"{i}"
                ](
                    x,
                    iden_feats[5 - i],
                    pose_feats[5 - i],
                )
            if i in self.adain_layer:
                x = AdaIN(iden_feats[5 - i], x)
            x = layer(x)
        x = self.toRGB(x)
        return x