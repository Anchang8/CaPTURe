import torch.nn as nn
import torch

from .blocks import ConvBlock, LinearBlock


def weights_init(init_type="default"):
    """
    Adopted and modified from FUNIT
    Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
    Licensed under the CC BY-NC-SA 4.0 license
    (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
    """

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            if init_type == "gaussian":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=2 ** 0.5)
            elif init_type == "kaiming":  # naver는 kaiming 씀.
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_int")
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization : {}".format(init_type)

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    return init_fun


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def masked_calc_mean_std(feat, attention, eps=1e-9):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]

    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    feat_mean = torch.div(
        (feat_mean * size[2] * size[3]),
        (torch.sum(attention, dim=-1).view(N, C, 1, 1)),
    )
    feat_var = (
        torch.sum(((feat - feat_mean.expand(size)).view(N, C, -1) ** 2), dim=-1) + eps
    )
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    return feat_mean, feat_std


class AdaIN_code(nn.Module):
    def __init__(self, c_in):
        super(AdaIN_code, self).__init__()
        self.c_in = c_in

        self.downTo_idenVector = nn.Sequential(
            ConvBlock(c_in, c_in * 2, downsample=True, activ="lrelu"),
            ConvBlock(c_in * 2, c_in * 2, downsample=True, activ="lrelu"),
        )
        self.downTo_poseVector = nn.Sequential(
            ConvBlock(c_in, c_in * 2, downsample=True, activ="lrelu"),
            ConvBlock(c_in * 2, c_in * 2, downsample=True, activ="lrelu"),
        )
        self.idenVector_fc = nn.Sequential(
            LinearBlock(c_in * 2, c_in * 2, activ="lrelu"),
        )
        self.poseVector_fc = nn.Sequential(
            LinearBlock(c_in * 2, c_in * 2, activ="lrelu"),
        )
        self.paramVector_fc = nn.Sequential(
            LinearBlock(c_in * 2, c_in * 2, activ="lrelu"),
        )

    def forward(self, iden, pose):
        b, c, _, _ = iden.size()
        iden_vector = self.downTo_idenVector(iden)
        iden_vector = iden_vector.mean(dim=(2, 3))
        iden_vector = self.idenVector_fc(iden_vector)

        pose_vector = self.downTo_poseVector(pose)
        pose_vector = pose_vector.mean(dim=(2, 3))
        pose_vector = self.poseVector_fc(pose_vector)

        mean_std = torch.mul(iden_vector, pose_vector)
        mean_std = self.paramVector_fc(mean_std)
        target_mean, target_std = (
            mean_std[:, : self.c_in].view(b, c, 1, 1),
            mean_std[:, self.c_in :].view(b, c, 1, 1),
        )

        pose_mean, pose_std = calc_mean_std(pose)
        normalized_pose = (pose - pose_mean.expand(pose.size())) / pose_std.expand(
            pose.size()
        )
        adained_pose = normalized_pose * target_std.expand(
            normalized_pose.size()
        ) + target_mean.expand(normalized_pose.size())

        return adained_pose


class AttnModule(nn.Module):
    def __init__(self, c_in, t=None):
        super(AttnModule, self).__init__()
        self.t = t
        self.c_in = c_in

        self.query_conv = nn.Conv2d(c_in, c_in // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(c_in, c_in // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(c_in, c_in, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        ##TODO: 지우기
        self.value_conv2 = nn.Conv2d(c_in, c_in, kernel_size=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

        self.conv_comp = ConvBlock(c_in * 2, c_in, activ="lrelu")

        self.beta = nn.Parameter(torch.ones(1, requires_grad=True))
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(
        self, inp, x, y
    ):  # x is the identity feature, y is the pose feature
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, w * h)
        proj_key = self.key_conv(y).view(b, -1, w * h)
        proj_value = self.value_conv(x).view(b, -1, w * h)

        if not self.t:
            self.t = proj_query.size(1) ** 0.5

        energy_idenRow = torch.bmm(proj_query.permute(0, 2, 1), proj_key) / self.t
        attention_idenRow = self.softmax(energy_idenRow)
        proj_query_poseRow = self.query_conv(y).view(b, -1, w * h)
        proj_key_poseRow = self.key_conv(x).view(b, -1, w * h)
        energy_poseRow = (
            torch.bmm(proj_query_poseRow.permute(0, 2, 1), proj_key_poseRow) / self.t
        )
        attention_poseRow = self.softmax(energy_poseRow)

        x_update = torch.bmm(proj_value, (attention_idenRow).permute(0, 2, 1))
        x_update = x_update.view(b, c, h, w)

        x_update = torch.add((1 - self.beta) * x, self.beta * x_update)
        target_mean, target_std = calc_mean_std(x)

        # input AdaIN with caculated Params
        inp_mean, inp_std = calc_mean_std(inp)
        normalized_inp = (inp - inp_mean.expand(inp.size())) / inp_std.expand(
            inp.size()
        )
        adained_inp = normalized_inp * target_std.expand(
            normalized_inp.size()
        ) + target_mean.expand(inp.size())

        # BMM with adained input & inversed attention mask
        permuted_adained_inp = adained_inp.view(b, -1, w * h)
        reverse_attentioned_adained_inp = torch.bmm(
            permuted_adained_inp.view(b, -1, w * h),
            (1 - attention_poseRow).permute(0, 2, 1),
        ).view(b, c, h, w)
        reverse_attentioned_adained_inp = torch.add(
            (1 - self.alpha) * reverse_attentioned_adained_inp,
            self.alpha * adained_inp,
        )
        out = self.conv_comp(
            torch.cat([x_update, reverse_attentioned_adained_inp], dim=1)
        )
        return out

def AdaIN(iden, pose):
    assert iden.size()[:2] == pose.size()[:2]
    size = pose.size()
    iden_mean, iden_std = calc_mean_std(iden)
    pose_mean, pose_std = calc_mean_std(pose)

    normalized_feat = (pose - pose_mean.expand(size)) / pose_std.expand(size)
    return normalized_feat * iden_std.expand(size) + iden_mean.expand(size)
