import torch.nn as nn

from .decoder import Decoder
from .encoder import PoseEncoder, IdentityEncoder


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        poseE_cfg = cfg["poseE_config"]
        idenE_cfg = cfg["idenE_config"]
        dec_cfg = cfg["dec_config"]

        self.idenity_encoder = IdentityEncoder(**idenE_cfg)
        self.pose_encoder = PoseEncoder(**poseE_cfg)
        dec_cIn = min(idenE_cfg["c"] * (2 ** idenE_cfg["downs"]), dec_cfg["min_ch"])
        self.decoder = Decoder(c_in=dec_cIn, ups=idenE_cfg["downs"], **dec_cfg)

    def forward(self, iden, pose):
        iden_feats = self.idenity_encoder(iden)
        pose_feats = self.pose_encoder(pose, iden_feats)
        out = self.decoder(
            iden_feats, pose_feats
        )
        return (
            out,
            pose_feats,
            iden_feats,
        )
