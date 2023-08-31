import argparse
import os
import time
import numpy as np
from sconf.config import Config
from PIL import Image
from pathlib import Path

import torch
from torchvision import transforms, utils

from models.generator import Generator
from utils import pad_same_hw

@torch.no_grad()
def main(args, cfg):
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available()) else "cpu")

    gen = Generator(cfg["generator_config"]).to(device)

    gen.load_state_dict(
        torch.load(
            os.path.join(args.ckpt_dir, "CheckPoint", "latest_checkpoint.pt"),
            map_location=device,
        )["gen_state_dict"]
    )
    gen.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    save_dir = Path(f"{args.save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    iden = pad_same_hw(np.array(Image.open(args.iden_path).convert("RGB")))
    pose = pad_same_hw(np.array(Image.open(args.pose_path).convert("RGB")))
    iden = transform(iden).unsqueeze(0).to(device)
    pose = transform(pose).unsqueeze(0).to(device)

    start = time.time()
    oups, _, _ = gen(iden, pose)
    times = (time.time() - start)
    utils.save_image(utils.make_grid(torch.cat([iden, pose, oups]).detach().cpu()), save_dir / 'iden_pose_output.png')
    oups = (oups + 1) / 2
    utils.save_image(oups.detach().cpu(), save_dir / 'output.png')
    print(f"It takes {times:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", default="", type=str)
    parser.add_argument("--iden_path", default="", type=str)
    parser.add_argument("--pose_path", default="", type=str)
    parser.add_argument("--save_dir", default="Inference_result", type=str)
    parser.add_argument("--gpu", default=0)

    args, left_argv = parser.parse_known_args()
        
    cfg = Config(
        os.path.join(args.ckpt_dir, "0config.yaml"),
        default=None,
        colorize_modified_item=True,
    )
    main(args, cfg)
