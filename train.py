import argparse
import os
from trainer import Trainer
from sconf import Config, dump_args
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from dataloader import get_dloader, get_val_dloader
from utils import rand_fix, str_to_bool
from models.generator import Generator
from models.discriminator import Discriminator
from models.modules.modules import weights_init


def setup_args_and_config():
    parser = argparse.ArgumentParser(description="argparse")
    parser.add_argument("config_path", nargs="*")
    parser.add_argument("--show", default="True", type=str_to_bool, help="Print the information of configures.")
    parser.add_argument("--gpu", default=0, help="GPU Number to train.")
    parser.add_argument("--dp", default="False", type=str_to_bool, help="Usage of DataParallel.")
    parser.add_argument("--dp_device", default="012", help="GPU numbers to use DataParallel.")

    args, left_argv = parser.parse_known_args()  
    cfg = Config(
        *args.config_path, default="defaults.yaml", colorize_modified_item=True
    )
    cfg.argv_update(left_argv)

    # Save yaml file to the checkpoint directory.
    save_dir = Path(
        f"{cfg['main_config']['result_path']}/{cfg['main_config']['result_title']}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/0config.yaml", "w") as f:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.dump(cfg.asdict(), f)
    return args, cfg

def load_checkpoint(path, gen, dis, g_optim, d_optim, device):
    checkpoint = torch.load(path, map_location=device)

    gen.load_state_dict(checkpoint["gen_state_dict"])
    g_optim.load_state_dict(checkpoint["gen_opt"])

    if dis is not None:
        dis.load_state_dict(checkpoint["disc_state_dict"])
        d_optim.load_state_dict(checkpoint["disc_opt"])

    epoch = checkpoint["epoch"]

    return gen, dis, g_optim, d_optim, epoch

def build_model_optim(cfg, args):
    gen_cfg = cfg["generator_config"]
    dis_cfg = cfg["discriminator_config"]
    main_cfg = cfg["main_config"]
    train_cfg = cfg["train_config"]
    latest_ckpt_path = Path(f"{cfg['main_config']['result_path']}/{cfg['main_config']['result_title']}") / 'CheckPoint' / 'latest_checkpoint.pt'

    dp = args.dp
    dp_device = list(args.dp_device)
    dp_device = list(map(lambda x: int(x), dp_device))

    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available()) else "cpu")

    gen = Generator(gen_cfg).to(device)
    dis = Discriminator(**dis_cfg).to(device)

    gen.apply(weights_init(main_cfg["init"]))
    dis.apply(weights_init(main_cfg["init"]))

    g_optim = optim.Adam(
        [
            {
                "params": gen.idenity_encoder.parameters(),
                "lr": train_cfg["lr_G"],
                "betas": train_cfg["adam_betas"],
            },
            {
                "params": gen.pose_encoder.parameters(),
                "lr": train_cfg["lr_G"],
                "betas": train_cfg["adam_betas"],
            },
            {
                "params": gen.decoder.parameters(),
                "lr": train_cfg["lr_G"],
                "betas": train_cfg["adam_betas"],
            },
        ],
        lr=train_cfg["lr_G"],
        betas=train_cfg["adam_betas"],
    )
    d_optim = optim.Adam(
        dis.parameters(), lr=train_cfg["lr_D"], betas=train_cfg["adam_betas"]
    )
    
    if dp:
        gen = nn.DataParallel(gen, device_ids=dp_device)
        dis = nn.DataParallel(dis, device_ids=dp_device)
    if os.path.exists(latest_ckpt_path):
        gen, dis, g_optim, d_optim, epoch = load_checkpoint(
            latest_ckpt_path, gen, dis, g_optim, d_optim, device
        )
    else:
        epoch = 0
    return gen, dis, g_optim, d_optim, epoch


def main():
    args, cfg = setup_args_and_config()

    if args.show:
        print("Changed Informations")
        s = dump_args(args)
        print(s + "\n")
        print(cfg.dumps())

    # Fix the random seed
    rand_fix(cfg["train_config"]["seed"])

    # Set up transform
    size = (256, 256)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transform
    # Setup dataset
    dloader = get_dloader(transform, cfg)
    i_val_dloader = get_val_dloader(transform_test, False, cfg)
    p_val_dloader = get_val_dloader(transform_test, False, cfg)

    ##########################
    # Build Model & Optimizer
    ##########################
    gen, dis, g_optim, d_optim, epoch = build_model_optim(cfg, args)
    print("\nComplete model building")

    ##########################
    # Start Training
    ##########################
    trainer = Trainer(
        gen,
        dis,
        g_optim,
        d_optim,
        dloader,
        i_val_dloader,
        p_val_dloader,
        transform,
        cfg,
        args,
    )
    trainer.train(epoch)


if __name__ == "__main__":
    main()
