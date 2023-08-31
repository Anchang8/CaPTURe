import random
import numpy as np
import datetime
import os
import time
from pathlib import Path

import torch
import torchvision.utils as utils

def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")

def rand_fix(random_seed=777):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_checkpoint(state, dirpath, epoch, interval=1, minimum_epoch=0, latest=False):
    save_dir = dirpath / "CheckPoint"
    save_dir.mkdir(parents=True, exist_ok=True)
    if latest:
        filename = "latest_checkpoint.pt"
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(state, checkpoint_path)
        print("--- checkpoint saved to " + str(checkpoint_path) + " ---")
    elif epoch % interval == 0 and epoch > minimum_epoch:
        filename = f"{epoch}epoch_checkpoint.pt"
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(state, checkpoint_path)
        print("--- checkpoint saved to " + str(checkpoint_path) + " ---")

def present_time():
    now = datetime.datetime.now()
    nowDatetime = now.strftime("%Y-%m-%d %H:%M:%S")
    return nowDatetime

def save_train_result(gen, x, y, epoch, main_cfg):
    gen.eval()
    result = []
    with torch.no_grad():
        output, _, _ = gen(x, y)
        output2, _, _ = gen(y, x)
        recon_oup, _, _ = gen(x, x)
        
        # Unnormalized the output of the generator(Tanh).
        output = (output + 1) / 2
        output2 = (output2 + 1) / 2
        recon_oup = (recon_oup + 1) / 2
        x = (x + 1) / 2
        y = (y + 1) / 2
        
    for b in range(output.size(0)):
        result.extend([x[b].cpu(), y[b].cpu(), output[b].cpu()])
    for b in range(output2.size(0)):
        result.extend([y[b].cpu(), x[b].cpu(), output2[b].cpu()])
    for b in range(recon_oup.size(0)):
        result.extend([x[b].cpu(), x[b].cpu(), recon_oup[b].cpu()])
    result_img = utils.make_grid(result, padding=2, nrow=3)

    save_dir = Path(f"{main_cfg['result_path']}/{main_cfg['result_title']}/train_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{epoch}epoch.png"

    utils.save_image(result_img, save_path)


def save_test_result(gen, dloader, epoch, cfg, args):
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available()) else "cpu")

    gen.eval()

    result = []
    for sample_dict in dloader:
        iden = sample_dict["sample1"].to(device)
        pose = sample_dict["sample2"].to(device)
        break
    oup1, _, _ = gen(iden, pose)
    oup2, _, _ = gen(pose, iden)
    
    # Unnormalized the output of the generator(Tanh).
    pose = (pose + 1) / 2
    iden = (iden + 1) / 2
    oup1 = (oup1 + 1) / 2
    oup2 = (oup2 + 1) / 2
    for i in range(iden.size(0)):
        result.extend(
            [iden[i].cpu().detach(), pose[i].cpu().detach(), oup1[i].cpu().detach()]
        )
    for i in range(iden.size(0)):
        result.extend(
            [pose[i].cpu().detach(), iden[i].cpu().detach(), oup2[i].cpu().detach()]
        )

    result_img = utils.make_grid(result, padding=2, nrow=3)

    save_dir = Path(f"{cfg['result_path']}/{cfg['result_title']}/test_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{epoch}epoch.png"

    utils.save_image(result_img, save_path)

def print_about_train_time(epoch_start, epoch, cfg):
    print("=" * 100)
    running_time = time.time() - epoch_start
    print(
        "Time taken by 1epoch: {:.0f}h {:.0f}m {:.0f}s".format(
            ((running_time) // 60) // 60,
            ((running_time) // 60) % 60,
            (running_time) % 60,
        )
    )
    remain_running_time = (cfg["num_epochs"] - epoch) * running_time
    print(
        "Remaining training time is : {:.0f}h {:.0f}m {:.0f}s".format(
            ((remain_running_time) // 60) // 60,
            ((remain_running_time) // 60) % 60,
            (remain_running_time) % 60,
        )
    )

def print_train_finish(cfg, train_start_time):
    print("Training is finished")
    print(
        "Time taken by {}epochs: {:.0f}h {:.0f}m {:.0f}s".format(
            cfg["num_epochs"],
            ((time.time() - train_start_time) // 60) // 60,
            ((time.time() - train_start_time) // 60) % 60,
            (time.time() - train_start_time) % 60,
        )
    )
    
def pad_same_hw(img):
        """
        Make the image has the same width and height.
        """
        # input img shape H x W x C
        H, W, _ = img.shape
        img = np.transpose(img, (2,0,1)) # Make C x H x W
        for i, ch_img in enumerate(img):
            if H > W:
                diff = H - W
                left = diff // 2
                right = diff - left
                pad_img = np.pad(ch_img, ((0,0), (left,right)), constant_values=255)
            elif W > H:
                diff = W - H
                top = diff // 2
                bottom = diff - top
                pad_img = np.pad(ch_img, ((top,bottom), (0,0)), constant_values=255)
            else:
                pad_img = ch_img
            if i == 0:
                pad_imgs = np.expand_dims(pad_img, axis=0)
            else:
                pad_imgs = np.vstack([pad_imgs, np.expand_dims(pad_img,axis=0)])
        pad_imgs = np.transpose(pad_imgs, (1,2,0))
        return pad_imgs