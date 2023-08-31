import os
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from utils import pad_same_hw

def get_dloader(transform, cfg):
    dset_cfg = cfg["dataset_config"]
    if 'fashion' in dset_cfg.dataset_path.lower():
        dataset = FashionDataset(cfg, transform)
    else:
        dataset = MaximoDataset(cfg, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=dset_cfg["batch_size"],
        shuffle=True,
        num_workers=dset_cfg["num_workers"],
        pin_memory=True,
    )
    return dataloader

def get_val_dloader(transform, shuffle, cfg):
    dset_cfg = cfg["dataset_config"]
    if 'fashion' in dset_cfg.dataset_path:
        dataset = FashionDataset(cfg, transform, test=True)
    else:
        dataset = MaximoDataset(cfg, transform, test=True)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=shuffle, # For visualize random validation sets' results.
        num_workers=dset_cfg["num_workers"],
        pin_memory=True,
    )
    return dataloader

class MaximoDataset(Dataset):
    def __init__(self, cfg, transform=None, test=False):
        self.cfg = cfg
        self.transform = transform
        self.test = test
        if self.test:
            dataset_path = "test"
        else:
            dataset_path = "train"
        self.dataset_dir = (
            Path(cfg["dataset_config"]["dataset_path"]) / dataset_path
        )
        self.file_list = os.listdir(self.dataset_dir)
    def __len__(self):
        if self.test:
            length = len(self.file_list)
        else:
            length = 4000
        return length
    def __getitem__(self, idx):
        if self.test:
            # The test dataset consists of images.
            sample1, sample2 = random.sample(self.file_list, 2)
            sample1 = Image.open(os.path.join(self.dataset_dir, sample1)).convert("RGB")
            sample2 = Image.open(os.path.join(self.dataset_dir, sample2)).convert("RGB")

            if self.transform:
                sample1 = self.transform(sample1)
                sample2 = self.transform(sample2)
            return {"sample1": sample1, "sample2": sample2}
        else:
            # The train dataset consists of directories of each character.
            # Sampling Identity Encoder's input
            iEnc_sample_character, pEnc_sample_character = random.sample(
                self.file_list, 2
            )
            action_names_list = os.listdir(
                os.path.join(self.dataset_dir, iEnc_sample_character)
            )
            iEnc_sample_action, real_sample = random.sample(action_names_list, 2)
            action_path = os.path.join(
                self.dataset_dir, iEnc_sample_character, iEnc_sample_action
            )
            real_pair_path2 = os.path.join(
                self.dataset_dir, iEnc_sample_character, real_sample
            )

            iEnc_sample_pose = random.choice(os.listdir(action_path))
            real_sample_pose = random.choice(os.listdir(real_pair_path2))

            iEnc_sample_path = os.path.join(action_path, iEnc_sample_pose)
            real_sample_path = os.path.join(real_pair_path2, real_sample_pose)

            iEnc_sample = Image.open(iEnc_sample_path).convert("RGB")
            real_sample = Image.open(real_sample_path).convert("RGB")

            # Sampling Pose Encoder's input
            pEnc_character_path = os.path.join(self.dataset_dir, pEnc_sample_character)
            pEnc_sample_action = random.choice(os.listdir(pEnc_character_path))

            pEnc_action_path = os.path.join(pEnc_character_path, pEnc_sample_action)
            pEnc_sample_pose = random.choice(os.listdir(pEnc_action_path))

            pEnc_sample_path = os.path.join(pEnc_action_path, pEnc_sample_pose)
            pEnc_sample = Image.open(pEnc_sample_path).convert("RGB")

            recon_sample = iEnc_sample
            if self.transform:
                recon_sample = self.transform(recon_sample)

            if self.transform:
                state = torch.get_rng_state()
                iEnc_sample = self.transform(iEnc_sample)
                torch.set_rng_state(state)
                pEnc_sample = self.transform(pEnc_sample)
                torch.set_rng_state(state)
                real_sample = self.transform(real_sample)

            samples_dict = {
                "iEnc_sample": iEnc_sample,
                "pEnc_sample": pEnc_sample,
                "real_sample": real_sample,
                "recon_sample": recon_sample,
                "iEnc_sample_path": iEnc_sample_path,
                "pEnc_sample_path": pEnc_sample_path,
            }
            return samples_dict

class FashionDataset(Dataset):
    def __init__(self, cfg, transform=None, test=False):
        self.cfg = cfg
        self.transform = transform
        self.test = test
        if self.test:
            dataset_path = "test"
        else:
            dataset_path = "train"
        self.dataset_dir = (
            Path(cfg["dataset_config"]["dataset_path"]) / dataset_path
        )
        self.video_list = os.listdir(self.dataset_dir)
    def __len__(self):
        length = 4000
        return length
    def __getitem__(self, idx):
        if self.test:
            dir1 = random.choice(self.video_list)
            dir2 = random.choice(self.video_list)
            sample1 = random.choice(os.listdir(self.dataset_dir / dir1))
            sample2 = random.choice(os.listdir(self.dataset_dir / dir2))
            sample1 = pad_same_hw(np.array(Image.open(self.dataset_dir / dir1 / sample1).convert("RGB")))
            sample2 = pad_same_hw(np.array(Image.open(self.dataset_dir / dir2 / sample2).convert("RGB")))
            if self.transform:
                sample1 = self.transform(sample1)
                sample2 = self.transform(sample2)
            return {"sample1": sample1, "sample2": sample2}
        else:
            # Sampling Identity Encoder's input
            iEnc_sample_video, pEnc_sample_video = random.sample(
                self.video_list, 2
            )
            action_names_list = os.listdir(
                self.dataset_dir / iEnc_sample_video
            )
            iEnc_sample_action = random.choice(action_names_list)
            real_sample = random.choice(action_names_list)
            iEnc_sample_path = self.dataset_dir / iEnc_sample_video / iEnc_sample_action
            real_sample_path = self.dataset_dir / iEnc_sample_video / real_sample

            iEnc_sample = pad_same_hw(np.array(Image.open(iEnc_sample_path).convert("RGB")))
            real_sample = pad_same_hw(np.array(Image.open(real_sample_path).convert("RGB")))

            # Sampling Pose Encoder's input
            pEnc_video_path = self.dataset_dir / pEnc_sample_video
            pEnc_sample_action = random.choice(os.listdir(pEnc_video_path))

            pEnc_sample_path = pEnc_video_path / pEnc_sample_action
            pEnc_sample = pad_same_hw(np.array(Image.open(pEnc_sample_path).convert("RGB")))
            
            recon_sample = iEnc_sample
            if self.transform:
                state = torch.get_rng_state()
                iEnc_sample = self.transform(iEnc_sample)
                torch.set_rng_state(state)
                pEnc_sample = self.transform(pEnc_sample)
                torch.set_rng_state(state)
                real_sample = self.transform(real_sample)
                recon_sample = self.transform(recon_sample)
            samples_dict = {
                "iEnc_sample": iEnc_sample,
                "pEnc_sample": pEnc_sample,
                "real_sample": real_sample,
                "recon_sample": recon_sample,
                "iEnc_sample_path": str(iEnc_sample_path),
                "pEnc_sample_path": str(pEnc_sample_path),
            }
            return samples_dict