import torch
import torch.nn as nn


def dis_ganLoss(cfg, real, Real_label, fake, Fake_label):
    l_type = cfg["ganLoss_type"]
    w = cfg["dis_ganLoss_weight"]

    if l_type == "BCE":
        criterion = nn.BCELoss()
        d_loss_fake = criterion(fake, Fake_label)
        d_loss_real = criterion(real, Real_label)
    elif l_type == "MSE":
        criterion = nn.MSELoss()
        d_loss_fake = criterion(fake, Fake_label)
        d_loss_real = criterion(real, Real_label)
    elif l_type == "hinge":
        d_loss_real = torch.nn.ReLU()(1.0 - real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + fake).mean()

    d_loss = (d_loss_fake + d_loss_real) * w

    return d_loss, d_loss_fake, d_loss_real


def gen_ganLoss(cfg, fake, label):
    l_type = cfg["ganLoss_type"]
    w = cfg["gen_ganLoss_weight"]

    if l_type == "BCE":
        criterion = nn.BCELoss()
        loss = criterion(fake, label)
    elif l_type == "MSE":
        criterion = nn.MSELoss()
        loss = criterion(fake, label)
    elif l_type == "hinge":
        loss = -fake.mean()

    loss *= w
    return loss


def gen_reconLoss(cfg, pred, real):
    w = cfg["gen_reconLoss_weight"]

    criterion = nn.L1Loss()
    loss = criterion(pred, real)

    loss *= w
    return loss