def GGANLoss(D_on_fake):
    # (N, 3, h, w)
    # D(G(x)): (N)
    return ((D_on_fake - 1) ** 2).mean()

def DGANLoss(D_on_true, D_on_fake):
     return ((D_on_true - 1) ** 2 + D_on_fake ** 2).mean()

def CycleLoss(double_fake_x, double_fake_y, x, y):
    return ((double_fake_x - x).abs() + (double_fake_y - y).abs()).mean()

def GLoss(D_on_fake_x, D_on_fake_y, double_fake_x, double_fake_y, x, y, lbda):
    return GGANLoss(D_on_fake_x) + GGANLoss(D_on_fake_y) + lbda * CycleLoss(double_fake_x, double_fake_y, x, y)

def DLoss(D_on_true_x, D_on_fake_x, D_on_true_y, D_on_fake_y):
    return DGANLoss(D_on_true_x, D_on_fake_x) + DGANLoss(D_on_true_y, D_on_fake_y)

import os
import torch

from data import *

def load_facades():
    dir = "data/facades"
    files = sorted(os.listdir(dir))

    total = len(files)

    train = int(total * 0.9) - (int(total * 0.9) % 3)

    train_files = files[:train]

    test_files = files[train:]

    train_loader = torch.utils.data.DataLoader(
        FacadesDataset(train_files),
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        FacadesDataset(test_files),
        batch_size=1,
        shuffle=True,
        num_workers=1
    )

    return train_loader, test_loader

def load_maps():
    train_dirA = "data/maps/trainA"
    train_dirB = "data/maps/trainB"

    test_dirA = "data/maps/testA"
    test_dirB = "data/maps/testB"

    files_trainA = sorted(os.listdir(train_dirA))[:500]
    files_trainB = sorted(os.listdir(train_dirB))[:500]
    files_testA = sorted(os.listdir(test_dirA))[-12:]
    files_testB = sorted(os.listdir(test_dirB))[-12:]

    train_loader = torch.utils.data.DataLoader(
        MapsDataset(files_trainA, files_trainB, train_dirA, train_dirB),
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        MapsDataset(files_testA, files_testB, test_dirA, test_dirB),
        batch_size=1,
        shuffle=True,
        num_workers=1
    )

    return train_loader, test_loader
