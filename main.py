import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os

import torch
from torchvision import datasets
from torchvision import transforms as tf

from PIL import Image

from tqdm import tqdm

class Enter(nn.Module):
    def __init__(self, out_ch):
        super(Enter, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, out_ch, kernel_size=7, padding=3),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)

class Downscale(nn.Module):
    def __init__(self, in_ch):
        super(Downscale, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(in_ch * 2, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)

class Residual(nn.Module):
    def __init__(self, ch):
        super(Residual, self).__init__()

        self.relu = nn.ReLU()

        self.seq = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ch, affine=True)
        )

    def forward(self, x):
        return self.relu(x + self.seq(x))

class Upscale(nn.Module):
    def __init__(self, in_ch):
        super(Upscale, self).__init__()

        assert in_ch % 2 == 0

        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(in_ch // 2, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)

class Exit(nn.Module):
    def __init__(self, in_ch):
        super(Exit, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3),
            #nn.InstanceNorm2d(in_ch, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_ch, 3, kernel_size=5, padding=2),
            #nn.InstanceNorm2d(in_ch, affine=True),
            #nn.Conv2d(in_ch, 3, kernel_size=3, padding=1),
            #nn.InstanceNorm2d(in_ch, affine=True),
            #nn.Conv2d(in_ch, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)

class Poll(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Poll, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.seq(x)

class IdlePoll(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(IdlePoll, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.seq(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        seq = (
            [Enter(64), Downscale(64)] +
            [Residual(128)] * 6 +
            [Upscale(128), Exit(64)]
        )

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x) # (N, 3, h, w)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.seq = nn.Sequential(
            IdlePoll(3, 64),
            Poll(64, 128),
            Poll(128, 128),
            Poll(128, 128),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        probs = self.seq(x) # (N, 1, h, w)
        return probs.mean((1, 2, 3)) # (N)

class FacadesDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        super(FacadesDataset).__init__()

        self.trans = tf.Compose([
            tf.Resize((228, 228)),
            tf.ToTensor()
        ])

        self.X = []
        self.Y = []

        for file in files:
            if file[-3:] == "jpg":
                im = Image.open(f"data/facades/{file}").copy()
                self.X.append(im)
            elif file[-3:] == "png":
                im = Image.open(f"data/facades/{file}").copy().convert("RGB")
                self.Y.append(im)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.trans(self.X[idx])
        y = self.trans(self.Y[idx])
        return (x, y)

class MapsDataset(torch.utils.data.Dataset):
    def __init__(self, files_X, files_Y, dirX, dirY):
        super(FacadesDataset).__init__()

        self.trans = tf.Compose([
            tf.Resize((256, 256)),
            tf.ToTensor()
        ])

        self.X = []
        self.Y = []

        for file_X, file_Y in zip(files_X, files_Y):
            im = Image.open(f"{dirX}/{file_X}").copy()
            self.X.append(im)

            im = Image.open(f"{dirY}/{file_Y}").copy()
            self.Y.append(im)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.trans(self.X[idx])
        y = self.trans(self.Y[idx])
        return (x, y)

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lbda = 10

        self.loaders = [load_facades, load_maps]

        self.prefixes = ["facades", "maps"]

    def train(self, params):
        for dataset in [0, 1]:
            self.train_loader, self.val_loader = None, None

            self.G1 = None
            self.G2 = None

            self.D1 = None
            self.D2 = None

            self.g_optimizer = None

            self.d_optimizer = None

            epochs, save_every = params[dataset]
            pref = self.prefixes[dataset]
            self.train_loader, self.val_loader = self.loaders[dataset]()

            self.G1 = Generator().to(self.device)
            self.G2 = Generator().to(self.device)

            self.D1 = Discriminator().to(self.device)
            self.D2 = Discriminator().to(self.device)

            self.g_optimizer = torch.optim.AdamW(
                list(self.G1.parameters()) + list(self.G2.parameters()),
                lr=0.0002
            )

            self.d_optimizer = torch.optim.AdamW(
                list(self.D2.parameters()) + list(self.D1.parameters()),
                lr=0.0002
            )

            self.g_history = []
            self.d_history = []

            self.l2_fake_X = []
            self.l2_fake_Y = []

            self.l2_double_fake_X = []
            self.l2_double_fake_Y = []

            for epoch in range(epochs):
                start = time.time()
                print(f"\nepoch {epoch+1}")
                total = len(self.train_loader)
                i = 0

                self.G1.train()
                self.G2.train()
                self.D1.train()
                self.D2.train()
                for (x, y) in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    fake_y = self.G1(x)
                    fake_x = self.G2(y)

                    double_fake_x = self.G2(fake_y)
                    double_fake_y = self.G1(fake_x)

                    D1_on_true_x = self.D1(x)
                    D2_on_true_y = self.D2(y)

                    D1_on_fake_x = self.D1(fake_x)
                    D2_on_fake_y = self.D2(fake_y)

                    # Генераторы
                    self.g_optimizer.zero_grad()

                    gloss = GLoss(D1_on_fake_x, D2_on_fake_y, double_fake_x, double_fake_y, x, y, self.lbda)

                    gloss.backward(retain_graph=True)

                    self.g_optimizer.step()

                    self.g_history.append(gloss.item())

                    # Дискриминаторы
                    self.d_optimizer.zero_grad()

                    dloss = DLoss(D1_on_true_x, D1_on_fake_x, D2_on_true_y, D2_on_fake_y)

                    dloss.backward()

                    self.d_optimizer.step()

                    self.d_history.append(dloss.item())

                    i += 1
                    print(f"\r[{(i/total):.3f}] gloss: {gloss.item():.3f}, dloss: {dloss.item():.3f}, time: {time.time() - start:.1f}s", end='     ')


                self.G1.eval()
                self.G2.eval()
                self.D1.eval()
                self.D2.eval()
                for (x, y) in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    fake_x = torch.clamp(self.G2(y), 0.0, 1.0).detach()
                    fake_y = torch.clamp(self.G1(x), 0.0, 1.0).detach()

                    double_fake_x = torch.clamp(self.G2(fake_y), 0.0, 1.0).detach()
                    double_fake_y = torch.clamp(self.G1(fake_x), 0.0, 1.0).detach()

                    self.l2_fake_X.append(((x - fake_x) ** 2).mean())
                    self.l2_fake_Y.append(((y - fake_y) ** 2).mean())

                    self.l2_double_fake_X.append(((x - double_fake_x) ** 2).mean())
                    self.l2_double_fake_Y.append(((y - double_fake_y) ** 2).mean())


                if (epoch + 1) % save_every == 0:
                    self.test_and_save(
                        pref
                        , epoch+1
                        , self.g_history
                        , self.d_history
                        , self.l2_fake_X
                        , self.l2_fake_Y
                        , self.l2_double_fake_X
                        , self.l2_double_fake_Y
                    )

                    torch.save(self, f"{pref}/saved/epoch_{epoch+1}")


    def test_and_save(self
        , pref
        , epoch
        , g_history
        , d_history
        , l2_fake_X
        , l2_fake_Y
        , l2_double_fake_X
        , l2_double_fake_Y
    ):
        self.G1.eval()
        self.G2.eval()
        self.D1.eval()
        self.D2.eval()
        done = False

        plt.plot(g_history, label="Generator loss")
        plt.plot(d_history, label="Discriminator loss")
        plt.legend()

        plt.savefig(f"{pref}/loss_history/epoch_{epoch}.png", dpi=300)

        plt.show()

        plt.plot(l2_fake_X, label="l2 on fake X")
        plt.plot(l2_fake_Y, label="l2 on fake Y")
        plt.legend()

        plt.savefig(f"{pref}/fake_history/epoch_{epoch}.png", dpi=300)

        plt.show()

        plt.plot(l2_double_fake_X, label="l2 on double fake X")
        plt.plot(l2_double_fake_Y, label="l2 on double fake Y")
        plt.legend()

        plt.savefig(f"{pref}/double_fake_history/epoch_{epoch}.png", dpi=300)

        plt.show()

        done = 0
        for (x, y) in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)

            fake_x = torch.clamp(self.G2(y), 0.0, 1.0).detach()
            fake_y = torch.clamp(self.G1(x), 0.0, 1.0).detach()

            double_fake_x = torch.clamp(self.G2(fake_y), 0.0, 1.0).detach()
            double_fake_y = torch.clamp(self.G1(fake_x), 0.0, 1.0).detach()

            x_np = x[0].permute(1, 2, 0).cpu().numpy()
            y_np = y[0].permute(1, 2, 0).cpu().numpy()

            fake_x_np = fake_x[0].permute(1, 2, 0).cpu().numpy()
            fake_y_np = fake_y[0].permute(1, 2, 0).cpu().numpy()

            double_fake_x_np = double_fake_x[0].permute(1, 2, 0).cpu().numpy()
            double_fake_y_np = double_fake_y[0].permute(1, 2, 0).cpu().numpy()

            first_row = np.concatenate((x_np, fake_x_np, double_fake_x_np), axis=1)
            second_row = np.concatenate((y_np, fake_y_np, double_fake_y_np), axis=1)

            full = np.concatenate((first_row, second_row), axis=0)

            plt.grid(False)
            plt.imshow(full)

            if done != 4:
                done += 1
                plt.savefig(f"{pref}/results/epoch_{epoch}_{done}.png", dpi=300)

            plt.show()


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


trainer = Trainer()

params = [(60, 60), (10, 10)]

trainer.train(params)
