import torch.nn as nn
import torch.nn.functional as F

from models import *
from data import *
from utils import *

import numpy as np
import random

import time

# Danger zone
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
