import os

import torch
from torchvision import datasets
from torchvision import transforms as tf

from PIL import Image

from tqdm import tqdm

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
