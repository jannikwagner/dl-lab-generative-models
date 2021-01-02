import torch
import torch.nn as nn

c,h,w=1,28,28


class MNISTEncoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 2
        self.model = nn.Sequential(nn.Flatten(),
                                   nn.Linear(c * h * w, 2),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class MNISTDecoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 2
        self.model = nn.Sequential(nn.Linear(2, c * h * w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


class MNISTEncoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 10
        self.model = nn.Sequential(nn.Flatten(),
                                   nn.Linear(c * h * w, 10),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class MNISTDecoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 10
        self.model = nn.Sequential(nn.Linear(10, c * h * w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


class MNISTEncoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 10
        self.model = nn.Sequential(nn.Flatten(),
                                   nn.Linear(c * h * w, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 10),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class MNISTDecoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 10
        self.model = nn.Sequential(nn.Linear(10, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, c*h*w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


class MNISTEncoder4(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 10
        self.model = nn.Sequential(nn.Conv2d(c, 4, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(8, 16, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.Flatten(),
                                   nn.Linear(32 * ((h-4)//2-4)//2 * ((w-4)//2-4)//2, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 10),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


MNISTDecoder4 = MNISTDecoder3


class MNISTEncoder5(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 2
        self.model = nn.Sequential(nn.Conv2d(c, 4, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(8, 16, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.Flatten(),
                                   nn.Linear(32 * ((h-4)//2-4)//2 * ((w-4)//2-4)//2, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 2),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class MNISTDecoder5(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 2
        self.model = nn.Sequential(nn.Linear(2, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, c*h*w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)

