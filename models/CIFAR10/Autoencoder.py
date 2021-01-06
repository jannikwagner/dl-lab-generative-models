import torch
import torch.nn as nn
from models.base import Interpolate, MLP

c, h, w = 3, 32, 32


class CIFAR10Encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 2
        self.model = nn.Sequential(nn.Flatten(),
                                   nn.Linear(c * h * w, 2),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class CIFAR10Decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 2
        self.model = nn.Sequential(nn.Linear(2, c * h * w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


class CIFAR10Encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 10
        self.model = nn.Sequential(nn.Flatten(),
                                   nn.Linear(c * h * w, 10),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class CIFAR10Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 10
        self.model = nn.Sequential(nn.Linear(10, c * h * w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


class CIFAR10Encoder3(nn.Module):
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


class CIFAR10Decoder3(nn.Module):
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


class CIFAR10Encoder4(nn.Module):
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
                                   nn.Linear(32 * ((h-4)//2-4)//2 *
                                             ((w-4)//2-4)//2, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 10),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


CIFAR10Decoder4 = CIFAR10Decoder3


class CIFAR10Encoder5(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 100
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
                                   nn.Linear(32 * ((h-4)//2-4)//2 *
                                             ((w-4)//2-4)//2, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 100),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class CIFAR10Decoder5(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 100
        self.model = nn.Sequential(nn.Linear(100, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, c*h*w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


class CIFAR10Encoder6(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 100
        self.model = nn.Sequential(nn.Conv2d(c, 4, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.Dropout(),
                                   nn.Conv2d(8, 16, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.Flatten(),
                                   nn.Dropout(),
                                   nn.Linear(32 * ((h-4)//2-4)//2 *
                                             ((w-4)//2-4)//2, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(128, 100),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class CIFAR10Decoder6(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 100
        self.model = nn.Sequential(nn.Linear(100, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(128, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(256, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(1024, c*h*w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


CIFAR10Encoder7 = CIFAR10Encoder6


class CIFAR10Decoder7(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 100
        self.model = nn.Sequential(nn.Linear(100, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 256),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(256, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 2048),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(2048, 4096),
                                   nn.ReLU(),
                                   nn.Linear(4096, c*h*w),
                                   nn.Tanh(),
                                   nn.Unflatten(-1, (c, h, w)))

    def forward(self, x):
        return self.model(x)


def get_symmetric_fully_convolutional_autoencoder(channels, filter_sizes, pools, fc_layers=(), c=c, h=h, w=w, enc_fn=nn.Identity):
    if isinstance(fc_layers, int):
        fc_layers = (fc_layers,)
    new_h, new_w = h, w
    for filter_size, pool in zip(filter_sizes, pools):
        new_h -= filter_size-1
        new_w -= filter_size-1
        if pool != 1:
            new_h //= pool
            new_w //= pool
    fc_layers = (new_h*new_w*channels[-1],) + fc_layers

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = fc_layers[-1]
            layers = []
            for old_c, new_c, filter_size, pool in zip((c,) + channels[:-1], channels, filter_sizes, pools):
                layers.append(nn.Conv2d(old_c, new_c, filter_size))
                if pool != 1:
                    layers.append(nn.MaxPool2d(pool))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm2d(new_c))
            if len(fc_layers) == 1:  # no batchnorm or relu on last layer
                layers.pop()
                layers.pop()
                layers.append(enc_fn())
            layers.append(nn.Flatten())
            if len(fc_layers) > 1:
                layers.append(MLP(fc_layers, nn.ReLU, enc_fn))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)*4  # if activation function is tanh this could be quite useful

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = fc_layers[-1]
            new_h2, new_w2 = new_h, new_w
            layers = []
            if len(fc_layers) > 1:
                layers.append(MLP(list(reversed(fc_layers)), nn.ReLU, nn.ReLU))
                layers.append(nn.BatchNorm1d(new_h2*new_w2*channels[-1]))
            layers.append(nn.Unflatten(1, (channels[-1], new_h2, new_w2)))
            for i, (new_c, old_c, filter_size, pool) in reversed(list(enumerate(zip((c,) + channels[:-1], channels, filter_sizes, pools)))):
                if pool != 1:
                    new_h2 *= pool
                    new_w2 *= pool
                    layers.append(nn.Upsample(
                        scale_factor=pool, mode="bilinear"))
                layers.append(nn.ConvTranspose2d(old_c, new_c, filter_size))
                new_h2 += filter_size-1
                new_w2 += filter_size-1
                if i != 0:  # no relu and batchnorm in last layer
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm2d(new_c))
            layers.append(nn.Tanh())
            if new_h2 != h or new_w2 != w:
                layers.append(Interpolate((h, w)))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return Encoder, Decoder

def get_stacked_ful_conv_ae(channels, filter_sizes, pools, fc_layers=(), c=c, h=h, w=w, enc_fn=nn.Identity):
    if isinstance(fc_layers, int):
        fc_layers = (fc_layers,)
    new_h, new_w = h, w
    for filter_size, pool in zip(filter_sizes, pools):
        new_h -= filter_size-1
        new_w -= filter_size-1
        if pool != 1:
            new_h //= pool
            new_w //= pool
    fc_layers = (new_h*new_w*channels[-1],) + fc_layers

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = fc_layers[-1]
            layers = []
            self.stacks = []
            for i, (old_c, new_c, filter_size, pool) in enumerate(zip((c,) + channels[:-1], channels, filter_sizes, pools)):
                layers.append(nn.Conv2d(old_c, new_c, filter_size))
                if pool != 1:
                    layers.append(nn.MaxPool2d(pool))
                if i == len(channels) and len(fc_layers) == 1:  # last layer
                    layers.append(enc_fn())
                else:
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm2d(new_c))
                self.stacks.append(nn.Sequential(*layers))
            layers.append(nn.Flatten())
            if len(fc_layers) > 1:
                layers.append(MLP(fc_layers, nn.ReLU, enc_fn))
            self.model = nn.Sequential(*layers)

        def forward(self, x, stack=None):
            if stack is None or stack >= len(self.stacks):
                return self.model(x) * 4
            else:
                return self.stacks[stack](x) * 4

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = fc_layers[-1]
            new_h2, new_w2 = new_h, new_w
            layers = []
            inv_stacks = []
            if len(fc_layers) > 1:  # fully connected layers
                layers.append(MLP(list(reversed(fc_layers)), nn.ReLU, nn.ReLU))
                layers.append(nn.BatchNorm1d(new_h2*new_w2*channels[-1]))
            layers.append(nn.Unflatten(1, (channels[-1], new_h2, new_w2)))
            for i, (new_c, old_c, filter_size, pool) in reversed(list(enumerate(zip((c,) + channels[:-1], channels, filter_sizes, pools)))):
                inv_stacks.append(len(layers))
                if pool != 1:
                    new_h2 *= pool
                    new_w2 *= pool
                    layers.append(nn.Upsample(
                        scale_factor=pool, mode="bilinear"))
                layers.append(nn.ConvTranspose2d(old_c, new_c, filter_size))
                new_h2 += filter_size-1
                new_w2 += filter_size-1
                if i != 0:  # only if not last layer
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm2d(new_c))
            layers.append(nn.Tanh())
            if new_h2 != h or new_w2 != w:
                layers.append(Interpolate((h, w)))
            self.stacks = []
            for i in reversed(inv_stacks):
                self.stacks.append(nn.Sequential(*layers[i:]))
            self.model = nn.Sequential(*layers)

        def forward(self, x, stack=None):
            if stack is None or stack >= len(self.stacks):
                return self.model(x)
            return self.stacks[stack](x)

    return Encoder, Decoder
