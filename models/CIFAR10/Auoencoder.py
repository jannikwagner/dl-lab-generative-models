import torch
import torch.nn as nn

c,h,w=3,32,32


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
                                   nn.Linear(32 * ((h-4)//2-4)//2 * ((w-4)//2-4)//2, 512),
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
                                   nn.Linear(32 * ((h-4)//2-4)//2 * ((w-4)//2-4)//2, 1024),
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


class Interpolate(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


def get_symmetric_fully_convolutional_autoencoder(channels, filter_sizes, pools, latent_size, c=c, h=h, w=w):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = latent_size
            layers = []
            new_h, new_w = h, w
            for old_c, new_c, filter_size, pool in zip((c,) + channels[:-1], channels, filter_sizes, pools):
                new_h -= filter_size-1
                new_w -= filter_size-1
                layers.append(nn.Conv2d(old_c, new_c, filter_size))
                if pool != 1:
                    new_h //= pool
                    new_w //= pool
                    layers.append(nn.MaxPool2d(pool))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm2d((new_c)))
            layers.append(nn.Flatten())
            layers.append(nn.Linear(new_h*new_w*channels[-1], latent_size))
            layers.append(nn.Tanh())
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = latent_size
            new_h, new_w = h, w
            for filter_size, pool in zip(filter_sizes, pools):
                new_h -= filter_size-1
                new_w -= filter_size-1
                if pool != 1:
                    new_h //= pool
                    new_w //= pool
            layers = [nn.Linear(latent_size, new_h*new_w*channels[-1]),
                      nn.ReLU(),
                      nn.BatchNorm1d(new_h*new_w*channels[-1]),
                      nn.Unflatten(1, (channels[-1], new_h, new_w))]
            for i, (new_c, old_c, filter_size, pool) in reversed(list(enumerate(zip((c,) + channels[:-1], channels, filter_sizes, pools)))):
                if pool != 1:
                    new_h *= pool
                    new_w *= pool
                    layers.append(nn.Upsample(scale_factor=pool, mode="nearest"))
                layers.append(nn.ConvTranspose2d(old_c, new_c, filter_size))
                new_h += filter_size-1
                new_w += filter_size-1
                if i != 0:
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm2d(new_c))
            layers.append(nn.Tanh())
            if new_h != h or new_w != w:
                layers.append(Interpolate((h, w)))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)
    return Encoder, Decoder
