import torch
import torch.nn as nn
from defaults import CelebA_size
from models.base import Interpolate, MLP
c,h,w=CelebA_size


def get_sym_fully_conv_vae(channels, filter_sizes, pools, fc_layers=(), c=c, h=h, w=w, enc_fn=nn.Tanh):
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
            layers.append(nn.Flatten())
            if len(fc_layers) <= 2:
                self.encoded_size = fc_layers[0]
            if len(fc_layers) >= 3:
                self.encoded_size = fc_layers[-2]
                layers.append(MLP(fc_layers[:-1], nn.ReLU, nn.ReLU))
                layers.append(nn.BatchNorm1d(fc_layers[-2]))
            
            self.model = nn.Sequential(*layers)
            self.mean = nn.Sequential(nn.Linear(self.encoded_size, self.latent_size), enc_fn())
            self.log_var = nn.Sequential(nn.Linear(self.encoded_size, self.latent_size), enc_fn())

        def forward(self, x):
            x_encoded = self.model(x)
            return self.mean(x_encoded), self.log_var(x_encoded)
        
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
                if i != 0:
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm2d(new_c))
            layers.append(nn.Tanh())
            if new_h2 != h or new_w2 != w:
                layers.append(Interpolate((h, w)))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return Encoder, Decoder


