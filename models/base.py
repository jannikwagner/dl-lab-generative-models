import torch
import torch.nn as nn


class Interpolate(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class MLP(nn.Module):
    def __init__(self, layer_sizes, fn=nn.ReLU, end_fn=nn.ReLU):
        super().__init__()
        layers = []
        for i, (s1, s2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(s1, s2))
            if i != len(layer_sizes) - 2:
                layers.append(fn())
                layers.append(nn.BatchNorm1d(s2))
            else:
                layers.append(end_fn())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
