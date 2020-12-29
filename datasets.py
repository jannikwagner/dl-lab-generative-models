import torch
import torchvision
from torchvision import transforms
from utility import plot_images2
from torch.utils.data import DataLoader

from defaults import BATCH_SIZE, DATA_PATH, MNIST, FashionMNIST, CIFAR10


def getMNIST():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(
        root=DATA_PATH, train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
    )
    return train_loader


def getCIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_PATH, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    return train_loader


def getCelebA():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.CelebA(
        root=DATA_PATH, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    return train_loader


def getFashionMNIST():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_PATH, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    return train_loader


def getImageNet():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.ImageNet(
        root=DATA_PATH, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    return train_loader

_funs = {MNIST: getMNIST, FashionMNIST: getFashionMNIST, CIFAR10: getCIFAR10}
_data_loaders = {name: None for name in _funs.keys()}


def get_data_loader(name):
    assert name in _data_loaders
    if _data_loaders[name] is None:
        _data_loaders[name] = _funs[name]()
    return _data_loaders[name]


if __name__ == "__main__":
    d = getCelebA()
    img = next(iter(d))
    plot_images2(img[0])
