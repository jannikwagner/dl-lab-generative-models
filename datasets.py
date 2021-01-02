import torch
import torchvision
from torchvision import transforms
from utility import plot_images2
from torch.utils.data import DataLoader
import time

from defaults import BATCH_SIZE, DATA_PATH, MNIST, FashionMNIST, CIFAR10, CelebA


class OneClassIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, label, batchsize=64):
        self.dataset = dataset
        self.label = label
        self.batchsize = batchsize
        self.size = len([(x,l) for (x,l) in dataset if l==label])
    
    def __iter__(self):
        x = self.size // self.batchsize * self.batchsize
        for tensor, label in self.dataset:
            if label == self.label:
                x -= 1
                yield (tensor, label)
            if x == 0:
                break


def getMNIST():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(
        root=DATA_PATH, train=True, download=True, transform=transform
    )
    return train_set


def getCIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_PATH, download=True, transform=transform
    )
    return train_set


def getCelebA():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.CelebA(
        root=DATA_PATH, download=True, transform=transform
    )
    return train_set


def getFashionMNIST():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_PATH, transform=transform, download=True
    )
    return train_set


def getImageNet():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.ImageNet(
        root=DATA_PATH, transform=transform
    )
    return train_set

_funs = {MNIST: getMNIST, FashionMNIST: getFashionMNIST, CIFAR10: getCIFAR10, CelebA: getCelebA}
_data_sets = {name: None for name in _funs.keys()}


def get_data_set(name):
    assert name in _data_sets
    if _data_sets[name] is None:
        _data_sets[name] = _funs[name]()
    return _data_sets[name]


def get_data_loader(name, label=None):
    train_set = get_data_set(name)
    if label is None:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            OneClassIterableDataset(train_set, label), batch_size=BATCH_SIZE, shuffle=False,
        )
    return train_loader

def downloader():
    while True:
        try:
            d = getCelebA()
            break
        except:
            print("hi")
            time.sleep(600)

if __name__ == "__main__":
    d=get_data_set(CelebA)
    print(len(d))
