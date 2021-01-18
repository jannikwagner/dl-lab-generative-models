import torch

CKPT_PATH = "ckpts"
BATCH_SIZE = 64
EPOCHS = 50
DATA_PATH = "./data" if torch.cuda.is_available() else "~/pytorch/data"  # laufe ich auf der dgx oder dem Laptop?
MNIST = "MNIST"
torch.manual_seed(111)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
FashionMNIST = "FashionMNIST"
CIFAR10 = "CIFAR10"
CelebA = "CelebA"
CelebA_size = (3,218,178)
CIFAR10_size = (3, 32, 32)
