from TrainModels import TrainAE
from datasets import getMNIST, getCIFAR10, getFashionMNIST, get_data_loader
from defaults import device, MNIST, FashionMNIST, CIFAR10, CelebA, CelebA_size, CIFAR10_size, CKPT_PATH
from models.MNIST.Autoencoder import *
from models.CIFAR10.Autoencoder import *
from train.Autoencoder import train_autoencoder, train_stacked_ae
import torch.nn as nn
from models.VarAE import get_sym_fully_conv_vae
from models.base import get_sym_resnet_ae, get_sym_ful_conv_ae2
import time
import os

from utility import save_model, extract_images_of_models, load_model, latent_space_pca, sample_in_pc, \
    plot_images2, normal_to_pc, get_sample_k_of_d, labeled_latent_space_pca, param_count, param_print, imgs_to_gif


def tests():
    a = (MNISTEncoder4, MNISTDecoder4, "AE4")
    b = (MNISTEncoder4, MNISTDecoder4, "FashionAE4")
    dl_a = get_data_loader("MNIST")
    dl_b = get_data_loader("FashionMNIST")
    encoder_a, decoder_a = load_model(*a)
    encoder_b, decoder_b = load_model(*b)
    mean_a, cov_a = latent_space_pca(encoder_a, dl_a)
    mean_b, cov_b = latent_space_pca(encoder_b, dl_a)
    ls = encoder_a.latent_size
    plot_images2(decoder_a(sample_in_pc(9, mean_a, cov_a)).detach())
    plot_images2(decoder_a(normal_to_pc(get_sample_k_of_d(9, 4, ls) * 3, mean_a, cov_a)).detach())
    plot_images2(decoder_a(normal_to_pc(torch.eye(ls) * 3, mean_a, cov_a)).detach())
    batch_a = next(iter(dl_a))[0]
    batch_b = next(iter(dl_b))[0]
    plot_images2(decoder_a(encoder_a(batch_a)).detach())
    plot_images2(decoder_a(encoder_a(batch_b)).detach())
    plot_images2(decoder_a(encoder_b(batch_a)).detach())
    plot_images2(decoder_a(encoder_b(batch_b)).detach())
    plot_images2(decoder_b(encoder_a(batch_a)).detach())
    plot_images2(decoder_b(encoder_a(batch_b)).detach())
    plot_images2(decoder_b(encoder_b(batch_a)).detach())
    plot_images2(decoder_b(encoder_b(batch_b)).detach())
    x_a = labeled_latent_space_pca(encoder_a, dl_a)
    plot_images2(decoder_a(sample_in_pc(9, *x_a[7])).detach())
    plot_images2(decoder_a(torch.stack([mean for label, (mean, cov) in x_a.items()]).view(-1,ls)).detach())
    x_b = labeled_latent_space_pca(encoder_b, dl_b)
    plot_images2(decoder_b(sample_in_pc(9, *x_b[9])).detach())
    plot_images2(decoder_b(torch.stack([mean for label, (mean, cov) in x_b.items()]).view(-1,ls)).detach())


def reconstruct(svd):
    return  svd[0]@torch.diag(svd[1])@svd[2].transpose(0,1)


def model_summary(E,D):
    size = CelebA_size
    e,d=E(),D()
    param_print(e)
    print(param_count(e))
    print(e)
    param_print(d)
    print(param_count(d))
    print(d)
    x=torch.randn(32,*size)
    a=x
    for mod in list(e.modules())[2:]:
        if isinstance(mod, MLP) or isinstance(mod, nn.Sequential):
            continue
        print(mod)
        a = mod(a)
        print(a.size())
        print(torch.prod(torch.as_tensor(a.size()[1:])))
        print()
    z=e(x)
    y=d(z)
    test_stacked_ae(e,d)


def test_stacked_ae(e,d):    
    x=torch.randn(32,*CelebA_size)
    for stack in range(len(e.stacks)):
        z = e(x,stack)
        y=d(z,stack)
        print(z.size())
        print(x.size())


def get_gifs(name):
    imgs_to_gif(os.path.join(CKPT_PATH, name, "compressed_images"), os.path.join(CKPT_PATH, name, name+"c.gif"))
    imgs_to_gif(os.path.join(CKPT_PATH, name, "generated_images"),os.path.join(CKPT_PATH, name, name+"g.gif"))
    imgs_to_gif(os.path.join(CKPT_PATH, name, "pca_gen_images"), os.path.join(CKPT_PATH, name, name+"pg.gif"))
    imgs_to_gif(os.path.join(CKPT_PATH, name, "labeled_pca_gen_images", "0"), os.path.join(CKPT_PATH, name,name+"lpg0.gif"))
    imgs_to_gif(os.path.join(CKPT_PATH, name, "labeled_pca_gen_images", "1"), os.path.join(CKPT_PATH, name,name+"lpg1.gif"))


if __name__ == "__main__":
    pass
