from TrainModels import TrainAE
from datasets import getMNIST, getCIFAR10, getFashionMNIST, get_data_loader
from defaults import device, MNIST, FashionMNIST, CIFAR10
from models.MNIST.Autoencoder import *
from models.CIFAR10.Auoencoder import *
from train.Autoencoder import train_autoencoder

from utility import save_model, extract_images_of_models, load_model, latent_space_pca, sample_in_pc, \
    plot_images2, normal_to_pc, get_sample_k_of_d, labeled_latent_space_pca


mnist_train_aes = [
    TrainAE(MNISTEncoder1, MNISTDecoder1, "dummy", 2, MNIST),
    TrainAE(MNISTEncoder1, MNISTDecoder1, "AE1", 50, MNIST),
    TrainAE(MNISTEncoder2, MNISTDecoder2, "AE2", 50, MNIST),
    TrainAE(MNISTEncoder3, MNISTDecoder3, "AE3", 50, MNIST),
    TrainAE(MNISTEncoder4, MNISTDecoder4, "AE4", 50, MNIST),
    TrainAE(MNISTEncoder5, MNISTDecoder5, "AE5", 50, MNIST),
]

fashionMnist_train_aes = [
    TrainAE(MNISTEncoder1, MNISTDecoder1, "Fashiondummy", 2, FashionMNIST),
    TrainAE(MNISTEncoder1, MNISTDecoder1, "FashionAE1", 50, FashionMNIST),
    TrainAE(MNISTEncoder2, MNISTDecoder2, "FashionAE2", 50, FashionMNIST),
    TrainAE(MNISTEncoder3, MNISTDecoder3, "FashionAE3", 50, FashionMNIST),
    TrainAE(MNISTEncoder4, MNISTDecoder4, "FashionAE4", 50, FashionMNIST),
    TrainAE(MNISTEncoder5, MNISTDecoder5, "FashionAE5", 50, FashionMNIST),
]

train_cifar10_aes = [
    TrainAE(CIFAR10Encoder1, CIFAR10Decoder1, "CIFAR10dummy", 2, CIFAR10),
    TrainAE(CIFAR10Encoder1, CIFAR10Decoder1, "CIFAR10AE1", 50, CIFAR10),
    TrainAE(CIFAR10Encoder2, CIFAR10Decoder2, "CIFAR10AE2", 50, CIFAR10),
    TrainAE(CIFAR10Encoder3, CIFAR10Decoder3, "CIFAR10AE3", 50, CIFAR10),
    TrainAE(CIFAR10Encoder4, CIFAR10Decoder4, "CIFAR10AE4", 50, CIFAR10),
    TrainAE(CIFAR10Encoder5, CIFAR10Decoder5, "CIFAR10AE5", 50, CIFAR10),
    TrainAE(CIFAR10Encoder6, CIFAR10Decoder6, "CIFAR10AE6", 50, CIFAR10),
    TrainAE(CIFAR10Encoder7, CIFAR10Decoder7, "CIFAR10AE7", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),100), "CIFAR10AE8", 1, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),100), "CIFAR10AE8", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(5,3,3),(2,1,1),100), "CIFAR10AE9", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(5,3,3,3,3),(2,1,1,1,1),100), "CIFAR10AE10", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,20,24,28,32,36,40),(3,)*10,(1,)*10,100), "CIFAR10AE11", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64,128),(3,)*6,(1,1,2,1,1,2),100), "CIFAR10AE12", 100, CIFAR10),
]


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


if __name__ == "__main__":
    for trainae in train_cifar10_aes[-6:]:
        trainae.train()
