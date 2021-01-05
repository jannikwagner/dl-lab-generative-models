from TrainModels import TrainAE
from datasets import getMNIST, getCIFAR10, getFashionMNIST, get_data_loader
from defaults import device, MNIST, FashionMNIST, CIFAR10, CelebA, CelebA_size
from models.MNIST.Autoencoder import *
from models.CIFAR10.Autoencoder import *
from train.Autoencoder import train_autoencoder
from train.TrainVarAE import train_vae
import torch.nn as nn
import torch.optim as optim
from models.VarAE import get_sym_fully_conv_vae

from utility import save_model, extract_images_of_models, load_model, latent_space_pca, sample_in_pc, \
    plot_images2, normal_to_pc, get_sample_k_of_d, labeled_latent_space_pca, param_count, param_print


Optim1 = lambda p: optim.lr_scheduler.StepLR(optim.Adam(p, 0.001), step_size=1, gamma=0.97)
Optim2 = lambda p: optim.lr_scheduler.StepLR(optim.Adam(p, 0.01), step_size=1, gamma=0.95)
Optim3 = lambda p: optim.lr_scheduler.StepLR(optim.SGD(p, 0.001, weight_decay=0.01, momentum=0.9), step_size=1, gamma=0.98)


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
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),100), "CIFAR10AE8", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(5,3,3),(2,1,1),100), "CIFAR10AE9", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(5,3,3,3,3),(2,1,1,1,1),100), "CIFAR10AE10", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,20,24,28,32,36,40),(3,)*10,(1,)*10,100), "CIFAR10AE11", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64,128),(3,)*6,(1,1,2,1,1,2),100), "CIFAR10AE12", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),50), "CIFAR10AE13", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32),(5,3,3,3),(1,1,1,2),25), "CIFAR10AE14", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,32,32,32),(3,)*7,(1,)*7,100), "CIFAR10AE15", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,32,32),(3,)*6,(1,)*6,100), "CIFAR10AE16", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64,64,64),(3,)*7,(1,)*7,200), "CIFAR10AE17", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),200,enc_fn=nn.Identity), "CIFAR10AE18", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(3,5,7,9,11),(1,1,1,1,1),256,enc_fn=nn.Identity), "CIFAR10AE19s", 10, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(3,5,7,9,11),(1,1,1,1,1),256,enc_fn=nn.Identity), "CIFAR10AE19", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(3,3,5,7,9),(1,1,1,1,1),512), "CIFAR10AE20", 100, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(3,3,5,7,9),(1,1,1,1,1),512), "CIFAR10AE21", 100, CIFAR10, label=1),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),100), "CIFAR10AE22", 1000, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),100), "CIFAR10AE23", 1000, CIFAR10, label=2),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(3,3,5,7,9),(1,1,1,1,1),512), "CIFAR10AE24", 1000, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,64),(3,3,5,7,9),(1,1,1,1,1),512), "CIFAR10AE25", 1000, CIFAR10, label=3),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),512), "CIFAR10AE26", 1000, CIFAR10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),100), "CIFAR10AE27", 1000, CIFAR10, label=4),
]

train_CelebA_aes = [
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,)*4,(3,)*4,(2,)*4,128, *CelebA_size),"CelebA4",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,48,64),(3,4,5,6,7,8),(1,2,1,2,1,2),512,*CelebA_size),"CelebA2",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,48,64,80,96),(3,3,3,3,3,3,3,3),(1,2,1,2,1,2,1,2),512, *CelebA_size),"CelebA3",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16,32,48,64),(3,3,3,3,3,3),(1,2,1,2,1,2),512, *CelebA_size),"CelebA1",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder(tuple(range(4,21)),(11,)*17,(1,)*17,1024, *CelebA_size),"CelebA5",100,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,20),(5,5,5,5,5),(1,2,2,2,2),(1024,800,600), *CelebA_size),"CelebA8",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,20,24,28,32),(3,5,3,5,3,5,3,7),(1,2,1,2,1,2,1,1),(), *CelebA_size),"CelebA7",100,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12),(3,3,3,3),(2,3,3), (),*CelebA_size),"CelebA10",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,20,24,28,32),(3,5,3,5,3,5,3,5),(1,2,1,2,1,2,1,2),(1024,1024), *CelebA_size),"CelebA9",100,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,20,24,28,32,36,40),(3,5,3,3,5,3,3,5,3,3),(1,1,2,1,1,2,1,1,2,1),(), *CelebA_size),"CelebA11",100,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA12",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA13",50,CelebA,Optimizer=Optim1),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA14",50,CelebA,Optimizer=Optim2),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA15",50,CelebA,Optimizer=Optim3),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,64,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA16",50,CelebA),
]

train_CelebA_vaes = [
    (*get_sym_fully_conv_vae((8,12,16,24,32,44,56,64),(5,3,3,3,3,3,3,3),(1,2,1,2,1,2,1,2),(2048,1024)), "CelebAVAE1",50,CelebA,Optim1),
]


if __name__ == "__main__":
    if False:
        trainees = train_CelebA_aes
        for trainae in trainees[-1:]:
            trainae.train()
            del trainae._encoder, trainae._decoder
            torch.cuda.empty_cache()
    else:
        for E,D,name,epochs,data,Optim in train_CelebA_vaes:
            e,d=E(),D()
            data_loader = get_data_loader(data)
            train_vae(e,d,data_loader,device,name, epochs,9,Optim)
