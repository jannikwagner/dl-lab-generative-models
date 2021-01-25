from TrainModels import TrainAE
from datasets import getMNIST, getCIFAR10, getFashionMNIST, get_data_loader
from defaults import device, MNIST, FashionMNIST, CIFAR10, CelebA, CelebA_size, CIFAR10_size, CKPT_PATH
from models.MNIST.Autoencoder import *
from models.CIFAR10.Autoencoder import *
from train.Autoencoder import train_autoencoder, train_stacked_ae
from train.TrainVarAE import train_vae
import torch.nn as nn
import torch.optim as optim
from models.VarAE import get_sym_fully_conv_vae
from models.base import get_sym_resnet_ae, get_sym_ful_conv_ae2
from train.trainGAN import train_gan
import numpy as np
import os

from utility import save_model, extract_images_of_models, load_model, latent_space_pca, sample_in_pc, \
    plot_images2, normal_to_pc, get_sample_k_of_d, labeled_latent_space_pca, param_count, param_print, get_pc_grid_gen_images, save_img


Optim1 = lambda p: optim.lr_scheduler.StepLR(optim.Adam(p, 0.001), step_size=1, gamma=0.97)
Optim2 = lambda p: optim.lr_scheduler.StepLR(optim.Adam(p, 0.01), step_size=1, gamma=0.95)
Optim3 = lambda p: optim.lr_scheduler.StepLR(optim.SGD(p, 0.001, weight_decay=0.01, momentum=0.9), step_size=1, gamma=0.98)
Optim4 = lambda p: optim.lr_scheduler.StepLR(optim.Adam(p, 0.01), step_size=1, gamma=0.92)
Optim5 = lambda p: optim.lr_scheduler.StepLR(optim.Adam(p, 0.001), step_size=1, gamma=0.92)
Optim6 = lambda p: optim.lr_scheduler.StepLR(optim.Adam(p, 0.0001), step_size=1, gamma=0.92)

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
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(3,3,3),(1,1,1),100), "CIFAR10AE8", 100, CIFAR10),  # blurry, generation useless
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,16),(5,3,3),(2,1,1),100), "CIFAR10AE9", 100, CIFAR10),  # blurry, generation useless
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
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,),(3,),(1,),(2048,1536,1024), *CIFAR10_size), "CIFAR10NAE1", 100, CIFAR10,Optimizer=Optim1,normal_loss=1/10),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,12,16,32),(3,3,3,3,),(2,1,1,1,),(2048,1024), *CIFAR10_size), "CIFAR10AE28", 100, CIFAR10,Optimizer=Optim2),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,5,6,7,8,9,10,11,),(3,3,3,3,3,3,3,3,),(1,)*8,(2048,1024), *CIFAR10_size), "CIFAR10AE29", 100, CIFAR10,Optimizer=Optim2),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,5,6,7,),(3,)*4,(1,)*4,(1024,), *CIFAR10_size), "CIFAR10AE30", 100, CIFAR10,Optimizer=Optim2),  # gute reconstruction
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,512,1024),(3,5,3,3,3,3,3,3,),(1,2,1,1,1,1,1,1),(), *CIFAR10_size), "CIFAR10AE33", 100, CIFAR10,Optimizer=Optim2),  # up to 1 pixel no dense leayer?
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,512,1024),(3,5,3,3,3,3,3,3,),(1,2,1,1,1,1,1,1),(1024,), *CIFAR10_size), "CIFAR10AE32", 100, CIFAR10,Optimizer=Optim2), # up to 1 pixel
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,512,1024),(3,)*8,(1,2,1,1,1,1,1,1),(1024,), *CIFAR10_size), "CIFAR10AE31", 100, CIFAR10,Optimizer=Optim2),  # latent_size probably way too large
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,512,1024),(11,8,5,3,3,3,3,3,),(1,1,1,1,1,1,1,1),(1024,), *CIFAR10_size), "CIFAR10AE34", 100, CIFAR10,Optimizer=Optim2),

    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,5,6,7,),(3,)*4,(1,)*4,(1024,), *CIFAR10_size), "CIFAR10NAE1", 100, CIFAR10,Optimizer=Optim1, normal_loss=1),  # gut!!!]
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,5,6,),(3,)*3,(1,)*3,(512,256,128,64), *CIFAR10_size), "CIFAR10AE35", 50, CIFAR10,Optimizer=Optim4, normal_loss=False, label=1),
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
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA12",50,CelebA),  # bad idea
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA13",50,CelebA,Optimizer=Optim1),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA14",50,CelebA,Optimizer=Optim2),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,256,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA15",50,CelebA,Optimizer=Optim3),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,16,32,64,128,64,32),(3,3,3,3,3,3,3),(1,2,1,2,1,2,1),(1024,), *CelebA_size),"CelebA16",50,CelebA),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((8,12,16,24,32,44,56,64),(5,3,3,3,3,3,3,3),(1,2,1,2,1,2,1,2),(2048,),*CelebA_size), "CelebANAE1",50,CelebA,Optimizer=Optim1,normal_loss=1/100),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,32),(3,3,3,3,3),(1,2,1,2,1,),(1024,), *CelebA_size), "CelebAAE17",100,CelebA,Optimizer=Optim2),  # womöglich latent_space zu groß?
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,32),(3,3,3,3,3),(1,2,1,2,1,),(128,), *CelebA_size), "CelebAAE18",100,CelebA,Optimizer=Optim2),  # stärkere Generalisierung, sampling klappt besser
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,5,6),(3,3,3),(1,1,1),(256,192,128,), *CelebA_size), "CelebAAE21",50,CelebA,Optimizer=Optim4),  # pooling scheint notwendig zu sein, verrauschtere Bilder
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,6,8,10,12),(3,3,3,3,3),(1,1,1,2,1,),(256,128,), *CelebA_size), "CelebAAE20",50,CelebA,Optimizer=Optim4),
    TrainAE(*get_symmetric_fully_convolutional_autoencoder((4,8,12,16,32),(3,3,3,3,3),(1,2,1,2,1,),(1024,128,), *CelebA_size), "CelebAAE19",50,CelebA,Optimizer=Optim4),
    TrainAE(*get_sym_resnet_ae((3,4,4,8,8,12,12),(3,)*7,(1,2,1,2,1,2,1),(3,)*7,(256,)), "CelebARNAE4",50,CelebA,Optimizer=Optim4),
    TrainAE(*get_sym_ful_conv_ae2((4,8,12,16,20,24,32),(3,4,3,4,3,4,3),None,(1,2,1,2,1,2,1),(128,)),"CelebAAE22",50,CelebA,Optim4),
    TrainAE(*get_sym_ful_conv_ae2((4,8,16,24,32),(3,4,3,4,3,),None,(1,2,1,2,1,),(128,)),"CelebAAE23",50,CelebA,Optim4),  # works
    TrainAE(*get_sym_ful_conv_ae2((4,8,16,24,32),(4,)*5,None,(1,2,1,2,1,),(64,)),"CelebAAE24",50,CelebA,Optim4),  # works
    TrainAE(*get_sym_ful_conv_ae2((4,8,16,24,32),(4,)*5,None,(1,2,1,2,1,),(256,)),"CelebAAE25",50,CelebA,Optim4),  # overfitted
    TrainAE(*get_sym_ful_conv_ae2((4,8,16,24,32),(4,)*5,None,(1,2,1,2,1,),(32,)),"CelebAAE26",50,CelebA,Optim4),  # latent dim too low
    TrainAE(*get_sym_ful_conv_ae2((4,8,16,24,32),(3,4,3,4,3,),None,(1,2,1,2,1,),(100,),norm_layer=nn.InstanceNorm2d),"CelebAAE27",50,CelebA,Optim4),  # instance norm does not work well

]

train_CelebA_vaes = [
    (*get_sym_fully_conv_vae((8,12,16,24,32,44,56,64),(5,3,3,3,3,3,3,3),(1,2,1,2,1,2,1,2),(2048,)), "CelebAVAE1",50,CelebA,Optim1,0),
    (*get_sym_fully_conv_vae((8,12,16,24,32,44,56,64),(5,3,3,3,3,3,3,3),(1,2,1,2,1,2,1,2),(2048,)), "CelebAVAE3",50,CelebA,Optim1,1),
    (*get_sym_fully_conv_vae((8,12,16,24,32,44,56,64),(5,3,3,3,3,3,3,3),(1,2,1,2,1,2,1,2),(2048,)), "CelebAVAE2",50,CelebA,Optim1,0),
    (*get_sym_ful_conv_ae2((4,8,16,24,32),(4,4,4,4,4,),None,(1,2,1,2,1,),(1024,),vae=True), "CelebAVAE5",50,CelebA,Optim6,1)
]

train_CelebA_stacked_aes = [
    (*get_stacked_ful_conv_ae((8,12,16,24,32,44,56,64),(5,3,3,3,3,3,3,3),(1,2,1,2,1,2,1,2),(2048,),*CelebA_size), "CelebASAE1",50,CelebA,Optim1),
]

train_CIFAR10_vaes = [    
    (*get_sym_fully_conv_vae((4,5,6,7,),(3,)*4,(1,)*4,(1024,), *CIFAR10_size), "CIFAR10VAE2", 100, CIFAR10,Optim1,1),  # gut!!!
    (*get_sym_fully_conv_vae((4,5,6,7,),(3,)*4,(1,)*4,(1024,), *CIFAR10_size), "CIFAR10VAE1s", 100, CIFAR10,Optim1,0),  # gut!!!
]

def fun(model, d0,d1,name):
    images = get_pc_grid_gen_images(model.decoder,*model.pca, d0,d1,rand=False)
    save_img(images, os.path.join(CKPT_PATH, name,f"{name}grid{d0}_{d1}.png"))
    images = get_pc_grid_gen_images(model.decoder,*model.pca, d0,d1,rand=True)
    save_img(images, os.path.join(CKPT_PATH, name,f"{name}grid{d0}_{d1}r.png"))

def train():
    mode = None
    if mode == "gan":
        D = get_sym_ful_conv_ae2((4,8,16,24,32,48,64),(4,)*7,None,(1,2,1,2,1,2,1),(32,1),enc_fn=nn.Sigmoid)[0]
        G = get_sym_ful_conv_ae2((4,8,16,24,32),(4,4,4,4,4),None,(1,2,1,2,1),(256,))[1]
        d,g = D(),G()
        train_gan(d,g,"CelebAGAN2",get_data_loader(CelebA,split="train"),get_data_loader(CelebA,split="test"),10,9,Optim5,0,1)
    elif mode == "ae":
        for tae in train_CelebA_aes[-1:]:
            tae.train()
            del tae._encoder, tae._decoder
            torch.cuda.empty_cache()
    elif mode == "vae":
        for E,D,name,epochs,data,Optim,loss_type in train_CelebA_vaes[-1:]:
            train_vae(E(),D(),get_data_loader(data), device, name, epochs, 9, Optim, loss_type,get_data_loader(data,split="test"))

def get_grids():
    for model in train_CelebA_aes + mnist_train_aes+fashionMnist_train_aes+train_cifar10_aes:
        try:
            print(model.name)
            torch.cuda.empty_cache()
            name = model.name
            path = os.path.join(CKPT_PATH, name, )
            latent_size = model.encoder.latent_size
            fun(model,0,1,name)
            d0 = np.random.randint(0,latent_size)
            d1 = np.random.randint(0,latent_size)
            fun(model,d0,d1,name)
            fun(model,2,3,name)
        except Exception as e:
            print(model.name, "failed:", e)
        finally:
            del model._decoder, model._encoder

if __name__ == "__main__":
    pass
