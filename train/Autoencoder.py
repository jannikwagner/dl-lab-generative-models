import logging
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import os

from utility import get_bunch, get_stacked_bunch, latent_space_pca, labeled_latent_space_pca, normal_to_pc, save_images, save_img, get_grid, save_model, save_image, save_labeled_pca_gen_images
from defaults import CKPT_PATH

logging.basicConfig(level=logging.DEBUG)


def train_autoencoder(encoder: nn.Module, decoder: nn.Module, train_loader, device, name, latent_size=4,  epochs=50, num_imgs=9, Optimizer=optim.Adam):
    train_losses = []
    bunch = get_bunch(train_loader)
    latent_sample = torch.randn(size=(num_imgs, latent_size,), device=device)
    latent_sample[0] = 0
    original_images = next(iter(train_loader))[0].to(device)
    original_images = original_images[:min(num_imgs, original_images.size()[0])]
    save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_images", "original.png"))
    criterion = nn.MSELoss()
    scheduler = optimizer = Optimizer(list(encoder.parameters()) + list(decoder.parameters()))
    while True:
        try:
            optimizer = optimizer.optimizer
        except:
            break
    if scheduler == optimizer:
        scheduler = False
    for epoch in tqdm.trange(epochs):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        decoder = decoder.train().to(device)
        encoder = encoder.train().to(device)
        running_loss = 0.0
        print(epoch)
        for i, data in tqdm.tqdm(enumerate(train_loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = decoder(encoder(inputs))
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if scheduler:
            scheduler.step()
        train_losses.append(running_loss/i)
        logging.info(f"epoch {epoch}: loss={running_loss/i}")
        decoder = decoder.eval()
        save_image(decoder(encoder(original_images)).detach().to("cpu"), "compressed_images", name, epoch)
        save_image(decoder(latent_sample).detach().to("cpu"), "generated_images", name, epoch)        
        #mean, cov = latent_space_pca(encoder, train_loader)
        torch.cuda.empty_cache()
        mean, cov = latent_space_pca(encoder, bunch)
        save_image(decoder(normal_to_pc(latent_sample, mean.to(device), cov.to(device))).detach().to("cpu"), "pca_gen_images", name, epoch)
        del mean, cov
        torch.cuda.empty_cache()
        save_labeled_pca_gen_images(encoder, decoder, latent_sample, bunch, name, epoch)
        save_model(encoder, decoder, name)
    return train_losses

def train_stacked_ae(encoder: nn.Module, decoder: nn.Module, train_loader, device, name, latent_size=4,  epochs=50, num_imgs=9, Optimizer=optim.Adam):
    train_losses = []
    bunch = get_bunch(train_loader)
    latent_sample = torch.randn(size=(num_imgs, latent_size,), device=device)
    latent_sample[0] = 0
    original_images = next(iter(train_loader))[0].to(device)
    original_images = original_images[:min(num_imgs, original_images.size()[0])]
    save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_images", "original.png"))
    criterion = nn.MSELoss()
    scheduler = optimizer = Optimizer(list(encoder.parameters()) + list(decoder.parameters()))
    while True:
        try:
            optimizer = optimizer.optimizer
        except:
            break
    if scheduler == optimizer:
        scheduler = False
    for epoch in tqdm.trange(epochs):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        decoder = decoder.train().to(device)
        encoder = encoder.train().to(device)
        running_loss = 0.0
        print(epoch)
        for i, data in tqdm.tqdm(enumerate(train_loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = decoder(encoder(inputs, epoch), epoch)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if scheduler:
            scheduler.step()
        train_losses.append(running_loss/i)
        logging.info(f"epoch {epoch}: loss={running_loss/i}")
        decoder = decoder.eval()
        save_image(decoder(encoder(original_images, epoch), epoch).detach().to("cpu"), "compressed_images", name, epoch)
        save_image(decoder(latent_sample).detach().to("cpu"), "generated_images", name, epoch)        
        #mean, cov = latent_space_pca(encoder, train_loader)
        torch.cuda.empty_cache()
        mean, cov = latent_space_pca(encoder, bunch)
        save_image(decoder(normal_to_pc(latent_sample, mean.to(device), cov.to(device))).detach().to("cpu"), "pca_gen_images", name, epoch)
        del mean, cov
        torch.cuda.empty_cache()
        save_labeled_pca_gen_images(encoder, decoder, latent_sample, bunch, name, epoch)
        save_model(encoder, decoder, name)
    return train_losses
