import logging
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import os
import matplotlib.pyplot as plt

from utility import get_bunch, get_stacked_bunch, latent_space_pca, labeled_latent_space_pca, normal_to_pc, save_images, save_img, get_grid, save_model, save_image, save_labeled_pca_gen_images
from defaults import CKPT_PATH

logging.basicConfig(level=logging.INFO)


def train_autoencoder(encoder: nn.Module, decoder: nn.Module, train_loader, device, name, latent_size=4, epochs=50, num_imgs=9, Optimizer=optim.Adam, normal_loss_factor=False, val_loader=None):
    os.makedirs(os.path.join(CKPT_PATH, name), exist_ok=True)
    with open(os.path.join(CKPT_PATH, name, "loss_log.txt"),"w") as loss_log:
        pass
    bunch = get_bunch(train_loader)
    latent_sample = torch.randn(size=(num_imgs, latent_size,), device=device)
    latent_sample[0] = 0
    original_images = next(iter(train_loader))[0].to(device)
    original_images = original_images[:min(num_imgs, original_images.size()[0])]
    save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_images", "original.png"))
    criterion = nn.MSELoss()
    scheduler = optimizer = Optimizer(list(encoder.parameters()) + list(decoder.parameters()))
    train_losses=[]
    train_norm_losses=[]
    val_losses=[]
    val_norm_losses=[]
    while True:
        try:
            optimizer = optimizer.optimizer
        except:
            break
    if scheduler == optimizer:
        scheduler = False

    for epoch in tqdm.trange(epochs):  # loop over the dataset multiple times
        train_loss, train_norm_loss = train_epoch(encoder, decoder, train_loader, device, criterion, normal_loss_factor, optimizer, scheduler)
        val_loss, val_norm_loss = val_epoch(encoder, decoder, val_loader, device, criterion, normal_loss_factor, 50)

        train_losses.append(train_loss)
        train_norm_losses.append(train_norm_loss)
        val_losses.append(val_loss)
        val_norm_losses.append(val_norm_loss)
        loss_msg = f"epoch {epoch}: train_loss={train_loss}, train_norm_loss={train_norm_loss}, val_loss={val_loss}, val_norm_loss={val_norm_loss}"
        logging.info(loss_msg)
        with open(os.path.join(CKPT_PATH, name, "loss_log.txt"),"a") as loss_log:
            loss_log.write(loss_msg)
        plot_loss(train_losses, train_norm_losses, val_losses, val_norm_losses, name)
        make_images(decoder, encoder, original_images, name, epoch, latent_sample, bunch, device)
        save_model(encoder, decoder, name)

def make_images(decoder, encoder, original_images, name, epoch, latent_sample, bunch, device):
    decoder.eval().to(device)
    save_image(decoder(encoder(original_images)).detach().to("cpu"), "compressed_images", name, epoch)
    save_image(decoder(latent_sample).detach().to("cpu"), "generated_images", name, epoch)        
    #mean, cov = latent_space_pca(encoder, train_loader)
    torch.cuda.empty_cache()
    mean, cov = latent_space_pca(encoder, bunch)
    save_image(decoder(normal_to_pc(latent_sample, mean.to(device), cov.to(device))).detach().to("cpu"), "pca_gen_images", name, epoch)
    save_labeled_pca_gen_images(encoder, decoder, latent_sample, bunch, name, epoch)

def plot_loss(train_losses, train_norm_losses, val_losses, val_norm_losses, name):
    plt.clf()
    plt.plot(range(len(train_losses)), train_losses, "b", label="train_loss")
    plt.plot(range(len(train_losses)), train_norm_losses, "r", label="train_norm_loss")
    plt.plot(range(len(train_losses)), val_losses, "y", label="val_loss")
    plt.plot(range(len(train_losses)), val_norm_losses, "g", label="val_norm_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(CKPT_PATH, name, "loss_plot.png"))

def train_epoch(encoder, decoder, train_loader, device, criterion, normal_loss_factor, optimizer, scheduler):
    torch.cuda.empty_cache()
    decoder.train().to(device)
    encoder.train().to(device)
    torch.cuda.empty_cache()
    running_loss = 0.0
    running_norm = 0.0
    for i, (inputs, labels) in enumerate(tqdm.tqdm(train_loader), 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs; data is a list of [inputs, labels]
        loss, norm = forward(inputs, device, encoder, decoder, criterion, normal_loss_factor)

        loss.backward()
        optimizer.step()

        # print statistics
        running_norm += norm.item()
        running_loss += loss.item()
    running_loss /= (i+1)
    running_norm /= (i+1)
    if scheduler:
        scheduler.step()
    return running_loss, running_norm

def forward(inputs, device, encoder, decoder, criterion, normal_loss):
    inputs = inputs.to(device=device)

    # forward + backward + optimize
    latent = encoder(inputs)
    outputs = decoder(latent)
    loss = criterion(outputs, inputs)
    norm = torch.mean(torch.square(torch.mean(latent,dim=0)))*normal_loss + torch.mean(torch.square(torch.var(latent,dim=0)-1))*normal_loss
    loss += norm
    return loss, norm

def val_epoch(encoder, decoder, val_loader, device, criterion, normal_loss_factor, max_iterations=None):
    if val_loader is None:
        return 0, 0
    torch.cuda.empty_cache()
    decoder.eval().to(device)
    encoder.eval().to(device)
    torch.cuda.empty_cache()
    running_loss = 0.0
    running_norm = 0.0
    for i, (inputs, labels) in enumerate(tqdm.tqdm(val_loader), 0):
        # get the inputs; data is a list of [inputs, labels]
        loss, norm = forward(inputs, device, encoder, decoder, criterion, normal_loss_factor)

        # print statistics
        running_norm += norm.item()
        running_loss += loss.item()
        if max_iterations is not None and i > max_iterations:
            break
    running_loss /= (i+1)
    running_norm /= (i+1)
    return running_loss, running_norm

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
        torch.cuda.empty_cache()
        mean, cov = latent_space_pca(encoder, bunch)
        save_image(decoder(normal_to_pc(latent_sample, mean.to(device), cov.to(device))).detach().to("cpu"), "pca_gen_images", name, epoch)
        del mean, cov
        torch.cuda.empty_cache()
        save_labeled_pca_gen_images(encoder, decoder, latent_sample, bunch, name, epoch)
        save_model(encoder, decoder, name)
    return train_losses
