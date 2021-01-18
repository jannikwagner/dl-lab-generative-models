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


def train_vae(encoder: nn.Module, decoder: nn.Module, train_loader, device, name,  epochs=50, num_imgs=9, Optimizer=optim.Adam, loss_type=0, val_loader=None):
    os.makedirs(os.path.join(CKPT_PATH, name), exist_ok=True)
    with open(os.path.join(CKPT_PATH, name, "loss_log.txt"),"w") as loss_log:
        pass
    latent_sample = torch.randn(size=(num_imgs, encoder.latent_size,), device=device)
    latent_sample[0] = 0
    original_images = next(iter(train_loader))[0].to(device)
    original_images = original_images[:min(num_imgs, original_images.size()[0])]
    save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_images", "original.png"))
    save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_sampled_images", "original.png"))
    criterion = nn.MSELoss()
    scheduler = optimizer = Optimizer(list(encoder.parameters()) + list(decoder.parameters()))
    log_scale = nn.Parameter(torch.Tensor([0.0])).to(device)
    train_losses = []
    train_kls = []
    val_losses = []
    val_kls = []
    while True:
        try:
            optimizer = optimizer.optimizer
        except:
            break
    if scheduler == optimizer:
        scheduler = False

    for epoch in tqdm.trange(epochs):  # loop over the dataset multiple times
        train_loss, train_kl = train_epoch(decoder, device, encoder, train_loader, loss_type, criterion, log_scale, optimizer, scheduler)
        val_loss, val_kl = val_epoch(decoder, device, encoder, val_loader, loss_type, criterion, log_scale)

        train_losses.append(train_loss)
        train_kls.append(train_kl)
        val_losses.append(val_loss)
        val_kls.append(val_kl)
        loss_msg = f"epoch {epoch}: train_loss={train_loss}, train_kl={train_kl}, val_loss={val_loss}, val_kl={val_kl}"
        logging.info(loss_msg)
        with open(os.path.join(CKPT_PATH, name, "loss_log.txt"),"a") as loss_log:
            loss_log.write(loss_msg)
        plot_loss(train_losses, train_kls, val_losses, val_kls, name)
        make_images(decoder, device, encoder, original_images, name, epoch, latent_sample)     
        save_model(encoder, decoder, name)

def make_images(decoder, device, encoder, original_images, name, epoch, latent_sample):
    decoder.eval().to(device)
    mean, log_var = encoder(original_images)
    sample_ls = torch.distributions.Normal(mean, torch.exp(log_var/2)).sample()
    save_image(decoder(mean).detach().to("cpu"), "compressed_images", name, epoch)
    save_image(decoder(sample_ls).detach().to("cpu"), "compressed_sampled_images", name, epoch)
    save_image(decoder(latent_sample).detach().to("cpu"), "generated_images", name, epoch)  

def plot_loss(train_losses, train_kls, val_losses, val_kls, name):
    plt.clf()
    plt.plot(range(len(train_losses)), train_losses, "b", label="train_loss")
    plt.plot(range(len(train_losses)), train_kls, "r", label="train_kl")
    plt.plot(range(len(train_losses)), val_losses, "y", label="val_loss")
    plt.plot(range(len(train_losses)), val_kls, "g", label="val_kl")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(CKPT_PATH, name, "loss_plot.png"))

def train_epoch(decoder, device, encoder, data_loader, loss_type, criterion, log_scale, optimizer, scheduler):
    decoder.train().to(device)
    encoder.train().to(device)
    running_loss = 0.0
    running_kl = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(data_loader, 0)):
        # zero the parameter gradients
        optimizer.zero_grad()
        loss, kl = forward(inputs, device, encoder, decoder, loss_type, criterion, log_scale)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_kl += kl.item()
    running_loss /= (i+1)
    running_kl /= (i+1)
    if scheduler:
        scheduler.step()
    return running_loss, running_kl

def val_epoch(decoder, device, encoder, data_loader, loss_type, criterion, log_scale):
    if data_loader is None:
        return 0,0
    decoder.eval().to(device)
    encoder.eval().to(device)
    running_loss = 0.0
    running_kl = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(data_loader, 0)):
        loss, kl = forward(inputs, device, encoder, decoder, loss_type, criterion, log_scale)

        running_loss += loss.item()
        running_kl += kl.item()
    running_loss /= (i+1)
    running_kl /= (i+1)
    return running_loss, running_kl

def forward(inputs, device, encoder, decoder, loss_type, criterion, log_scale):
    inputs = inputs.to(device=device)

    # forward + backward + optimize
    mean, log_var = encoder(inputs)
    scale = torch.exp(log_var/2)
    dist = torch.distributions.Normal(mean, scale)
    sample = dist.rsample()
    outputs = decoder(sample)
    if loss_type == 0:
        kl = torch.mean(-0.5 * torch.mean(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = criterion(outputs, inputs) + kl
    if loss_type == 1:
        recon_loss = gaussian_likelihood(outputs, log_scale, inputs)

        # kl
        kl = kl_divergence(sample, mean, scale)

        # elbo
        loss = (kl - recon_loss)
        loss = loss.mean()
        kl = kl.mean()
    return loss, kl


def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))

def kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl
