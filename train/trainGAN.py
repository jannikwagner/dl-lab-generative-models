import logging
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import os
import matplotlib.pyplot as plt

from utility import get_bunch, get_stacked_bunch, latent_space_pca, labeled_latent_space_pca, normal_to_pc, save_images, save_img, get_grid, save_model, save_image, save_labeled_pca_gen_images
from defaults import CKPT_PATH, device

logging.basicConfig(level=logging.INFO)


def train_gan(discriminator: nn.Module, generator: nn.Module, name, train_loader, val_loader=None, epochs=50, num_imgs=9, Optimizer_fn=optim.Adam, loss_type=0, k=1):
    os.makedirs(os.path.join(CKPT_PATH, name), exist_ok=True)
    with open(os.path.join(CKPT_PATH, name, "loss_log.txt"),"w") as loss_log:
        pass
    latent_sample = torch.randn(size=(num_imgs, generator.latent_size,), device=device)
    latent_sample[0] = 0
    losses = dict(
        train_d_real_losses=[],
        train_d_fake_losses=[],
        train_g_losses=[],
        val_d_real_losses=[],
    )
    d_optimizer, d_scheduler = get_optimizer_scheduler(Optimizer_fn, discriminator.parameters())
    g_optimizer, g_scheduler = get_optimizer_scheduler(Optimizer_fn, generator.parameters())

    for epoch in tqdm.trange(epochs):  # loop over the dataset multiple times
        train_d_real_loss, train_d_fake_loss, train_g_loss = train_epoch(discriminator, generator, train_loader, device, d_optimizer, d_scheduler, g_optimizer, g_scheduler, loss_type, k)
        val_d_real_loss = val_epoch(discriminator, generator, val_loader, device, loss_type, 50)

        losses["train_d_real_losses"].append(train_d_real_loss)
        losses["train_d_fake_losses"].append(train_d_fake_loss)
        losses["train_g_losses"].append(train_g_loss)
        losses["val_d_real_losses"].append(val_d_real_loss)
        loss_msg = f"epoch {epoch}: train_d_real_loss={train_d_real_loss}, train_d_fake_loss={train_d_fake_loss}, train_g_loss={train_g_loss}, val_d_real_loss={val_d_real_loss}"
        logging.info(loss_msg)
        with open(os.path.join(CKPT_PATH, name, "loss_log.txt"),"a") as loss_log:
            loss_log.write(loss_msg)
        plot_losses(losses, name)
        generator.eval().to(device)
        save_image(generator(latent_sample).detach().to("cpu"), "generated_images", name, epoch)   
        save_model(discriminator, generator, name)

def get_optimizer_scheduler(Optimizer, params):
    scheduler = optimizer = Optimizer(params)
    while True:
        try:
            optimizer = optimizer.optimizer
        except:
            break
    if scheduler == optimizer:
        scheduler = False
    return optimizer, scheduler

def plot_losses(losses, name):
    plt.clf()
    for key, values in losses.items():
        plt.plot(range(len(values)), values, label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(CKPT_PATH, name, "loss_plot.png"))

def train_epoch(discriminator, generator, train_loader, device, d_optimizer, d_scheduler, g_optimizer, g_scheduler, loss_type=0, k=1):
    torch.cuda.empty_cache()
    discriminator.train().to(device)
    generator.train().to(device)
    torch.cuda.empty_cache()
    running_d_fake_loss = 0.0
    running_d_real_loss = 0.0
    running_g_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm.tqdm(train_loader), 0):
        # zero the parameter gradients
        d_optimizer.zero_grad()

        # discriminator
        inputs = inputs.to(device)
        d_real_loss = -torch.mean(torch.log(discriminator(inputs)))
        latent_samples = torch.randn(inputs.size()[0], generator.latent_size, device=device)
        fake_images_predictions = discriminator(generator(latent_samples))
        d_fake_loss = -torch.mean(torch.log(1-fake_images_predictions))
        d_loss = d_fake_loss + d_real_loss
        d_loss.backward()
        d_optimizer.step()

        # generator
        if (i-1) % k == 0:
            if not loss_type==1:
                latent_samples = torch.randn(inputs.size()[0], generator.latent_size, device=device)
            fake_images_predictions = discriminator(generator(latent_samples))
            g_optimizer.zero_grad()
            g_loss = - torch.mean(torch.log(fake_images_predictions))
            g_loss.backward()
            g_optimizer.step()
            running_g_loss += g_loss.item()

        # print statistics
        running_d_fake_loss += d_fake_loss.item()
        running_d_real_loss += d_real_loss.item()
    running_d_fake_loss /= (i+1)
    running_d_real_loss /= (i+1)
    running_g_loss /= (i+1)//k
    if d_scheduler:
        d_scheduler.step()
    if g_scheduler:
        g_scheduler.step()
    return running_d_real_loss, running_d_fake_loss, running_g_loss

def val_epoch(discriminator, generator, val_loader, device, loss_type, max_iterations=None):
    if val_loader is None:
        return 0, 0
    torch.cuda.empty_cache()
    generator.eval().to(device)
    discriminator.eval().to(device)
    torch.cuda.empty_cache()
    running_d_real_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm.tqdm(val_loader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = inputs.to(device)
        d_real_loss = -torch.mean(torch.log(discriminator(inputs)))

        # print statistics
        running_d_real_loss += d_real_loss.item()
        if max_iterations is not None and i > max_iterations:
            break
    running_d_real_loss /= (i+1)
    return running_d_real_loss
