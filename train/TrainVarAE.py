import logging
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import os

from utility import get_bunch, get_stacked_bunch, latent_space_pca, labeled_latent_space_pca, normal_to_pc, save_images, save_img, get_grid, save_model, save_image, save_labeled_pca_gen_images
from defaults import CKPT_PATH

logging.basicConfig(level=logging.DEBUG)


def train_vae(encoder: nn.Module, decoder: nn.Module, train_loader, device, name,  epochs=50, num_imgs=9, Optimizer=optim.Adam, loss_type=0):
    with open(os.path.join(CKPT_PATH, name, "loss_log.txt"),"w") as loss_log:
        latent_sample = torch.randn(size=(num_imgs, encoder.latent_size,), device=device)
        original_images = next(iter(train_loader))[0].to(device)
        original_images = original_images[:min(num_imgs, original_images.size()[0])]
        save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_images", "original.png"))
        save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_sampled_images", "original.png"))
        criterion = nn.MSELoss()
        scheduler = optimizer = Optimizer(list(encoder.parameters()) + list(decoder.parameters()))
        log_scale = nn.Parameter(torch.Tensor([0.0])).to(device)
        while True:
            try:
                optimizer = optimizer.optimizer
            except:
                break
        if scheduler == optimizer:
            scheduler = False
        for epoch in tqdm.trange(epochs):  # loop over the dataset multiple times
            #torch.cuda.empty_cache()
            decoder = decoder.train().to(device)
            encoder = encoder.train().to(device)
            running_loss = 0.0
            running_kl = 0.0
            logging.debug(f"Epoch: {epoch}")
            for i, data in tqdm.tqdm(enumerate(train_loader, 0)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device=device)

                # zero the parameter gradients
                optimizer.zero_grad()

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
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_kl += kl.mean().item()
            if scheduler:
                scheduler.step()
            logging.info(f"epoch {epoch}: loss={running_loss/i}, kl={running_kl/i}")
            loss_log.write(f"epoch {epoch}: loss={running_loss/i}, kl={running_kl/i}\n")
            decoder = decoder.eval()
            mean, log_var = encoder(original_images)
            sample_ls = torch.distributions.Normal(mean, torch.exp(log_var/2)).sample()
            save_image(decoder(mean).detach().to("cpu"), "compressed_images", name, epoch)
            save_image(decoder(sample_ls).detach().to("cpu"), "compressed_sampled_images", name, epoch)
            save_image(decoder(latent_sample).detach().to("cpu"), "generated_images", name, epoch)       
            save_model(encoder, decoder, name)


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
