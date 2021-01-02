import logging
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import os

from utility import get_bunch, latent_space_pca, labeled_latent_space_pca, normal_to_pc, save_images, save_img, get_grid
from defaults import CKPT_PATH

logging.basicConfig(level=logging.INFO)


def train_autoencoder(encoder: nn.Module, decoder: nn.Module, train_loader, device, name, latent_size=4, lr=0.001, epochs=50, num_imgs=9):
    train_losses = []
    pca_batch = get_bunch(train_loader)
    latent_sample = torch.randn(size=(num_imgs, latent_size,), device=device)
    original_images = next(iter(train_loader))[0].to(device)
    original_images = original_images[:min(num_imgs, original_images.size()[0])]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    for epoch in tqdm.trange(epochs):  # loop over the dataset multiple times
        decoder = decoder.train()
        encoder = encoder.train()
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
        train_losses.append(running_loss/i)
        logging.info(f"epoch {epoch}: loss={running_loss/i}")
        decoder = decoder.eval()
        compressed_images=decoder(encoder(original_images)).detach().to("cpu")
        generated_images=decoder(latent_sample).detach().to("cpu")
        #mean, cov = latent_space_pca(encoder, train_loader)
        mean, cov = latent_space_pca(encoder, pca_batch)
        labeled_pca = labeled_latent_space_pca(encoder, pca_batch)
        # labeled_pca = {l: (m.to(device), c.to(device)) for l, (m, c) in labeled_pca.items()}  is on device
        pca_gen_images=decoder(normal_to_pc(latent_sample, mean.to(device), cov.to(device))).detach().to("cpu")
        labeled_pca_gen_images = {label: decoder(normal_to_pc(latent_sample, *labeled_pca[label])).detach().to("cpu") for label in labeled_pca}
        save_images(generated_images, compressed_images, pca_gen_images, labeled_pca_gen_images, name, epoch)
    save_img(get_grid(original_images.to("cpu")), os.path.join(CKPT_PATH, name, "compressed_images", "original.png"))
    return train_losses
