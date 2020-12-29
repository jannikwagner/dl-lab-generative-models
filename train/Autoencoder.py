import logging
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm

from utility import latent_space_pca, labeled_latent_space_pca, normal_to_pc

logging.basicConfig(level=logging.INFO)


def train_autoencoder(encoder: nn.Module, decoder: nn.Module, train_loader, device, latent_size=4, lr=0.001, epochs=50, save_images=16):
    generated_images = []
    pca_gen_images = []
    labeled_pca_gen_images = []
    compressed_images = []
    train_losses = []
    latent_sample = torch.randn(size=(save_images, latent_size,), device=device)
    original = next(iter(train_loader))[0].to(device)
    original = original[:min(save_images, original.size()[0])]
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
        compressed_images.append(decoder(encoder(original)).detach())
        generated_images.append(decoder(latent_sample).detach())
        mean, cov = latent_space_pca(encoder, train_loader)
        labeled_pca = labeled_latent_space_pca(encoder, train_loader)
        # labeled_pca = {l: (m.to(device), c.to(device)) for l, (m, c) in labeled_pca.items()}  is on device
        pca_gen_images.append(decoder(normal_to_pc(latent_sample, mean.to(device), cov.to(device))).detach())
        labeled_pca_gen_images.append({label: decoder(normal_to_pc(latent_sample, *labeled_pca[label])).detach() for label in labeled_pca})

    return generated_images, compressed_images, pca_gen_images, labeled_pca_gen_images, train_losses
