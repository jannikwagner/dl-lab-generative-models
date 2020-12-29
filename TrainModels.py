import os

import torch

from datasets import get_data_loader
from defaults import CKPT_PATH, device
from train.Autoencoder import train_autoencoder
from utility import latent_space_pca, save_model, extract_images_of_models, labeled_latent_space_pca


class TrainAE:
    def __init__(self, encoder_class, decoder_class, name, epochs, dataset_name):
        self._encoder_class = encoder_class
        self._decoder_class = decoder_class
        self.name = name
        self.epochs = epochs
        self._encoder = None
        self._decoder = None
        self._mean = None
        self._cov = None
        self._dataset_name = dataset_name
        self._data_loader = None
        self._trained = os.path.exists(os.path.join(CKPT_PATH, name))
        self._labeled_pca = None

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self._encoder_class().to(device)
            if self._trained:
                self._encoder.load_state_dict(torch.load(os.path.join(CKPT_PATH, self.name, "encoder.pth")))
        return self._encoder

    @property
    def decoder(self):
        if self._decoder is None:
            self._decoder = self._decoder_class().to(device)
            if self._trained:
                self._decoder.load_state_dict(torch.load(os.path.join(CKPT_PATH, self.name, "decoder.pth")))
        return self._decoder

    @property
    def pca(self):
        if self._mean is None or self._cov is None:
            path = os.path.join(CKPT_PATH, self.name, "pca.pth")
            if os.path.exists(path):
                pca = torch.load(path)
                self._mean, self._cov = pca[0].to(device), pca[1].to(device)
            elif self._trained:
                self.make_pca()
        return self._mean, self._cov

    @property
    def labeled_pca(self):
        if self._labeled_pca is None:
            path = os.path.join(CKPT_PATH, self.name, "labeled_pca.pth")
            if os.path.exists(path):
                labeled_pca = torch.load(path)
                self._labeled_pca = {l: (m.to(device), c.to(device)) for l, (m, c) in labeled_pca.items()}
            elif self._trained:
                self.make_labeled_pca()
        return self._mean, self._cov

    def make_pca(self):
        assert self._trained
        pca = latent_space_pca(self.encoder, self.data_loader)
        self._mean, self._cov = pca
        pca_cpu = pca[0].to("cpu"), pca[1].to("cpu")
        torch.save(pca_cpu, os.path.join(CKPT_PATH, self.name, "pca.pth"))

    def make_labeled_pca(self):
        assert self._trained
        labeled_pca = labeled_latent_space_pca(self.encoder, self.data_loader)
        self._labeled_pca = labeled_pca
        labeled_pca = {l: (m.to("cpu"), c.to("cpu")) for l, (m, c) in labeled_pca.items()}
        # everything should be saved for cpu
        torch.save(labeled_pca, os.path.join(CKPT_PATH, self.name, "labeled_pca.pth"))

    @property
    def data_loader(self):
        if self._data_loader is None:
            self._data_loader = get_data_loader(self._dataset_name)
        return self._data_loader

    def train(self):
        self._decoder = self._decoder_class().to(device).train()
        self._encoder = self._encoder_class().to(device).train()
        gen_images, compressed_images, pca_gen_images, labeled_pca_gen_images, loss =\
            train_autoencoder(self.encoder, self.decoder, self.data_loader, device,
                              self.encoder.latent_size, epochs=self.epochs)
        self._trained = True
        save_model(self.encoder, self.decoder, self.name, gen_images, compressed_images, pca_gen_images, labeled_pca_gen_images, loss)
        extract_images_of_models(self.name)
        self.make_pca()
        self.make_labeled_pca()
