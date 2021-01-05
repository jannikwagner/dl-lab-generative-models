import os

import torch
import torchvision
from matplotlib import pyplot as plt
import torch.nn as nn
import math
from defaults import CKPT_PATH, device


def param_count(module):
    x = 0
    for p in module.parameters():
        y = 1
        for a in p.size():
            y *= a
        x += y
    return x


def param_print(module):
    for x in module.parameters():
        print(x.size())


def delete_pth(path, files=("gen_images.pth", "compressed_images.pth", "pca_gen_images.pth", "labeled_pca_gen_images.pth")):
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            os.remove(os.path.join(path, file))
        

def save_model(encoder, decoder, name, loss=None):
    os.makedirs(os.path.join(CKPT_PATH, name), exist_ok=True)
    torch.save(decoder.to("cpu").state_dict(), os.path.join(CKPT_PATH, name, "decoder.pth"))
    torch.save(encoder.to("cpu").state_dict(), os.path.join(CKPT_PATH, name, "encoder.pth"))
    if loss is not None:
        with open(os.path.join(CKPT_PATH, name, "loss.txt"),"w") as file:
            file.write("\n".join([str(x) for x in loss]))


def extract_images_of_models(name, rows=None):
    for img_type in ["gen_images", "compressed_images", "pca_gen_images"]:
        path = os.path.join(CKPT_PATH, name, img_type)
        images_pth_path = os.path.join(CKPT_PATH, name, f"{img_type}.pth")
        if os.path.isfile(images_pth_path):
            os.makedirs(path, exist_ok=True)
            images = torch.load(images_pth_path)
            for epoch_num, epoch_images in enumerate(images):
                grid = get_grid(epoch_images, rows)
                if img_type == "compressed_images" and epoch_num == len(images)-1:
                    image_path = os.path.join(path, "original.png")
                else:
                    image_path = os.path.join(path, f"{epoch_num}.png")
                save_img(grid, image_path)
    img_type = "labeled_pca_gen_images"
    path = os.path.join(CKPT_PATH, name, img_type)
    images_pth_path = os.path.join(CKPT_PATH, name, f"{img_type}.pth")
    if os.path.isfile(images_pth_path):
        os.makedirs(path, exist_ok=True)
        images = torch.load(images_pth_path)
        for epoch_num, epoch_images in enumerate(images):
            for label, label_images in epoch_images.items():
                os.makedirs(os.path.join(path, str(label)), exist_ok=True)
                grid = get_grid(label_images, rows)
                image_path = os.path.join(path, str(label), f"{epoch_num}.png")
                save_img(grid, image_path)


def save_images(gen_images, compressed_images, pca_gen_images, labeled_pca_gen_images, name, epoch):
    path = os.path.join(CKPT_PATH, name)
    for img_type, images in zip(["gen_images", "compressed_images", "pca_gen_images"], (gen_images, compressed_images, pca_gen_images)):
        grid = get_grid(images)
        image_path = os.path.join(path, img_type, f"{epoch}.png")
        save_img(grid, image_path)
    for label, label_images in labeled_pca_gen_images.items():
        grid = get_grid(label_images)
        image_path = os.path.join(path, "labeled_pca_gen_images", str(label), f"{epoch}.png")
        save_img(grid, image_path)


def save_image(images, img_type, name, epoch):
    path = os.path.join(CKPT_PATH, name, img_type)
    grid = get_grid(images)
    image_path = os.path.join(path, f"{epoch}.png")
    save_img(grid, image_path)


def save_img(img: torch.Tensor, image_path):
    os.makedirs(os.path.split(image_path)[0], exist_ok=True)
    if len(img.size()) == 4:
        img = img[0]
    c = img.size()[0]
    img = img.transpose(0,1).transpose(1,2)
    img = img / 2 + 0.5
    if c == 1:
        plt.imsave(image_path, img[:, :, 0], cmap="gray_r")
    else:
        plt.imsave(image_path, img.numpy())


def show_img(img):
    if len(img.size()) == 4:
        img = img[0]
    c = img.size()[0]
    img = img.transpose(0,1).transpose(1,2)
    img = img / 2 + 0.5
    if c == 1:
        plt.imshow(img[:, :, 0], cmap="gray_r")
    else:
        plt.imshow(img)


def get_grid(images, rows=None):
    if rows is None:
        rows = int(math.sqrt(images.size()[0]))
    return torchvision.utils.make_grid(images, rows)[:images.size()[1]]


def get_bunch(train_loader, n_batches=100):
    bunch = []
    iterator = iter(train_loader)
    for i in range(n_batches):
        try:
            input, labels = next(iterator)
        except StopIteration:
            break
        bunch.append((input.to(device), labels.to(device)))
    return bunch


def get_stacked_bunch(train_loader, n_batches=100):  # maybe unecessary
    bunch = get_bunch(train_loader, n_batches)
    return stack_bunch(bunch)

def stack_bunch(bunch):
    bunch, labels = [x[0] for x in bunch], [x[1] for x in bunch]
    bunch, labels = torch.stack(bunch), torch.stack(labels)
    size = bunch.size()
    bunch = bunch.view(size[0]*size[1], *size[2:])
    labels = labels.view(size[0]*size[1], *labels.size()[2:])
    return bunch, labels


def latent_space_pca_stacked_bunch(encoder: nn.Module, stacked_bunch):
    """
    Does use way to much memory -> non functional
    """
    encoder = encoder.to(device).eval()
    latent_space = encoder(stacked_bunch)
    size = latent_space.size()
    latent_space = latent_space.view(size[0]*size[1], size[2])
    return pca(latent_space)


def latent_space_pca(encoder: nn.Module, train_loader, n_batches=100):  # everything is transposed compared to what i am used to
    latent_space = []
    encoder = encoder.to(device).eval()
    iterator = iter(train_loader)
    for i in range(n_batches):
        try:
            input, labels = next(iterator)
        except StopIteration:
            break
        latent_space.append(encoder(input.to(device)).detach())

    latent_space = torch.stack(latent_space)
    size = latent_space.size()
    latent_space = latent_space.view(size[0]*size[1], size[2])
    return pca(latent_space)


def save_labeled_pca_gen_images(encoder, decoder, latent_sample, bunch, name, epoch):
    encoder=encoder.to(device).eval()
    latent_bunch = [(encoder(input.to(device)).detach(), labels) for input, labels in bunch]
    latent_stacked_bunch, labels = stack_bunch(latent_bunch)
    if len(labels.size()) > 1:
        labels = labels[:,0]
    for label in torch.unique(labels):
        torch.cuda.empty_cache()
        grid = get_images_labeled_space_pca(decoder, latent_sample, latent_stacked_bunch, labels, label)
        save_img(grid, os.path.join(CKPT_PATH, name, "labeled_pca_gen_images", str(label.item()), f"{epoch}.png"))


def get_images_labeled_space_pca(decoder, latent_sample, latent_stacked_bunch, labels, label):
    labeled_latent_stacked_bunch = latent_stacked_bunch[labels == label]
    mean, cov = pca(labeled_latent_stacked_bunch)
    images = decoder(normal_to_pc(latent_sample, mean, cov)).detach().to("cpu")
    grid = get_grid(images)
    return grid


def labeled_latent_space_pca(encoder: nn.Module, train_loader, n_batches=100):  # everything is transposed compared to what i am used to
    labeled_latent_space = dict()
    encoder = encoder.to(device).eval()  # on cpu
    iterator = iter(train_loader)
    for i in range(n_batches):
        try:
            input, labels = next(iterator)
        except StopIteration:
            break
        labels = torch.stack([torch.as_tensor(label)[0] for label in labels])  # 
        new_ls_samples = encoder(input.to(device)).detach()
        for label in torch.unique(labels):
            label = label.item()
            label_ls_samples = new_ls_samples[labels == label]
            if label not in labeled_latent_space:
                labeled_latent_space[label] = []
            labeled_latent_space[label].append(label_ls_samples)
    for label in labeled_latent_space:
        count = sum(batch.size()[0] for batch in labeled_latent_space[label])
        ls = torch.zeros(count, encoder.latent_size, device=device)
        i = 0
        for batch in labeled_latent_space[label]:
            ls[i:i+batch.size()[0], :] = batch
            i += batch.size()[0]
        labeled_latent_space[label] = ls
    #label_ls = {label: torch.stack(label_ls[label]).view(-1, encoder.latent_size).detach() for label in label_ls}
    return {label: pca(labeled_latent_space[label]) for label in labeled_latent_space}


def pca(space):
    mean = space.mean(axis=0).view(1, -1)
    space = space - mean
    svd = torch.svd(space.transpose(1, 0) @ space / space.size()[0])
    cov = torch.diag(torch.sqrt(svd[1])) @ svd[0].transpose(0, 1)
    return mean, cov


def load_model(Encoder, Decoder, name):
    encoder = Encoder().to("cpu")
    decoder = Decoder().to("cpu")
    encoder.load_state_dict(torch.load(os.path.join(CKPT_PATH, name, "encoder.pth")))
    decoder.load_state_dict(torch.load(os.path.join(CKPT_PATH, name, "decoder.pth")))
    return encoder.eval(), decoder.eval()


def sample_in_pc(num, mean, cov):
    X = torch.randn(num, mean.size()[1])
    return normal_to_pc(X, mean, cov)


def normal_to_pc(X, mean, cov):
    return X @ cov + mean


def plot_images(images, ri=4, ci=4):  # bad
    for i in range(min(ri*ci,len(images))):
        ax = plt.subplot(ri, ci, i + 1)
        show_img(images[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_images2(images, rows=None):  # good
    grid = get_grid(images, rows)
    show_img(grid)
    plt.show()


def fill_to_d(X, d, val=0):
    M = torch.zeros(X.size()[0], d) + val
    M[:, :X.size()[1]] = X
    return M


def get_sample_k_of_d(num, k, d, val=0):
    X = torch.zeros(num, d) + val
    for i in range(k):
       X[:,i] = torch.rand(num)
    return X
