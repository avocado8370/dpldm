import argparse
import os
import pickle
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm

from models.diffusion.ddpm import UNetModel, GaussianDiffusion
from train.encode import load_first_stage_model, load_first_stage


def load_second_stage_model(conf):
    """
    :param conf:
    :return:
    """
    u_model = UNetModel(
        in_channels=conf.unetconfig.in_channels,
        model_channels=conf.unetconfig.model_channels,
        out_channels=conf.unetconfig.out_channels,
        num_res_blocks=conf.unetconfig.num_res_blocks,
        channel_mult=conf.unetconfig.channel_mult,
        attention_resolutions=conf.unetconfig.attention_resolutions,
        conv_resample=conf.unetconfig.conv_resample
    )
    u_model.to(conf.device)
    timesteps = conf.timesteps
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    optimizer = torch.optim.Adam(u_model.parameters(), lr=conf.lr)
    return gaussian_diffusion, u_model, optimizer


def train_ddpm(conf_stage1, conf, path):
    gaussian_diffusion, u_model, optimizer = load_second_stage_model(conf)
    autoencoder, _ = load_first_stage(conf_stage1, path, False)
    batch_size = conf.batch_size
    timesteps = conf.timesteps
    device = conf.device
    os.makedirs(conf.zpath, exist_ok=True)
    pkl_file = conf.encodedpath
    with open(pkl_file, 'rb') as f:
        encoded_dataset = pickle.load(f)

    data_tensor = encoded_dataset.tensors[0]
    label_tensor = encoded_dataset.tensors[1]

    # Calculate the mean and variance for the first batch
    batch_mean = torch.mean(data_tensor)
    batch_var = torch.var(data_tensor)

    # Calculate the scale factor
    scale_factor = 1 / torch.sqrt(batch_var)

    # Rescale the entire dataset
    rescaled_train_images = (data_tensor - batch_mean) * scale_factor

    new_dataset = TensorDataset(rescaled_train_images, label_tensor)
    train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    epochs = conf.epoch
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        for step, (images, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            batch_size = images.shape[0]
            images = images.to(device)
            # print(images.size())
            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = gaussian_diffusion.train_losses(u_model, images, t)

            loss.backward()
            optimizer.step()
        print("Loss:", loss.item())
    # todo save diffusion model

    for i in range(200):
        print(f"epoch {i}")
        generated_images = gaussian_diffusion.sample(u_model, conf.resolution, batch_size=30,
                                                     channels=conf.unetconfig.out_channels)
        # generate new images
        # imgs = generated_images[-1].squeeze()
        # imgs = torch.from_numpy(imgs)
        scale_factor = scale_factor.to("cpu")
        batch_mean = batch_mean.to("cpu")
        generated_images_last_layer = generated_images[-1]
        generated_images_last_layer = torch.from_numpy(generated_images_last_layer)
        imgs = (generated_images_last_layer / scale_factor) + batch_mean

        # imgs = imgs.to(device)

        with open(conf.zpath + f"/{i}.pkl", 'wb') as f:
            pickle.dump(imgs, f)

        # images = autoencoder._module.decode(imgs)
        # os.makedirs(conf.output, exist_ok=True)
        # for j, image in enumerate(images):
        #     save_image(image,  conf.output + f'/{i * 600 + j}_{read_label}.png')


