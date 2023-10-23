import argparse
import os
import pickle
import time

import numpy as np
import torch
from omegaconf import OmegaConf
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
    batch_size = 256
    timesteps = conf.timesteps
    device = conf.device
    for read_label in range(conf.epoch):
        pkl_file = conf.encodedpath
        with open(pkl_file, 'rb') as f:
            encoded_dataset = pickle.load(f)
        train_loader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        epochs = 10
        for epoch in range(epochs):
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

        for i in range(10):
            print(f"epoch {i}")
            generated_images = gaussian_diffusion.sample(u_model, conf.resolution, batch_size=600, channels=conf.unetconfig.out_channels)
            # generate new images
            generated_tensor = torch.from_numpy(np.array(generated_images))
            # print(generated_tensor.size())
            imgs = generated_images[-1].squeeze()
            imgs = torch.from_numpy(imgs)
            imgs = imgs.to(device)

            images = autoencoder._module.decode(imgs)
            for j, image in enumerate(images):
                save_image(image,  conf.output + f'/{i * 600 + j}_{read_label}.png')


