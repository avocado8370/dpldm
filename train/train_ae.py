import csv
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.util import instantiate_from_config
from train.encode import load_first_stage_model


def train_ae(conf, output_dir):
    autoencoder, aeopt, trainloader, evalloader, privacy_engine = load_first_stage_model(conf)
    loss = instantiate_from_config(conf.lossconfig)
    loss = loss.to(conf.device)

    delta = conf.delta
    target_epsilon = conf.target_epsilon
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f'{output_dir}/input'):
        os.makedirs(f'{output_dir}/input')
    if not os.path.exists(f'{output_dir}/output'):
        os.makedirs(f'{output_dir}/output')

    csv_data = [['Loss', 'ε', 'δ']]
    global_steps = 0
    for epoch in range(conf.num_epochs):
        autoencoder.train()
        losses = []
        for i, data in enumerate(tqdm(trainloader)):
            inputs, _ = data
            inputs = inputs.to(conf.device)
            aeopt.zero_grad()
            reconstructions, posterior = autoencoder(inputs)
            aeloss, log_dict_ae = loss(inputs, reconstructions, posterior, 0, global_steps,
                                       last_layer=autoencoder._module.get_last_layer(), split="train")
            aeloss.backward()
            aeopt.step()
            losses.append(aeloss.item())
            global_steps = global_steps + 1
            if global_steps % 1000 == 0:
                print("aeloss: {}".format(aeloss))
                print("global_step: " + str(global_steps))

        # autoencoder.eval()
        with torch.no_grad():
            autoencoder.eval()
            # save generated images
            inputs, _ = next(iter(evalloader))
            inputs = inputs.to(conf.device)
            reconstructions, posterior = autoencoder(inputs)
            save_image(inputs, f'{output_dir}/input/input_{epoch}.png')
            save_image(reconstructions, f'{output_dir}/output/output_{epoch}.png')
        if target_epsilon > 0:
            epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"(ε = {epsilon:.2f}, δ = {delta})"
            )
        else:
            epsilon = '∞'
            delta = 0
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
            )
        tempCsv = [np.mean(losses), epsilon, delta]
        csv_data.append(tempCsv)

    if not os.path.exists(f'{output_dir}/autoencoder'):
        os.makedirs(f'{output_dir}/autoencoder')
    torch.save(autoencoder.state_dict(), f'{output_dir}/autoencoder/checkpoint.pth')
    with open(f'{output_dir}/output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
