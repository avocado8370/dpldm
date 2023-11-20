import os
import pickle
import time

import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torchvision import transforms, datasets
from tqdm import tqdm

from models.autoencoder import AutoencoderKL
from models.util import instantiate_from_config



def load_first_stage_model(conf, poisson=True):
    MNIST_MEAN = conf.MNIST_MEAN
    MNIST_STD = conf.MNIST_STD
    dataset_name = conf.dataset_name
    batch_size = conf.batch_size
    target_epsilon = conf.target_epsilon

    num_workers = conf.num_workers

    if dataset_name == 'FashionMNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
        trainset = torchvision.datasets.FashionMNIST(root='../data', train=True,
                                                     download=True, transform=transform)
        evalset = torchvision.datasets.FashionMNIST(root='../data', train=False,
                                                    download=True, transform=transform)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                              download=True, transform=transform)
        evalset = torchvision.datasets.MNIST(root='../data', train=False,
                                             download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)
        evalset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=True, transform=transform)
    elif dataset_name == 'celebA':
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CelebA(root='../data', train=True,
                                               download=True, transform=transform)
        evalset = torchvision.datasets.CelebA(root='../data', train=False,
                                              download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=num_workers, pin_memory=True)
    # evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size,
    #                                          shuffle=True, num_workers=num_workers, pin_memory=True)
    combined_dataset = ConcatDataset([trainset, evalset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True)

    num_epochs = conf.num_epochs
    delta = conf.delta
    c = conf.c
    lr = conf.lr
    privacy_engine = PrivacyEngine()
    autoencoder = AutoencoderKL(conf.ddconfig, conf.embed_dim, conf.lr)
    autoencoder.to(conf.device)
    if target_epsilon > 0:
        autoencoder = ModuleValidator.fix(autoencoder)
        aeopt = torch.optim.SGD(autoencoder.parameters(), lr=lr)
        autoencoder, aeopt, combined_loader = privacy_engine.make_private_with_epsilon(
            module=autoencoder,
            optimizer=aeopt,
            data_loader=combined_loader,
            # noise_multiplier=sigma,
            epochs=num_epochs,
            target_delta=delta,
            target_epsilon=target_epsilon,
            max_grad_norm=c,
            poisson_sampling=poisson
        )
    else:
        aeopt = torch.optim.Adam(autoencoder.parameters(), lr=lr, betas=(0.5, 0.9))
    return autoencoder, aeopt, combined_loader, combined_loader, privacy_engine


def load_first_stage(conf, path, poisson=True):
    autoencoder, _, train_loader, _, _ = load_first_stage_model(conf, poisson)
    state_dict = torch.load(path)
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if k.startswith('_module.'):
    #         name = k[8:]  # remove the '_module.' prefix
    #     else:
    #         name = k
    #     new_state_dict[name] = v
    autoencoder.load_state_dict(state_dict)
    return autoencoder, train_loader


def encode(conf, path, target_path="", labeled=False):
    autoencoder, train_loader = load_first_stage(conf, path, False)
    autoencoder.eval()
    encoded_dataset = []
    is_dp = conf.target_epsilon > 0
    os.makedirs(target_path, exist_ok=True)
    target_path = target_path + '/encoded_dataset.pkl'
    model = autoencoder if conf.target_epsilon == 0 else autoencoder._module
    with torch.no_grad():
        if labeled:
            for iteration in range(10):
                print(f'current label {iteration}')
                if os.path.exists(target_path):
                    with open(target_path, 'rb') as f:
                        encoded_dataset = pickle.load(f)
                else:
                    for image, label in tqdm(train_loader):
                        if label == iteration:
                            image = image.to(conf.device)
                            if is_dp:
                                posterior = model.encode(image)
                            else:
                                posterior = autoencoder.encode(image)
                            encoded_image = posterior.sample()
                            encoded_dataset.append((encoded_image.squeeze(), label))
                    encoded_images, labels = zip(*encoded_dataset)
                    encoded_images = torch.stack(encoded_images)  # 将图像张量堆叠为一个张量
                    labels = torch.tensor(labels)
                    encoded_dataset = torch.utils.data.TensorDataset(encoded_images, labels)
                with open(target_path, 'wb') as f:
                    pickle.dump(encoded_dataset, f)
        else:

            if os.path.exists(target_path):
                with open(target_path, 'rb') as f:
                    encoded_dataset = pickle.load(f)
            else:
                for image, label in tqdm(train_loader):
                    image = image.to(conf.device)
                    if is_dp:
                        posterior = model.encode(image)
                    else:
                        posterior = autoencoder.encode(image)
                    encoded_image = posterior.sample()
                    encoded_dataset.append((encoded_image.squeeze(), label))
                encoded_images, labels = zip(*encoded_dataset)
                # encoded_images = torch.stack(encoded_images)  # 将图像张量堆叠为一个张量
                encoded_images = torch.cat(encoded_images, dim=0)
                # print(labels.shape)
                # labels = torch.tensor(labels)
                # 假设 labels 是一个包含多个一维张量的列表
                labels = torch.cat(labels, dim=0)
                encoded_dataset = torch.utils.data.TensorDataset(encoded_images, labels)
            with open(target_path, 'wb') as f:
                pickle.dump(encoded_dataset, f)
        return encoded_dataset
