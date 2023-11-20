import importlib
import math

import torch
from torch import nn


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_input_size(dataset):
    if dataset in {'mnist', 'omniglot'}:
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    else:
        raise NotImplementedError


def init_temb_fun(embedding_type, embedding_scale, embedding_dim):
    if embedding_type == 'positional':
        temb_fun = PositionalEmbedding(embedding_dim, embedding_scale)
    elif embedding_type == 'fourier':
        temb_fun = RandomFourierEmbedding(embedding_dim, embedding_scale)
    else:
        raise NotImplementedError

    return temb_fun


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.scale
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class RandomFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(RandomFourierEmbedding, self).__init__()
        self.w = nn.Parameter(torch.randn(size=(1, embedding_dim // 2)) * scale, requires_grad=False)

    def forward(self, timesteps):
        emb = torch.mm(timesteps[:, None], self.w * 2 * 3.14159265359)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


def mask_inactive_variables(x, is_active):
    x = x * is_active
    return x

def sample_gaussian_like(y):
    return torch.randn_like(y, device='cuda')

def get_mixed_prediction(mixed_prediction, param, mixing_logit, mixing_component=None):
    if mixed_prediction:
        assert mixing_component is not None, 'Provide mixing component when mixed_prediction is enabled.'
        coeff = torch.sigmoid(mixing_logit)
        param = (1 - coeff) * mixing_component + coeff * param

    return param

def trace_df_dx_hutchinson(f, x, noise, no_autograd):
    """
    Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
    """
    if no_autograd:
        # the following is compatible with checkpointing
        torch.sum(f * noise).backward()
        # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
        jvp = x.grad
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        x.grad = None
    else:
        jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

    return trJ