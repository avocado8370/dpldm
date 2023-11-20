import torch


def log_p_var_normal(samples, var):
    log_p = - 0.5 * torch.square(samples) / var - 0.5 * np.log(var) - 0.9189385332  # 0.5 * np.log(2 * np.pi)
    return log_p

@torch.jit.script
def log_p_standard_normal(samples):
    log_p = - 0.5 * torch.square(samples) - 0.9189385332  # 0.5 * np.log(2 * np.pi)
    return log_p