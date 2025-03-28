import torch

def gaussian_kernel_1d(mean, sigma, size):
    
    weight = torch.arange(size)

    return torch.exp(-0.5 * (weight - mean) ** 2 / sigma ** 2)