import numpy as np
import torch
from torch import nn

def one_hot_encoder(input, n_classes):
    """One hot encode

    Args:
        input (torch tensor): B, C = 1, H, W, D
    """
    input = input.numpy()
    b, c, h, w, d = input.shape
    one_hot = np.zeros((b, n_classes, h, w, d))
    one_hot[:, 0, :, :, :] = np.where(input == 0, 1, 0)
    one_hot[:, 1, :, :, :] = np.where(input == 1, 1, 0)
    
    return torch.tensor(one_hot)

def init_weights(m):
  if isinstance(m, nn.Conv3d):
    torch.nn.init.kaiming_normal(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)