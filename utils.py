import numpy as np
import torch
from torch import nn

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    epochs = 2
    n_classes = 2
    patience = 5
    train = torch.load("dataset/dataset.pth")
    val = torch.load("dataset/val.pth")
    mode = "training"

def one_hot_encoder(input, n_classes):
    """One hot encode

    Args:
        input (torch tensor): B, C = 1, H, W, D
    
    Returns:
        one hot: cuda
    """
    b, c, h, w, d = input.shape
    one_hot = torch.zeros((b, n_classes, h, w, d))
    one_hot[:, 0, :, :, :] = torch.where(input == 0, 1, 0)
    one_hot[:, 1, :, :, :] = torch.where(input == 1, 1, 0)
    
    return one_hot.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def init_weights(m):
  if isinstance(m, nn.Conv3d):
    torch.nn.init.kaiming_normal(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)