import torch
from torch import nn
import numpy as np
from utils import one_hot_encoder

class Dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, gt, pred):
        """
        pred is unnormalized logits of shape (N, num_classes, H, W, D)
        """
        mask = one_hot_encoder(pred)
        intersect = torch.sum(gt * mask)
        union = torch.sum(gt + mask)
        eps = 1e-5
        dice = 2 * (intersect + eps)/(union + eps)
        return 1 - dice 
        
        