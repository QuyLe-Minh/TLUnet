import torch
from torch import nn
import numpy as np

class Dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def one_hot_encoder(self, pred):
        mask = torch.zeros(pred.shape).to("cuda")
        one_hot = pred.argmax(1)
        mask[:, 0, :, :, :] = torch.where(one_hot==0, 1, 0)
        mask[:, 1, :, :, :] = torch.where(one_hot==1, 1, 0)
        
        return mask
    
    def forward(self, pred, gt):
        """
        pred is unnormalized logits of shape (N, num_classes, H, W, D)
        """
        mask = self.one_hot_encoder(pred)
        intersect = torch.sum(gt * mask)
        union = torch.sum(gt + mask)
        eps = 1e-5
        dice = 2 * (intersect + eps)/(union + eps)
        return 1 - dice 
        
        