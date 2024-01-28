import torch
from torch import nn
import numpy as np

class Dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def one_hot_encode(self, pred):
        mask = np.zeros(pred.shape)
        one_hot = pred.argmax(1).cpu().detach()
        mask[:, 0, :, :] = np.where(one_hot == 0, 1, 0)
        mask[:, 1, :, :] = np.where(one_hot == 1, 1, 0)
        mask[:, 2, :, :] = np.where(one_hot == 2, 1, 0)
        return torch.tensor(mask)
    
    def forward(self, gt, pred):
        """
        pred is unnormalized logits of shape (N, num_classes, H, W, D)
        """
        mask = self.one_hot_encode(pred)
        mask = mask.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        intersect = torch.sum(gt * mask)
        union = torch.sum(gt + mask)
        eps = 1e-5
        dice = 2 * (intersect + eps)/(union + eps)
        return 1 - dice 
        
        