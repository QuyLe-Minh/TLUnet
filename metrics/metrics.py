import torch
import numpy as np
from utils import one_hot_encoder

def iou(pred, gt):
    """Calculate iou score

    Args:
        pred (_Tensor_): cuda:0
        gt (_Tensor_): cuda:0
    """
    mask = one_hot_encoder(pred).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    eps = 1e-5
    intersect = torch.sum(mask * gt)
    union = torch.sum(pred + gt)
    iou = (intersect + eps)/(union - intersect + eps)
    return iou

def dice(pred, gt):
    """Calculate dice score

    Args:
        pred (_Tensor_): cuda:0
        gt (_Tensor_): cuda:0
    """
    mask = one_hot_encoder(pred).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    eps = 1e-5
    intersect = torch.sum(mask * gt)
    union = torch.sum(pred + gt)
    dice = 2 * (intersect + eps)/(union + eps)
    return dice

def acc(pred, gt):
    """Calculate accuracy

    Args:
        pred (_Tensor_): cuda:0
        gt (_Tensor_): cuda:0
    """
    mask = one_hot_encoder(pred).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return torch.mean(mask == gt)            
    
    