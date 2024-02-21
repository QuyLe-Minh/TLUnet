import torch
from utils import one_hot_encoder

def iou(pred, gt):
    """Calculate iou score

    Args:
        pred (_Tensor_): normalized B, C, H, W, D
        gt (_Tensor_): one hot vector B, C, H, W, D
    """
    eps = 1e-5
    intersect = torch.sum(pred * gt)
    union = torch.sum(pred + gt)
    iou = (intersect + eps)/(union - intersect + eps)
    return iou

def dice(pred, gt):
    """Calculate dice score

    Args:
        pred (_Tensor_): cuda:0
        gt (_Tensor_): cuda:0
    """
    eps = 1e-5
    intersect = torch.sum(pred * gt)
    union = torch.sum(pred + gt)
    dice = 2 * (intersect + eps)/(union + eps)
    return dice

def acc(pred, gt):
    """Calculate accuracy

    Args:
        pred (_Tensor_): cuda:0
        gt (_Tensor_): cuda:0
    """
    mask = one_hot_encoder(pred)
    return torch.mean(mask == gt)  

def dice_global(pred, gt):
    pass

def dice_per_case(pred, gt):
    pass

def VOE(pred, gt):
    # IOU loss

    return (1 - iou(pred, gt)) * 100

def HD95(pred, gt):
    pass


    
    