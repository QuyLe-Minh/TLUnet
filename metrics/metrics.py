import torch
import numpy as np

def one_hot_encoder(pred):
    mask = np.zeros(pred.shape)
    one_hot = pred.argmax(1).cpu().detach()
    mask[:, 0, :, :] = np.where(one_hot == 0, 1, 0)
    mask[:, 1, :, :] = np.where(one_hot == 1, 1, 0)
    mask[:, 2, :, :] = np.where(one_hot == 2, 1, 0)
    return torch.tensor(mask)

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
    
    