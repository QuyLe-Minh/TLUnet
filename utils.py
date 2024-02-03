import torch
from math import *
import os

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 5
    epochs = 1000
    n_classes = 2
    patience = 3
    n_freeze = 36
    train = [f"dataset/train/{file}" for file in os.listdir("dataset/train")]
    val = [f"dataset/val/{file}" for file in os.listdir("dataset/val")]
    mode = "training"

def one_hot_encoder(input, n_classes=2):
    """One hot encode

    Args:
        input (torch tensor): B, C = 1, H, W, D
    
    Returns:
        one hot: cuda
    """
    tmp = input.squeeze(1)
    b, c, h, w, d = input.shape
    one_hot = torch.zeros((b, n_classes, h, w, d))
    one_hot[:, 0, :, :, :] = torch.where(tmp == 0, 1, 0)
    one_hot[:, 1, :, :, :] = torch.where(tmp == 1, 1, 0)
    
    return one_hot.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      
      
def manual_crop(X):
    X_cropped_collection = []
    _, _, h, w, d = X.shape
    for i in range(ceil(h/192)):
        if (i+1)*192 > h:
            start_i = h-192
        else:
            start_i = i*192
        for j in range(ceil(w/192)):
            if (j+1)*192 > w:
                start_j = w - 192
            else:
                start_j = j * 192
            for k in range(ceil(d/64)):
                if (k+1)*64 > d:
                    start_k = d - 64
                else:
                    start_k = k*64
                X_cropped = X[:, :, start_i : start_i+192, start_j : start_j+192, start_k : start_k+64]
                X_cropped_collection.append(X_cropped.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
                
    return X_cropped_collection

def concat(y, y_cropped_collection):
    """Concat pieces of cube

    Args:
        y (ground truth): b=1, c, 512, 512, d
        y_cropped_collection (pred): collection of cubes: n, b_, c_, 192, 192, 64
    """
    b, c, h, w, d = y.shape
    pred = torch.empty(y.shape)
    sample = 0
    for i in range(ceil(h/192)):
        if (i+1)*192 > h:
            start_i = h-192
        else:
            start_i = i*192
        for j in range(ceil(w/192)):
            if (j+1)*192 > w:
                start_j = w - 192
            else:
                start_j = j * 192
            for k in range(ceil(d/64)):
                if (k+1)*64 > d:
                    start_k = d - 64
                else:
                    start_k = k*64
                pred[:, :, start_i:start_i+192, start_j:start_j+192, start_k:start_k+64] = y_cropped_collection[sample]
                sample += 1
                
    return pred.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))