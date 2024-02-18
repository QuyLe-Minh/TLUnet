import torch
from math import *
import os

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 7
    epochs = 1000
    n_classes = 2
    patience = 10
    n_freeze = 36
    train = [f"dataset/train_tlu/{file}" for file in os.listdir("dataset/train_tlu")]
    val = [f"dataset/val_tlu/{file}" for file in os.listdir("dataset/val_tlu")]
    model_weight = "cnn3d_3.pt"
    mode = "tuning"

config = Config()

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
                X_cropped = X_cropped = X[0, :, start_i : start_i+192, start_j : start_j+192, start_k : start_k+64]
                X_cropped_collection.append(X_cropped)
                
    n_cubes = len(X_cropped_collection)
    batches = []
    for i in range(ceil(n_cubes/config.batch_size)):
        if (i+1)*config.batch_size > n_cubes:
            batch = torch.stack(X_cropped_collection[i*config.batch_size:], dim = 0).cuda()
            batches.append(batch)
        else:
            batch = torch.stack(X_cropped_collection[i*config.batch_size:(i+1)*config.batch_size], dim = 0).cuda()
            batches.append(batch)
                
    return batches

def concat(y, y_cropped_collection):
    """Concat pieces of cube

    Args:
        y (ground truth): b=1, c = 1, 512, 512, d
        y_cropped_collection (pred): list of batches: n, b_, c_, 192, 192, 64
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
                pred[:, :, start_i:start_i+192, start_j:start_j+192, start_k:start_k+64] = y_cropped_collection[sample//config.batch_size][sample%config.batch_size]
                sample += 1
                
    return pred.cuda()
