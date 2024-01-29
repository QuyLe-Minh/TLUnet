import torch
from utils import one_hot_encoder, manual_crop, concat
from math import *

def val(config, dataloader, model, entropy_loss, dice_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(config.device)
            y = y.to(config.device)
            
            X_cropped_collection = manual_crop(X)
            y_cropped_collection = []
            for i in range(len(X_cropped_collection)):
                y_cropped = model(X_cropped_collection[i])[0]
                y_cropped_collection.append(y_cropped)
            
            pred = concat(y_cropped_collection)
            
            y_one_hot = one_hot_encoder(y)
            test_loss += entropy_loss(pred, y_one_hot).item() + dice_loss(pred, y_one_hot)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= (size * 192 * 192 * 64)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss