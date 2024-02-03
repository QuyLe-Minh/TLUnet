import torch
from utils import manual_crop, concat
from math import *
from metrics.metrics import *

def val(config, dataloader, model, entropy_loss, dice_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    dice_score_liver, iou_score_liver = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(torch.float).to(config.device)
            
            X_cropped_collection = manual_crop(X)
            y_cropped_collection = []
            for i in range(len(X_cropped_collection)):
                y_cropped = model(X_cropped_collection[i])[0]
                y_cropped_collection.append(y_cropped.detach().cpu())
            
            pred = concat(y, y_cropped_collection)
            test_loss += 0.4*entropy_loss(pred, y).item() + 0.6 * dice_loss(pred, y)
            correct += (pred.to(torch.uint8) == y.to(torch.uint8)).type(torch.float).mean().item()

            dice_score_liver += dice(pred, y)
            iou_score_liver += iou(pred, y)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    dice_score_liver /= size
    iou_score_liver /= size
    print(f"Dice score liver: {(100 * dice_score_liver):>0.3f}%, IoU score liver: {(iou_score_liver * 100):>0.3f}% \n")
    return test_loss