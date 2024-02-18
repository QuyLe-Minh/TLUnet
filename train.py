from monai.losses import DiceLoss
from architecture.TLUnet import TLUnet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch
from utils import manual_crop, concat
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
                y_cropped = model(X_cropped_collection[i])
                y_cropped_collection.append(y_cropped.detach().cpu())
            
            pred = concat(y, y_cropped_collection)
            
            test_loss += dice_loss(pred, y)
            correct += (torch.round(pred).to(torch.uint8) == torch.round(y).to(torch.uint8)).type(torch.float).mean().item()

            dice_score_liver += dice(pred, y)
            iou_score_liver += iou(pred, y)

    test_loss /= num_batches
    correct /= size
    dice_score_liver /= size
    iou_score_liver /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Dice score liver: {(100 * dice_score_liver):>0.3f}%, IoU score liver: {(iou_score_liver * 100):>0.3f}% \n")
    return test_loss

def train(config, dataloader, model, entropy_loss, dice_loss, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(config.device)
        y = y.to(torch.float).to(config.device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(X) #output, ds1, ds2, ds3 (least to most)
        
        loss = entropy_loss(pred, y) * 0.4 + 0.6 * dice_loss(pred, y)
        loss.backward()
        optimizer.step()

        correct += (torch.round(pred).to(torch.uint8) == torch.round(y).to(torch.uint8)).type(torch.float).mean().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct = correct / len(dataloader)
    print(f"Accuracy: {100 * correct:>7f}%")
    
def applyWeight(config, model, model_weight):
    weight = torch.load(model_weight)
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= config.n_freeze: break
        
        param.data = weight[name]
        param.requires_grad = False
    
    
def training(config, train_loader, val_loader, mode):
    model = TLUnet().to(config.device)
    if mode != "training":
        path = "tlu.pt" 
        model.load_state_dict(torch.load(path))
        print(f"Load model {path}...")
        
    applyWeight(config, model, config.model_weight)
    model.train()
    
    # entropy_loss = nn.CrossEntropyLoss()
    entropy_loss = nn.BCELoss()
    dice_loss = DiceLoss(squared_pred = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) #3e-5
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    torch.cuda.empty_cache()
    best_one = 100
    
    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(config, train_loader, model, entropy_loss, dice_loss, optimizer)
        val_loss = val(config, val_loader, model, entropy_loss, dice_loss)

        if val_loss < best_one:
            best_one = val_loss
            torch.save(model.state_dict(), "cnn3d_3.pt")
        torch.save(model.state_dict(), "cnn3d.pt")
        # else:
        #     # model.load_state_dict(torch.load("cnn3d_2.pt"))
        #     count+=1
        #     if count == config.patience:
        #         break

        # scheduler.step(val_loss)