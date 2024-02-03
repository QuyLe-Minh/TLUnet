from monai.losses import DiceLoss
from val import *
from architecture.CNN3D import CNN3D
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

def train(config, dataloader, model, entropy_loss, dice_loss, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(config.device)
        y = y.to(config.device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(X) #output, ds1, ds2, ds3 (least to most)
        loss = 0.4 * (entropy_loss(pred[0], y) * 0.53 + entropy_loss(pred[1], y) * 0.07 + entropy_loss(pred[2], y) * 0.13 + entropy_loss(pred[3], y) * 0.27) + 0.6 * dice_loss(pred[0], y)
        loss.backward()
        optimizer.step()

        correct += (pred[0].to(torch.uint8) == y).type(torch.float).mean().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct = correct / len(dataloader)
    print(f"Accuracy: {100 * correct:>7f}%")
    
    
def training(config, train_loader, val_loader, mode):
    model = CNN3D().to(config.device)
    if mode != "training":
        model.load_state_dict(torch.load("cnn3d.pt"))
        print("Load model cnn3d...")
    model.train()
    
    entropy_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    torch.cuda.empty_cache()
    best_one = 10
    count = 0
    
    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(config, train_loader, model, entropy_loss, dice_loss, optimizer)
        val_loss = val(config, val_loader, model, entropy_loss, dice_loss)

        if val_loss < best_one:
            count = 0
            best_one = val_loss
            torch.save(model.state_dict(), "cnn3d.pt")
        else:
            model.load_state_dict(torch.load("cnn3d.pt"))
            count+=1
            if count == config.patience:
                break

        scheduler.step(val_loss)