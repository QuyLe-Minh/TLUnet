import torch.nn.functional as F
from loss.dice_loss import *
from utils import one_hot_encoder, init_weights
from val import *
from architecture.TLUnet import TLUnet
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(config, dataloader, model, entropy_loss, dice_loss, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(config.device)
        y = y.to(config.device)
        
        y_one_hot = one_hot_encoder(y)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        b, c, h, w, d = X.shape
        down3 = F.interpolate(y_one_hot, (h//2, w//2, d//2))
        down2 = F.interpolate(y_one_hot, (h//4, w//4, d//4))
        down1 = F.interpolate(y_one_hot, (h//8, w//8, d//8))

        # forward + backward + optimize
        pred = model(X) #output, ds1 (min), ds2, ds3
        loss = entropy_loss(pred[0], y_one_hot) * 0.53 + entropy_loss(pred[1], down1) * 0.07 + entropy_loss(pred[2], down2) * 0.13 + entropy_loss(pred[3], down3) * 0.27 + dice_loss(pred[2], y_one_hot)
        loss.backward()
        optimizer.step()

        correct += (pred[2].argmax(1) == y).float().sum()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct = correct / (size * 192 * 192 * 64)
    print(f"Accuracy: {correct:>7f}")
    
    
def training(config, train_loader, val_loader, mode):
    model = TLUnet(config.n_classes).to(config.device)
    if mode == "training":
        model.apply(init_weights)
    else:
        model.load_state_dict(torch.load("model.pt"))
    model.train()
    
    entropy_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(config.device))
    dice_loss = Dice_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
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
            torch.save(model.state_dict(), "model.pt")
        else:
            model.load_state_dict(torch.load("model.pt"))
            count+=1
            if count == config.patience:
                break

        scheduler.step(val_loss)
    