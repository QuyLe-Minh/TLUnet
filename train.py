import torch.nn.functional as F
from loss.dice_loss import *

def train(config, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(config.device)
        y = y.to(config.device)
        # zero the parameter gradients
        optimizer.zero_grad()

        down1 = F.interpolate(y, 256)
        down2 = F.interpolate(y, 128)

        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred[2], y) * 0.57 + loss_fn(pred[1], down1) * 0.29 + loss_fn(pred[0], down2) * 0.14
        loss.backward()
        optimizer.step()

        correct += (pred[2].argmax(1) == y.argmax(1)).float().sum()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct = correct / (size * 512 * 512)
    print(f"Accuracy: {correct:>7f}")