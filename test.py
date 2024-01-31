import torch
from torch import nn
import torch.nn.functional as f
from architecture.CNN3D import CNN3D

model = CNN3D(n_classes=2)
n = 1
for name, param in model.named_parameters():
    print(name, n)
    n+=1
