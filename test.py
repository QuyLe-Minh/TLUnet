import torch
from torch import nn
import torch.nn.functional as f
from architecture.CNN3D import CNN3D
from architecture.TLUnet import TLUnet

def init_weights(m):
  if isinstance(m, nn.Conv3d):
    torch.nn.init.kaiming_normal(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)

model = TLUnet(n_classes=2)
# model.apply(init_weights)
n = 0
for param in model.named_parameters():
  if n < 37:
    print(param, n)
    n+=1
  else: break
