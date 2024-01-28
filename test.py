import torch
from torch import nn
input = torch.randn(1, 16, 64, 64)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=3, padding=2)
out = upsample(input, output_size = (1, 16, 128, 128))
print(out.size())
