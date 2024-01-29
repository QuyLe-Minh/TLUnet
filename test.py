import torch
from torch import nn
import torch.nn.functional as f

input = torch.randn(1, 16, 192, 192, 64)
# upsample = nn.ConvTranspose3d(16, 16, 2, stride=2)
# out = upsample(input, output_size = (1, 16, 192, 192, 64))
out = f.interpolate(input, (96, 96, 32))
print(out.size())
