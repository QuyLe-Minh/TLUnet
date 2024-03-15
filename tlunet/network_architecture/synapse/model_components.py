from torch import nn
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from tlunet.network_architecture.layers import LayerNorm
from tlunet.network_architecture.synapse.transformerblock import TransformerBlock
from tlunet.network_architecture.dynunet_block import get_conv_layer, UnetResBlock
import torch

einops, _ = optional_import("einops")

"""
TLUNET
"""
class ConvBlock(nn.Module):
    def __init__(self, kernel, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, 1, padding="same"),
            nn.BatchNorm3d(out_channels),
            nn.ELU()
        )
    
    def forward(self, x):
        output = self.conv(x)
        return output
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last = False):
        super().__init__()
        if is_last:
            self.conv1 = ConvBlock(3, in_channels, out_channels//2)
            self.conv2 = nn.Sequential(
                ConvBlock(3, out_channels//2, out_channels),
                ConvBlock(3, out_channels, out_channels)
            )
        else:
            self.conv1 = ConvBlock(3, in_channels, out_channels//2)
            self.conv2 = ConvBlock(3, out_channels//2, out_channels)
        self.pool = nn.MaxPool3d(2, stride = 2)
    
    def forward(self, x):
        x = self.conv1(x)
        skip = x
        x = self.pool(x)
        x = self.conv2(x)
        
        return x, skip
    
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip_conv = ConvBlock(5, in_channels//2, in_channels//2 - out_channels)
        self.conv1 = ConvBlock(3, in_channels, in_channels)
        self.conv2 = ConvBlock(3, in_channels, in_channels)
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, 2, stride = 2)
        
    def forward(self, x, skip):
        x = self.conv1(x)
        x = self.conv2(x)
        deconv = self.deconv(x)
        skip = self.skip_conv(skip)
        up = torch.cat([deconv, skip], dim = 1)
        return up, x
