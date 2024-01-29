import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, kernel, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, 1, padding="same"),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        output = self.conv(x)
        return output

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(3, in_channels, in_channels)
        self.conv2 = ConvBlock(3, in_channels, in_channels)
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, 2, stride = 2)
        
    def forward(self, expansive, contractive):
        fusion = torch.cat([expansive, contractive], dim = 1)
        up = self.conv1(fusion)
        up = self.conv2(up)
        up = self.deconv(up)
        return up
    
class EncoderBlock(nn.Module):
    def __init__(self, skip_channels, in_channels, out_channels, is_last = False):
        self.conv1 = ConvBlock(3, in_channels, out_channels//2)
        self.pool = nn.MaxPool3d(2, stride = 2)
        self.conv2 = ConvBlock(3, out_channels//2, out_channels)
        self.skip_conv = ConvBlock(5, out_channels//2, skip_channels)
        self.is_last = is_last
    
    def forward(self, x):
        x = self.conv1(x)
        skip = self.skip_conv(x)
        x = self.pool(x)
        if not self.is_last:
            x = self.conv2(x)
        
        return x, skip
        
class TLUnet(nn.Module):
    def __init__(self, n_classes = 2):
        super().__init__()
        filters = 32
        n_deconv = 96
        self.eblock1 = EncoderBlock(4, 1, filters)
        self.eblock2 = EncoderBlock(8, filters, filters * 2)
        self.eblock3 = EncoderBlock(16, filters * 2, filters * 4)
        self.eblock4 = EncoderBlock(32, filters * 4, filters * 4, is_last=True)
        
        self.bottle_neck = nn.Sequential(
            ConvBlock(3, filters * 4, filters * 8),
            ConvBlock(3, filters * 8, filters * 8),
            nn.ConvTranspose3d(filters * 8, n_deconv, 2, 2)
        )
        
        self.deconv1 = DeconvBlock(filters * 4, n_deconv//2)
        self.deconv2 = DeconvBlock(filters * 2, n_deconv//4)
        self.deconv3 = DeconvBlock(filters, n_deconv//8)
        
        self.head = nn.Sequential(
            ConvBlock(3, filters//2, filters//2),
            ConvBlock(3, filters//2, filters//2),
            nn.Conv3d(filters//2, n_classes, 1)
        )
        
        self.ds1 = nn.Conv3d(n_deconv//2, n_classes, 1)
        self.ds2 = nn.Conv3d(n_deconv//4, n_classes, 1)
        self.ds3 = nn.Conv3d(n_deconv//8, n_classes, 1)  
    def forward(self, input):
        x1, skip1 = self.eblock1(input)
        x2, skip2 = self.eblock2(x1)
        x3, skip3 = self.eblock3(x2)
        x4, skip4 = self.eblock4(x3)
        
        x4 = self.bottle_neck(x4)
        
        d1 = self.deconv1(x4, skip4)
        ds1 = self.ds1(d1)
        d2 = self.deconv2(d1, skip3)
        ds2 = self.ds2(d2)
        d3 = self.deconv3(d2, skip2)
        ds3 = self.ds3(d3)
        
        d4 = torch.cat((d3, skip1), dim = 1)
        res = self.head(d4)
        
        return res, ds1, ds2, ds3