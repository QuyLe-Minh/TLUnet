from architecture.components import *

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
        x = self.deconv(x)
        skip = self.skip_conv(skip)
        up = torch.cat([x, skip], dim = 1)
        return up

        
class TLUnet(nn.Module):
    def __init__(self, n_classes = 1):
        super().__init__()
        filters = 32
        n_deconv = 96
        self.eblock1 = EncoderBlock(1, filters)
        self.eblock2 = EncoderBlock(filters, filters * 2)
        self.eblock3 = EncoderBlock(filters * 2, filters * 4)
        self.eblock4 = EncoderBlock(filters * 4, filters * 8, is_last=True)
        
        self.deconv0 = nn.ConvTranspose3d(filters * 8, n_deconv, 2, 2)
        self.skip_conv = ConvBlock(5, filters * 4, filters * 4 - n_deconv)
        
        self.deconv1 = DeconvBlock(filters * 4, n_deconv//2)
        self.deconv2 = DeconvBlock(filters * 2, n_deconv//4)
        self.deconv3 = DeconvBlock(filters, n_deconv//8)
        
        self.head = nn.Sequential(
            ConvBlock(3, filters//2, filters//2),
            ConvBlock(3, filters//2, filters//2),
            nn.Conv3d(filters//2, n_classes, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        x1, skip1 = self.eblock1(input) #96
        # print(x1.shape, skip1.shape)
        x2, skip2 = self.eblock2(x1) #48
        # print(x2.shape, skip2.shape)
        x3, skip3 = self.eblock3(x2) #24
        # print(x3.shape, skip3.shape)
        x4, skip4 = self.eblock4(x3) #12
        
        x4 = self.deconv0(x4) #24
        skip4 = self.skip_conv(skip4)
        # print(x4.shape, skip4.shape)
        
        concat = torch.cat([x4, skip4], dim = 1) 
        # print(concat.shape)
        
        d1 = self.deconv1(concat, skip3) 
        d2 = self.deconv2(d1, skip2) 
        d3 = self.deconv3(d2, skip1)
        
        pred = self.head(d3)
        
        return pred