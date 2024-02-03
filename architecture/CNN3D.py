from architecture.components import *

class CNN3D(nn.Module):
    def __init__(self, n_classes = 1):
        super().__init__()
        filters = 32
        deconv = 16
        
        self.eblock1 = EncoderBlock(1, filters)
        self.eblock2 = EncoderBlock(filters, filters*2)
        self.eblock3 = EncoderBlock(filters*2, filters*4)
        self.eblock4 = EncoderBlock(filters*4, filters*8, is_last=True)
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(filters, deconv, 2, stride=2),
            nn.Conv3d(deconv, n_classes, 1),
            nn.Sigmoid()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(filters*2, deconv*2, 2, 2),
            nn.ConvTranspose3d(deconv*2, deconv, 2, 2),
            nn.Conv3d(deconv, n_classes, 1),
            nn.Sigmoid()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(filters*4, deconv*4, 2, 2),
            nn.ConvTranspose3d(deconv*4, deconv*2, 2, 2),
            nn.ConvTranspose3d(deconv*2, deconv, 2, 2),
            nn.Conv3d(deconv, n_classes, 1),
            nn.Sigmoid()          
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(filters*8, deconv*8, 2, 2),
            nn.ConvTranspose3d(deconv*8, deconv*4, 2, 2),
            nn.ConvTranspose3d(deconv*4, deconv*2, 2, 2),
            nn.ConvTranspose3d(deconv*2, deconv, 2, 2),
            nn.Conv3d(deconv, n_classes, 1),
            nn.Sigmoid()           
        )    
            
    def forward(self, x):
        x, _ = self.eblock1(x)
        x1, skip1 = self.eblock2(x)
        x2, skip2 = self.eblock3(x1)
        x3, skip3 = self.eblock4(x2)
        
        ds1 = self.deconv1(skip1)
        ds2 = self.deconv2(skip2)
        ds3 = self.deconv3(skip3)
        pred = self.deconv4(x3)
        
        return pred, ds1, ds2, ds3