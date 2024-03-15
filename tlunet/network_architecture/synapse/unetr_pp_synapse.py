import torch
from torch import nn
from typing import Tuple, Union
from tlunet.network_architecture.neural_network import SegmentationNetwork
from tlunet.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
# from tlunet.network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock
from tlunet.network_architecture.synapse.model_components import EncoderBlock, ConvBlock, DeconvBlock


class TLUnet(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(self, out_channels, do_ds):

        super().__init__()
        self.do_ds = do_ds
        filters = 32
        n_deconv = 96
        self.num_classes = out_channels
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
            UnetOutBlock(spatial_dims=3, in_channels=filters//2, out_channels=out_channels)
        )
        
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=filters, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=filters * 2, out_channels=out_channels)


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
        
        d1, _ = self.deconv1(concat, skip3) 
        d2, out3 = self.deconv2(d1, skip2) 
        d3, out2 = self.deconv3(d2, skip1)
        
        if self.do_ds:
            logits = (self.head(d3), self.out2(out2), self.out3(out3))
        else:
            logits = self.head(d3)
        
        return logits
