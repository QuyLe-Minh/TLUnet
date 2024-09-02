from torch import nn
from osnet.network_architecture.neural_network import SegmentationNetwork
from osnet.network_architecture.dynunet_block import UnetResBlock, UnetOutBlock
from osnet.network_architecture.synapse.model_components import Encoder, OSNUpBlock, OSBlock, Reconstructor, OSBlockwithDilation


class OSNet(nn.Module):

    def __init__(self, 
                 in_channels,
                 out_channels, 
                 hidden_size=32,
                 img_size=[64, 128, 128], 
                 depths = [2,2,2,2], 
                 dims=[64, 128, 256, 512],
                 do_ds=True,
                 **kwargs):

        super().__init__()
        
        if depths is None:
            depths = [2,2,2,2]
        if dims is None:
            dims = [80, 160, 320, 640]

        self.do_ds = do_ds
        self.num_classes = out_channels

        self.encoder = Encoder(in_channels, dims, depths)
        self.feature_extract = OSBlockwithDilation(in_channels=in_channels, out_channels=hidden_size)
        
        self.decoder5 = OSNUpBlock(in_channels=hidden_size*16, out_channels=hidden_size*8, upsample_kernel_size=2)
        self.decoder4 = OSNUpBlock(in_channels=hidden_size*8, out_channels=hidden_size*4, upsample_kernel_size=2)
        self.decoder3 = OSNUpBlock(in_channels=hidden_size*4, out_channels=hidden_size*2, upsample_kernel_size=2, dilation=True)
        self.decoder2 = OSNUpBlock(in_channels=hidden_size*2, out_channels=hidden_size, upsample_kernel_size=(2,4,4), depth=1, dilation=True)

        self.out1 = nn.Conv3d(hidden_size, out_channels, kernel_size=1)
        if self.do_ds:
            self.out2 = nn.Conv3d(hidden_size*2, out_channels, kernel_size=1)
            self.out3 = nn.Conv3d(hidden_size*4, out_channels, kernel_size=1)


    
    def forward(self, inp):
        x_output, hidden_states = self.encoder(inp)
        convBlock = self.feature_extract(inp)

        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        # enc4 = hidden_states[3]


        dec4 = self.decoder5(x_output, enc3)
        dec3 = self.decoder4(dec4, enc2)
        dec2 = self.decoder3(dec3, enc1)
        dec1 = self.decoder2(dec2, convBlock)

        if self.do_ds:
            logits = [self.out1(dec1), self.out2(dec2), self.out3(dec3)]
        else:
            logits = self.out1(dec1)

        return dec4, logits
    
class Network_with_Adversarial(SegmentationNetwork):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 hidden_size=32,
                 img_size=[64, 128, 128], 
                 depths = [2,2,2,2], 
                 dims=[64, 128, 256, 512],
                 do_ds=True,
                 **kwargs):
        super().__init__()
        self.segmentation_network = OSNet(in_channels, out_channels, hidden_size, img_size, depths, dims, do_ds)
        self.reconstructor = Reconstructor(in_channels)
        self.do_ds = do_ds

    def forward(self, inp, require_img=False):
        dec4, logits = self.segmentation_network(inp)
        if require_img:
            return self.reconstructor(dec4), logits
        else:
            if not self.do_ds:
                return logits[0]
            else:
                return logits

