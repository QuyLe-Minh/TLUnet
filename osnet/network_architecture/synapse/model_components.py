from torch import nn
from torch.nn import functional as F
from monai.utils import optional_import
from osnet.network_architecture.dynunet_block import UnetResBlock, LightConv3x3, LightConv3x3withDilation, ChannelGate, Conv1x1, Conv1x1Linear
from osnet.network_architecture.synapse.transformerblock import TransformerBlock
from monai.networks.layers.utils import get_act_layer, get_norm_layer

einops, _ = optional_import("einops")

class OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.lrelu = get_act_layer(name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))

    def forward_features(self, fn, x):
        inp = x
        attn = fn(x)
        x = inp * attn
        return x

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return self.lrelu(out)
    
class OSBlockwithDilation(nn.Module):
    """Omni-scale feature learning block."""
    
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlockwithDilation, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3withDilation(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3withDilation(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.lrelu = get_act_layer(name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))

    def forward_features(self, fn, x):
        inp = x
        attn = fn(x)
        x = inp * attn
        return x

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        # x2a = self.forward_features(self.conv2a, x1)
        # x2b = self.forward_features(self.conv2b, x1)
        # x2c = self.forward_features(self.conv2c, x1)
        # x2d = self.forward_features(self.conv2d, x1)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return self.lrelu(out)
    
class Reconstructor(nn.Module):
    def __init__(self, 
                 in_channels,
                 filters=64, 
                 ):
        super().__init__()
        self.in_planes = filters*4
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv_block2 = self._make_layer(UnetResBlock, 128, 2)
        self.up3 = nn.Upsample(scale_factor=2)
        self.conv_block3 = self._make_layer(UnetResBlock, 64, 2)
        self.up4 = nn.Upsample(scale_factor=(2,4,4))
        self.conv_block4 = self._make_layer(UnetResBlock, 32, 2)
        self.conv = nn.Conv3d(32, in_channels, stride=1, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks):
        modules = []
        for _ in range(blocks):
            modules.append(block(3, self.in_planes, planes, 3, stride=1, norm_name='batch'))
            self.in_planes = planes
        
        return nn.Sequential(*modules)
    
    def forward(self, inp):
        x = self.up2(inp)
        x = self.conv_block2(x)
        x = self.up3(x)
        x = self.conv_block3(x)
        x = self.up4(x)
        x = self.conv_block4(x)
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channels, dims = [64, 128, 256, 512], depths = [2,2,2,2], input_size=[8*8*8, 4*4*4], proj_size=[64, 32], **kwargs):
        super(Encoder, self).__init__() 
        
        self.downsample_layers = nn.ModuleList()
        stem_layer = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=(2,4,4), stride=(2,4,4), bias=False),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0])
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv3d(dims[i], dims[i+1], kernel_size=(2,2,2), stride=(2,2,2)),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1])
            )
            self.downsample_layers.append(downsample_layer)
            
        self.stages = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for _ in range(depths[i]):
                # if i >= 2:
                #     stage_blocks.append(TransformerBlock(input_size=input_size[i-2], hidden_size=dims[i], proj_size=proj_size[i-2], num_heads=4, dropout_rate=0.15, pos_embed=True))
                if i < 2:
                    stage_blocks.append(OSBlockwithDilation(dims[i], dims[i]))
                else:
                    stage_blocks.append(OSBlock(dims[i], dims[i]))
            self.stages.append(nn.Sequential(*stage_blocks))
            
        self.hidden_states = []
        
    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        # print(x.shape)
        x = self.stages[0](x)
        # print(x.shape)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            # print(x.shape)
            x = self.stages[i](x)
            # print(x.shape)
            hidden_states.append(x)
            
        return x, hidden_states
    
    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states
    
class OSNUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_kernel_size, depth = 2, dilation=False, **kwargs):
        super(OSNUpBlock, self).__init__()
        
        upsample_stride = upsample_kernel_size
        self.transpose_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride)
        self.decode_layers = nn.ModuleList()
        
        for _ in range(depth):
            # if transformer:
            #     self.decode_layers.append(TransformerBlock(input_size=64, hidden_size=out_channels, proj_size=64, num_heads=4, dropout_rate=0.15, pos_embed=True))
            if dilation:
                self.decode_layers.append(OSBlockwithDilation(out_channels, out_channels))
            else:
                self.decode_layers.append(OSBlock(out_channels, out_channels))
            
    def forward(self, x, skip):
        # print("x:", x.shape)
        out = self.transpose_conv(x)
        # print(out.shape)
        out = out+skip
        # print(out.shape)
        out = self.decode_layers[0](out)
        # print(out.shape)
        return out
    
