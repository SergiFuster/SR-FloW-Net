# https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels):
        super(EncoderBlock, self).__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = self.activation(self.conv1(x))
        x = self.pooling(x)

        x = self.activation(self.conv2(x))
        x = self.pooling(x)

        x = self.activation(self.conv3(x))
        x = self.pooling(x)

        x = self.activation(self.conv4(x))
        x = self.pooling(x)

        return x
    
class DualUNet(nn.Module):
    def __init__(self, in_channels):
        super(DualUNet,self).__init__()

        # region GENERIC LAYERS

        self.maxpooling_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling_2x2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.ReLU()

        # endregion

        # region NORMALIZATION BLOCK

        # Normalization for S2
        self.encoder = EncoderBlock(in_channels=in_channels)
        self.conv_s2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_s2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Normalization for S3
        self.conv_s3_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv_s3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # endregion

        # region ENCODER BLOCK

        # Encoder for S2
        self.conv_s2_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_s2_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv_s2_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_s2_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        # Encoder for Mixed
        self.conv_mixed_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_mixed_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv_mixed_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_mixed_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv_mixed_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_mixed_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # endregion

        # region BRIDGE BLOCK

        self.conv_bridge_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_bridge_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # endregion

        # region DECODER BLOCK

        self.conv_decoder_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv_decoder_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.conv_decoder_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_decoder_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.conv_decoder_5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_decoder_6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.conv_decoder_7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_decoder_8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # endregion

        # region OUTPUT BLOCK

        self.conv_output = nn.Conv2d(64, 16, 1, padding=0)

        # endregion

    def forward(self, s2, s3):
        # region NORMALIZATION BLOCK
        # Normalization for S2
        s2 = F.interpolate(s2, size= tuple(elem * 16 for elem in s3.shape[2:]), mode='bicubic', align_corners=True)
        s2 = self.encoder(s2)
        s2 = self.activation(self.conv_s2_1(s2))
        s2 = self.activation(self.conv_s2_2(s2))

        # Normalization for S3
        s3 = self.activation(self.conv_s3_1(s3))
        s3 = self.activation(self.conv_s3_2(s3)) # First skip connection and input for encoder concat 1
        # endregion

        # region ENCODER BLOCK
        # Encoder for S2
        s2_encoder_1 = self.maxpooling_2x2(s2)
        s2_encoder_1 = self.activation(self.conv_s2_3(s2_encoder_1))
        s2_encoder_1 = self.activation(self.conv_s2_4(s2_encoder_1)) # Input for encoder concat 2

        s2_encoder_2 = self.maxpooling_2x2(s2_encoder_1)
        s2_encoder_2 = self.activation(self.conv_s2_5(s2_encoder_2))
        s2_encoder_2 = self.activation(self.conv_s2_6(s2_encoder_2)) # Input for encoder concat 3

        # Concatenations
        encoder_concat_1 = torch.cat((s2, s3), 1)
        encoder_concat_1 = self.maxpooling_2x2(encoder_concat_1)
        encoder_concat_1 = self.activation(self.conv_mixed_1(encoder_concat_1))
        encoder_concat_1 = self.activation(self.conv_mixed_2(encoder_concat_1)) # Second skip connection and input for encoder concat 2

        encoder_concat_2 = torch.cat((s2_encoder_1, encoder_concat_1), 1)
        encoder_concat_2 = self.maxpooling_2x2(encoder_concat_2)
        encoder_concat_2 = self.activation(self.conv_mixed_3(encoder_concat_2))
        encoder_concat_2 = self.activation(self.conv_mixed_4(encoder_concat_2)) # Third skip connection and input for encoder concat 3

        encoder_concat_3 = torch.cat((s2_encoder_2, encoder_concat_2), 1)
        encoder_concat_3 = self.maxpooling_2x2(encoder_concat_3)
        encoder_concat_3 = self.activation(self.conv_mixed_5(encoder_concat_3))
        encoder_concat_3 = self.activation(self.conv_mixed_6(encoder_concat_3)) # Fourth skip connection and input for bridge

        # endregion

        # region BRIDGE BLOCK
        bridge = self.maxpooling_2x2(encoder_concat_3)
        bridge = self.activation(self.conv_bridge_1(bridge))
        bridge = self.activation(self.conv_bridge_2(bridge)) # Input for decoder concat 1
        # endregion

        # region DECODER BLOCK
        # (From bottom to top)
        # Level 1
        bridge = self.upsampling_2x2(bridge)
        decoder_concat_1 = torch.cat((encoder_concat_3, bridge), 1)
        decoder_concat_1 = self.activation(self.conv_decoder_1(decoder_concat_1))
        decoder_concat_1 = self.activation(self.conv_decoder_2(decoder_concat_1)) # Input for decoder concat 2

        # Level 2
        decoder_concat_1 = self.upsampling_2x2(decoder_concat_1)
        decoder_concat_2 = torch.cat((encoder_concat_2, decoder_concat_1), 1)
        decoder_concat_2 = self.activation(self.conv_decoder_3(decoder_concat_2))
        decoder_concat_2 = self.activation(self.conv_decoder_4(decoder_concat_2)) # Input for decoder concat 3

        # Level 3
        decoder_concat_2 = self.upsampling_2x2(decoder_concat_2)
        decoder_concat_3 = torch.cat((encoder_concat_1, decoder_concat_2), 1)
        decoder_concat_3 = self.activation(self.conv_decoder_5(decoder_concat_3))
        decoder_concat_3 = self.activation(self.conv_decoder_6(decoder_concat_3)) # Input for decoder concat 4

        # Level 4
        decoder_concat_3 = self.upsampling_2x2(decoder_concat_3)
        decoder_concat_4 = torch.cat((s3, decoder_concat_3), 1)
        decoder_concat_4 = self.activation(self.conv_decoder_7(decoder_concat_4))
        decoder_concat_4 = self.activation(self.conv_decoder_8(decoder_concat_4)) # Input for output

        # Output conv
        output = self.conv_output(decoder_concat_4)

        # endregion

        return output

class MinDualUNet(nn.Module):
    def __init__(self, in_channels):
        super(MinDualUNet,self).__init__()

        # region GENERIC LAYERS

        self.upsampling_2x2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.ReLU()

        # endregion

        # region NORMALIZATION BLOCK

        self.encoder = EncoderBlock(in_channels=in_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)

        # endregion

        # region ENCODER BLOCK
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        # endregion

        # region BRIDGE
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        # endregion

        # region DECODER BLOCK

        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, padding=0)
        # endregion

    def forward(self, s2, s3):

        # region ENCODER
        s2 = F.interpolate(s2, size= tuple(elem * 16 for elem in s3.shape[2:]), mode='bicubic', align_corners=True)
        s2_1 = self.encoder(s2)
        s2_2 = self.activation(self.conv1(s2_1))
        s2_3 = self.activation(self.conv2(s2_2))

        s3 = self.activation(self.conv3(s3)) # Skip connection 1
        mix_1 = torch.cat((s2_1, s3), 1)
        mix_1 = self.activation(self.conv4(mix_1)) # Skip connection 2
        mix_2 = torch.cat((s2_2, mix_1), 1)
        mix_2 = self.activation(self.conv5(mix_2)) # Skip connection 3
        mix_3 = torch.cat((s2_3, mix_2), 1)
        mix_3 = self.activation(self.conv6(mix_3)) # Skip connection 4
        # endregion
        
        # region BRIDGE
        curr = self.activation(self.conv7(mix_3))
        curr = self.upsampling_2x2(curr)
        # endregion

        # region DECODER
        curr = torch.cat((mix_3, curr), 1)
        curr = self.activation(self.conv8(curr))
        curr = self.upsampling_2x2(curr)
        curr = torch.cat((mix_2, curr), 1)
        curr = self.activation(self.conv9(curr))
        curr = self.upsampling_2x2(curr)
        curr = torch.cat((mix_1, curr), 1)
        curr = self.activation(self.conv10(curr))
        curr = self.upsampling_2x2(curr)
        curr = torch.cat((s3, curr), 1)
        output = self.activation(self.conv11(curr))
        # endregion

        return output

class MinNEDualUNet(nn.Module):
    def __init__(self, in_channels):
        super(MinNEDualUNet,self).__init__()

        # region GENERIC LAYERS

        self.upsampling_2x2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.ReLU()

        # endregion

        # region NORMALIZATION BLOCK

        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)

        # endregion

        # region ENCODER BLOCK
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        # endregion

        # region BRIDGE
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        # endregion

        # region DECODER BLOCK

        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, padding=0)
        # endregion

    def forward(self, s2, s3):

        # region ENCODER
        s2 = F.interpolate(s2, size=s3.shape[2:], mode='bicubic', align_corners=True)
        s2_1 = self.activation(self.conv0(s2))
        s2_2 = self.activation(self.conv1(s2_1))
        s2_3 = self.activation(self.conv2(s2_2))

        s3 = self.activation(self.conv3(s3)) # Skip connection 1
        mix_1 = torch.cat((s2_1, s3), 1)
        mix_1 = self.activation(self.conv4(mix_1)) # Skip connection 2
        mix_2 = torch.cat((s2_2, mix_1), 1)
        mix_2 = self.activation(self.conv5(mix_2)) # Skip connection 3
        mix_3 = torch.cat((s2_3, mix_2), 1)
        mix_3 = self.activation(self.conv6(mix_3)) # Skip connection 4
        # endregion
        
        # region BRIDGE
        curr = self.activation(self.conv7(mix_3))
        curr = self.upsampling_2x2(curr)
        # endregion

        # region DECODER
        curr = torch.cat((mix_3, curr), 1)
        curr = self.activation(self.conv8(curr))
        curr = self.upsampling_2x2(curr)
        curr = torch.cat((mix_2, curr), 1)
        curr = self.activation(self.conv9(curr))
        curr = self.upsampling_2x2(curr)
        curr = torch.cat((mix_1, curr), 1)
        curr = self.activation(self.conv10(curr))
        curr = self.upsampling_2x2(curr)
        curr = torch.cat((s3, curr), 1)
        output = self.activation(self.conv11(curr))
        # endregion

        return output

class SRUNet(nn.Module):
    def __init__(self, in_channels):
        super(SRUNet,self).__init__()

        # region GENERIC LAYERS
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.ReLU(inplace=False)
        # endregion

        """
        names: phase+conv+level+block+layer, for example: on econv132 is encoder, conv, level 1, block 3, layer 2
        """

        #region Encoder 1
        self.econv111 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.econv112 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.econv121 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.econv122 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.econv131 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.econv132 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.econv141 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.econv142 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #endregion
        #region Encoder 2
        self.econv211 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.econv212 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.econv221 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.econv222 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.econv231 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.econv232 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.econv241 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.econv242 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #endregion

        #region Encoder 3
        self.econv311 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv312 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv321 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv322 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv331 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv332 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv341 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv342 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv351 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv352 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv361 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.econv362 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Encoder 4
        self.econv411 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.econv412 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.econv421 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.econv422 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Bridge
        self.bconv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bconv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Decoder 1
        self.dconv111 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.dconv112 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Decoder 2
        self.dconv121 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        self.dconv122 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #endregion

        #region Decoder 3
        self.dconv131 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.dconv132 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #endregion

        #region Decoder 4
        self.dconv141 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        self.dconv142 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dconv143 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dconv144 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.dconv145 = nn.Conv2d(in_channels=16, out_channels=in_channels, kernel_size=3, padding=1)
        #endregion

    def forward(self, x, ratio):
        
        image_size = x.size(2) * ratio

        if image_size % 16 != 0: 
            raise ValueError("Image size must be a multiple of 16")
        
        x = F.interpolate(x, size=(image_size, image_size), mode='bicubic', align_corners=True)

        #region Encoder 1
        x = self.activation(self.econv111(x))
        x = self.activation(self.econv112(x))

        y = self.activation(self.econv121(x))
        y = self.activation(self.econv122(y))

        z = x + y

        x = self.activation(self.econv131(z))
        x = self.activation(self.econv132(x))

        y = x + z

        e1 = self.activation(self.econv141(y))
        e1 = self.activation(self.econv142(e1))
        #endregion

        #region Encoder 2
        x = self.maxpooling(e1)
        y = self.activation(self.econv211(x))
        y = self.activation(self.econv212(y))

        z = x + y

        x = self.activation(self.econv221(z))
        x = self.activation(self.econv222(x))

        y = z + x

        x = self.activation(self.econv231(y))
        x = self.activation(self.econv232(x))

        z = y + x

        e2 = self.activation(self.econv241(z))
        e2 = self.activation(self.econv242(e2))

        #endregion

        #region Encoder 3
        x = self.maxpooling(e2)
        y = self.activation(self.econv311(x))
        y = self.activation(self.econv312(y))

        z = x + y

        x = self.activation(self.econv321(z))
        x = self.activation(self.econv322(x))

        y = z + x

        x = self.activation(self.econv331(y))
        x = self.activation(self.econv332(x))

        z = x + y

        x = self.activation(self.econv341(z))
        x = self.activation(self.econv342(x))

        y = z + x

        x = self.activation(self.econv351(y))
        x = self.activation(self.econv352(x))

        z = x + y

        e3 = self.activation(self.econv361(z))
        e3 = self.activation(self.econv362(e3))
        #endregion

        #region Encoder 4
        x = self.maxpooling(e3)
        y = self.activation(self.econv411(x))
        y = self.activation(self.econv412(y))

        z = x + y

        x = self.activation(self.econv421(z))
        x = self.activation(self.econv422(x))

        y = x + z

        del z
        torch.cuda.empty_cache()

        e4 = self.activation(y)
        #endregion

        #region Bridge
        x = self.activation(self.bconv1(e4))
        x = self.activation(self.bconv2(x))
        #endregion

        #region Decoder 1
        y = torch.cat((x, e4), 1)
        del e4
        torch.cuda.empty_cache()

        x = self.activation(self.dconv111(y))
        x = self.activation(self.dconv112(x))

        x = self.upsampling(x)
        #endregion

        #region Decoder 2
        y = torch.cat((x, e3), 1)
        del e3
        torch.cuda.empty_cache()

        x = self.activation(self.dconv121(y))
        x = self.activation(self.dconv122(x))

        x = self.upsampling(x)
        #endregion

        #region Decoder 3
        y = torch.cat((x, e2), 1)
        del e2
        torch.cuda.empty_cache()

        x = self.activation(self.dconv131(y))
        x = self.activation(self.dconv132(x))

        x = self.upsampling(x)
        #endregion

        #region Decoder 4
        y = torch.cat((e1, x), 1)
        del e1
        torch.cuda.empty_cache()
        
        y = self.activation(self.dconv141(y))
        y = self.activation(self.dconv142(y))
        y = self.activation(self.dconv143(y))
        y = self.activation(self.dconv144(y))
        y = self.activation(self.dconv145(y))
        #endregion

        return y
        
class SRUNetv2(nn.Module):
    def __init__(self, in_channels):
        super(SRUNetv2,self).__init__()

        # region GENERIC LAYERS
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.ReLU(inplace=False)
        # endregion

        """
        names: phase+conv+level+block+layer, for example: on econv132 is encoder, conv, level 1, block 3, layer 2
        """

        #region Encoder 1
        self.econv111 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.econv112 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.econv121 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.econv122 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.econv131 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.econv132 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.econv141 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.econv142 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #endregion
        #region Encoder 2
        self.econv211 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.econv212 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.econv221 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.econv222 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.econv231 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.econv232 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.econv241 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.econv242 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #endregion

        #region Encoder 3
        self.econv311 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv312 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv321 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv322 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv331 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv332 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv341 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv342 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv351 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.econv352 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.econv361 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.econv362 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Encoder 4
        self.econv411 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.econv412 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.econv421 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.econv422 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Bridge
        self.bconv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bconv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Decoder 1
        self.dconv111 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.dconv112 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #endregion

        #region Decoder 2
        self.dconv211 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        self.dconv212 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #endregion

        #region Decoder 3
        self.dconv311 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.dconv312 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #endregion

        #region Decoder 4
        self.dconv411 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.dconv412 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.dconv421 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.dconv422 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.dconv431 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.dconv432 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.dconv441 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dconv442 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.dconv451 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dconv452 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.dconv453 = nn.Conv2d(in_channels=32, out_channels=in_channels, kernel_size=1, padding=0)

        #endregion

    def forward(self, x, upsamplings = 4):

        #region Encoder 1
        x = self.activation(self.econv111(x))
        x = self.activation(self.econv112(x))

        y = self.activation(self.econv121(x))
        y = self.activation(self.econv122(y))

        z = x + y

        x = self.activation(self.econv131(z))
        x = self.activation(self.econv132(x))

        y = x + z

        e1 = self.activation(self.econv141(y))
        e1 = self.activation(self.econv142(e1))
        #endregion

        #region Encoder 2
        x = self.maxpooling(e1)
        y = self.activation(self.econv211(x))
        y = self.activation(self.econv212(y))

        z = x + y

        x = self.activation(self.econv221(z))
        x = self.activation(self.econv222(x))

        y = z + x

        x = self.activation(self.econv231(y))
        x = self.activation(self.econv232(x))

        z = y + x

        e2 = self.activation(self.econv241(z))
        e2 = self.activation(self.econv242(e2))

        #endregion

        #region Encoder 3
        x = self.maxpooling(e2)
        y = self.activation(self.econv311(x))
        y = self.activation(self.econv312(y))

        z = x + y

        x = self.activation(self.econv321(z))
        x = self.activation(self.econv322(x))

        y = z + x

        x = self.activation(self.econv331(y))
        x = self.activation(self.econv332(x))

        z = x + y

        x = self.activation(self.econv341(z))
        x = self.activation(self.econv342(x))

        y = z + x

        x = self.activation(self.econv351(y))
        x = self.activation(self.econv352(x))

        z = x + y

        e3 = self.activation(self.econv361(z))
        e3 = self.activation(self.econv362(e3))
        #endregion

        #region Encoder 4
        x = self.maxpooling(e3)
        y = self.activation(self.econv411(x))
        y = self.activation(self.econv412(y))

        z = x + y

        x = self.activation(self.econv421(z))
        x = self.activation(self.econv422(x))

        y = x + z

        del z
        torch.cuda.empty_cache()

        e4 = self.activation(y)
        #endregion

        #region Bridge
        x = self.activation(self.bconv1(e4))
        x = self.activation(self.bconv2(x))
        #endregion

        #region Decoder 1
        y = torch.cat((x, e4), 1)
        del e4
        torch.cuda.empty_cache()

        x = self.activation(self.dconv111(y))
        x = self.activation(self.dconv112(x))

        x = self.upsampling(x)
        #endregion

        #region Decoder 2
        y = torch.cat((x, e3), 1)
        del e3
        torch.cuda.empty_cache()

        x = self.activation(self.dconv211(y))
        x = self.activation(self.dconv212(x))

        x = self.upsampling(x)
        #endregion

        #region Decoder 3
        y = torch.cat((x, e2), 1)
        del e2
        torch.cuda.empty_cache()

        x = self.activation(self.dconv311(y))
        x = self.activation(self.dconv312(x))

        x = self.upsampling(x)
        #endregion

        #region Decoder 4
        y = torch.cat((e1, x), 1)
        del e1
        torch.cuda.empty_cache()
        
        y = self.activation(self.dconv411(y))
        y = self.activation(self.dconv412(y))
        
        y = self.upsampling(y)

        y = self.activation(self.dconv421(y))
        y = self.activation(self.dconv422(y))

        if upsamplings > 1: y = self.upsampling(y)

        y = self.activation(self.dconv431(y))
        y = self.activation(self.dconv432(y))

        if upsamplings > 2: y = self.upsampling(y)

        y = self.activation(self.dconv441(y))
        y = self.activation(self.dconv442(y))

        if upsamplings > 3: y = self.upsampling(y)

        y = self.activation(self.dconv451(y))
        y = self.activation(self.dconv452(y))
        y = self.dconv453(y)
        #endregion

        return y

class NEDualUNet(nn.Module):
    def __init__(self, in_channels):
        super(NEDualUNet,self).__init__()

        # region GENERIC LAYERS

        self.maxpooling_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling_2x2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.ReLU()

        # endregion

        # region NORMALIZATION BLOCK

        # Normalization for S2
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv_s2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_s2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Normalization for S3
        self.conv_s3_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv_s3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # endregion

        # region ENCODER BLOCK

        # Encoder for S2
        self.conv_s2_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_s2_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv_s2_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_s2_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        #Encoder for Mixed
        self.conv_mixed_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_mixed_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv_mixed_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_mixed_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv_mixed_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_mixed_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # endregion

        # region BRIDGE BLOCK

        self.conv_bridge_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_bridge_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # endregion

        # region DECODER BLOCK

        self.conv_decoder_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv_decoder_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.conv_decoder_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_decoder_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.conv_decoder_5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_decoder_6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.conv_decoder_7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_decoder_8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # endregion

        # region OUTPUT BLOCK

        self.conv_output = nn.Conv2d(64, 16, 1, padding=0)

        # endregion

    def forward(self, s2, s3):
        # region NORMALIZATION BLOCK
        # Normalization for S2
        s2 = self.activation(self.conv0(s2))
        s2 = self.activation(self.conv_s2_1(s2))
        s2 = self.activation(self.conv_s2_2(s2))

        # Normalization for S3
        s3 = self.activation(self.conv_s3_1(s3))
        s3 = self.activation(self.conv_s3_2(s3)) # First skip connection and input for encoder concat 1
        # endregion

        # region ENCODER BLOCK
        # Encoder for S2
        s2_encoder_1 = self.maxpooling_2x2(s2)
        s2_encoder_1 = self.activation(self.conv_s2_3(s2_encoder_1))
        s2_encoder_1 = self.activation(self.conv_s2_4(s2_encoder_1)) # Input for encoder concat 2

        s2_encoder_2 = self.maxpooling_2x2(s2_encoder_1)
        s2_encoder_2 = self.activation(self.conv_s2_5(s2_encoder_2))
        s2_encoder_2 = self.activation(self.conv_s2_6(s2_encoder_2)) # Input for encoder concat 3

        # Concatenations
        encoder_concat_1 = torch.cat((s2, s3), 1)
        encoder_concat_1 = self.maxpooling_2x2(encoder_concat_1)
        encoder_concat_1 = self.activation(self.conv_mixed_1(encoder_concat_1))
        encoder_concat_1 = self.activation(self.conv_mixed_2(encoder_concat_1)) # Second skip connection and input for encoder concat 2

        encoder_concat_2 = torch.cat((s2_encoder_1, encoder_concat_1), 1)
        encoder_concat_2 = self.maxpooling_2x2(encoder_concat_2)
        encoder_concat_2 = self.activation(self.conv_mixed_3(encoder_concat_2))
        encoder_concat_2 = self.activation(self.conv_mixed_4(encoder_concat_2)) # Third skip connection and input for encoder concat 3

        encoder_concat_3 = torch.cat((s2_encoder_2, encoder_concat_2), 1)
        encoder_concat_3 = self.maxpooling_2x2(encoder_concat_3)
        encoder_concat_3 = self.activation(self.conv_mixed_5(encoder_concat_3))
        encoder_concat_3 = self.activation(self.conv_mixed_6(encoder_concat_3)) # Fourth skip connection and input for bridge

        # endregion

        # region BRIDGE BLOCK
        bridge = self.maxpooling_2x2(encoder_concat_3)
        bridge = self.activation(self.conv_bridge_1(bridge))
        bridge = self.activation(self.conv_bridge_2(bridge)) # Input for decoder concat 1
        # endregion

        # region DECODER BLOCK
        # (From bottom to top)
        # Level 1
        bridge = self.upsampling_2x2(bridge)
        decoder_concat_1 = torch.cat((encoder_concat_3, bridge), 1)
        decoder_concat_1 = self.activation(self.conv_decoder_1(decoder_concat_1))
        decoder_concat_1 = self.activation(self.conv_decoder_2(decoder_concat_1)) # Input for decoder concat 2

        # Level 2
        decoder_concat_1 = self.upsampling_2x2(decoder_concat_1)
        decoder_concat_2 = torch.cat((encoder_concat_2, decoder_concat_1), 1)
        decoder_concat_2 = self.activation(self.conv_decoder_3(decoder_concat_2))
        decoder_concat_2 = self.activation(self.conv_decoder_4(decoder_concat_2)) # Input for decoder concat 3

        # Level 3
        decoder_concat_2 = self.upsampling_2x2(decoder_concat_2)
        decoder_concat_3 = torch.cat((encoder_concat_1, decoder_concat_2), 1)
        decoder_concat_3 = self.activation(self.conv_decoder_5(decoder_concat_3))
        decoder_concat_3 = self.activation(self.conv_decoder_6(decoder_concat_3)) # Input for decoder concat 4

        # Level 4
        decoder_concat_3 = self.upsampling_2x2(decoder_concat_3)
        decoder_concat_4 = torch.cat((s3, decoder_concat_3), 1)
        decoder_concat_4 = self.activation(self.conv_decoder_7(decoder_concat_4))
        decoder_concat_4 = self.activation(self.conv_decoder_8(decoder_concat_4)) # Input for output

        # Output conv
        output = self.conv_output(decoder_concat_4)

        # endregion

        return output

class U_Net2(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net2,self).__init__()
        
               #nb_unet_features = [
            #[16, 32, 32, 32],               # encoder features
            #[32, 32, 32, 32, 32, 16, 16] 
        
        
        # encoder
        self.Conv1 = nn.Conv2d(in_channels=img_ch,out_channels=16,kernel_size=3,stride=2,padding=1)
        self.Conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.Conv3 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.Conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        
        # decoder
        self.Conv5 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.Conv6 = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.Conv7 = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.Conv8 = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=3,stride=1,padding=1)
        
        # extra Conv
        self.Conv9 = nn.Conv2d(in_channels=32+16,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.Conv10 = nn.Conv2d(in_channels=output_ch,out_channels=16,kernel_size=3,stride=1,padding=1)
    
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.LeakyReLU(0.2)

    


    def forward(self,x):
        
        # encoding path
        x1 = self.activation(self.Conv1(x))
        x2 = self.activation(self.Conv2(x1))
        x3 = self.activation(self.Conv3(x2))
        x4 = self.activation(self.Conv4(x3))

        # decoding + concat path
        d5 = self.activation(self.Conv5(x4))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.upsample(d5)

        d6 = self.activation(self.Conv6(d5))
        d6 = torch.cat((x3,d6),dim=1)
        d6 = self.upsample(d6)

        d7 = self.activation(self.Conv7(d6))
        d7 = torch.cat((x2,d7),dim=1)
        d7 = self.upsample(d7)

        d8 = self.activation(self.Conv8(d7))
        d8 = torch.cat((x1,d8),dim=1)
        d8 = self.upsample(d8)

        e1 = self.activation(self.Conv9(d8))
        e2 = self.activation(self.Conv10(e1))

        return e2