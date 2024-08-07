import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions.normal import Normal
from .unets import U_Net2, DualUNet, MinDualUNet, NEDualUNet, MinNEDualUNet, SRUNetv2
from .layers import SpatialTransformer
 
class RegNet(nn.Module):

    def __init__(self,img_shape, RATIO=1, CHAN_S2=1, CHAN_S3=1, backbone_in_ch=2, backbone_out_ch=16):

        super(RegNet, self).__init__()

        self.ratio = RATIO
        self.cs2 = CHAN_S2
        self.cs3 = CHAN_S3
        self.chan_s2 = nn.MaxPool3d(kernel_size=(4,1,1), stride = (1,1,1), padding = 0)
        self.chan_s3 = nn.MaxPool3d(kernel_size=(4,1,1), stride = (1,1,1), padding = 0)
        
        self.unchan = nn.Conv2d(1, self.cs3, 1, stride = 1, padding = 0)

        self.down_s2 = nn.Conv2d(1, 1, RATIO, stride=RATIO, padding=RATIO//2)
        
                      
        self.unet_model = U_Net2(
            img_ch=backbone_in_ch,
            output_ch=backbone_out_ch
        )
        
        
        '''
        nb_unet_features = [
            [16, 32, 32, 32],               # encoder features
            [32, 32, 32, 32, 32, 16, 16]    # decoder features
        ]
        backbone_out_ch = nb_unet_features[1][-1]
        nb_unet_levels = None
        unet_feat_mult = 1
        
        
        self.unet_model = Unet(
            inshape=img_shape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )
        '''

        # configure unet to flow field layer
        self.flow = nn.Conv2d(in_channels=backbone_out_ch, out_channels=len(img_shape), kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        '''
        int_steps = 7
        int_downsize = 2
        
        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(d / int_downsize) for d in img_shape]
        self.integrate = layers.VecInt(down_shape, int_steps)

        # configure optional resize layers
        self.resize = layers.ResizeTransform(int_downsize, len(img_shape))
        self.fullsize = layers.ResizeTransform(1 / int_downsize, len(img_shape))
        '''


        self.transformer = SpatialTransformer(img_shape)



    

    def forward(self, target, source, registration=False):

        # Source = S3, Target = S2 
        # Processing s2 data if required
        
        # ------------ HEAD ------------ #
        if self.cs3>1:
            source = self.chan_s3(source)
        
        if self.cs2>1:
            target = self.chan_s2(target)

        if self.ratio>1:
            target = self.down_s2(target)
        
        # Concatenate inputs (s3+s2)
        x = torch.cat([source, target], dim=1)

        # ------------ BODY ------------ #
        
        # Propagate U-Net
        x = self.unet_model(x)
        # ------------ TAIL ------------ #

        # Transformation into flow field
        flow_field = self.flow(x)
        # Resize flow for integration
        #flow_field = self.resize(flow_field)
        # Integrate to produce diffeomorphic warp
        #flow_field = self.integrate(flow_field)
        # Resize to final resolution
        #flow_field = self.fullsize(flow_field)

        # Warp image with flow field
        y_source = self.transformer(source, flow_field)
        if self.cs3>1:
            y_source = self.unchan(y_source)
        return y_source, flow_field

class FullRegNet(nn.Module):

    def __init__(self, input_size, sr_state_dict=None):
        channels, heigth, width = input_size
        assert heigth % 256 == 0, "Size of s2 have to be multiple of 256 because S3 is going to be downsampled to S2 size / 16 and next going to SR module (16 * 16)"
        super(FullRegNet, self).__init__()                 
        self.unet_model = NEDualUNet(channels)

        self.flow = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.transformer = SpatialTransformer((heigth, width))

        self.super_resolution = SRUNetv2(channels)

        if sr_state_dict: self.super_resolution.load_state_dict(sr_state_dict)

    def forward(self, s2, s3):
        assert s2.shape[2] % 16 == 0, 'Size of s2 not multiple of 16'

        """
        As original aspect ratio between s2 and s3 15 we have to normalize it to 16 first
        We could upsample s2 or downsample s3, we choose the second option.
        """
        s3_downsampled = F.interpolate(s3, size= tuple(elem // 16 for elem in s2.shape[2:]), mode='bilinear')
        s3_supersampled = self.super_resolution(s3_downsampled)

        unet_out = self.unet_model(s2, s3_supersampled)
        flow_field = self.flow(unet_out)

        registered = self.transformer(s3_supersampled, flow_field)

        return registered, flow_field, s3_supersampled

class SRegNet(nn.Module):

    def __init__(self, input_size, verbose=False):
        """
        Input size have to by a tuple of (channels, height, width) representing
        the shape of s3.
        """
        channels, heigth, width = input_size
        super(SRegNet, self).__init__()        
        self.verbose = verbose                      
        self.unet_model = DualUNet(channels)

        self.flow = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.transformer = SpatialTransformer((heigth, width))

    def forward(self, s2, s3):
        # input format -> (bands, heigth, width)
        # s3 waited size (1, x, y)
        # s2 waited size (1, x*15, y*15)
        # ------------ HEAD ------------ #
            
        # ------------ BODY ------------ #
        unet_out = self.unet_model(s2, s3)
        if self.verbose: print(f'Dual U-NET output shape: {unet_out.shape}')
        # ------------ TAIL ------------ #
        flow_field = self.flow(unet_out)
        if self.verbose: print(f'Flow field shape: {flow_field.shape}')
        registered = self.transformer(s3, flow_field)
        if self.verbose: print(f'Registered image shape: {registered.shape}')
        
        return registered, flow_field

class NESRegNet(nn.Module):

    def __init__(self, input_size, verbose=False):
        """
        Input size have to by a tuple of (channels, height, width) representing
        the shape of s3.
        """
        heigth, width, channels = input_size

        super(NESRegNet, self).__init__()        
        self.verbose = verbose                      
        self.unet_model = NEDualUNet(channels)

        self.flow = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.transformer = SpatialTransformer((heigth, width))

    def forward(self, forward, to_register=None):
        """
        Different images can be used for calculate the flow field and to register.
        parameters:
            forward: pair of image to perform the forward step (s2, s3)
            to_register: optional, image that will be registered,
            if not provided second input of forward will be used.
        """

        # input format -> (bands, heigth, width)
        # s3 waited size (1, x, y)
        # s2 waited size (1, x*15, y*15)
        s2, s3 = forward
        # ------------ BODY ------------ #
        unet_out = self.unet_model(s2, s3)
        if self.verbose: print(f'Dual U-NET output shape: {unet_out.shape}')
        # ------------ TAIL ------------ #
        flow_field = self.flow(unet_out)
        if self.verbose: print(f'Flow field shape: {flow_field.shape}')
        if to_register is None: to_register = s3
        registered = self.transformer(to_register, flow_field)
        if self.verbose: print(f'Registered image shape: {registered.shape}')
        
        return registered, flow_field
    
class MinSRegNet(nn.Module):

    def __init__(self, input_size, verbose=False):
        """
        Input size have to by a tuple of (channels, height, width) representing
        the shape of s3.
        """
        channels, heigth, width = input_size
        super(MinSRegNet, self).__init__()        
        self.verbose = verbose                      
        self.unet_model = MinDualUNet(channels)

        self.flow = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.transformer = SpatialTransformer((heigth, width))

    def forward(self, s2, s3):
        
        # input format -> (bands, heigth, width)
        # s3 waited size (1, x, y)
        # s2 waited size (1, x*15, y*15)

        # ------------ BODY ------------ #
        unet_out = self.unet_model(s2, s3)
        if self.verbose: print(f'Dual U-NET output shape: {unet_out.shape}')
        # ------------ TAIL ------------ #
        flow_field = self.flow(unet_out)
        if self.verbose: print(f'Flow field shape: {flow_field.shape}')
        registered = self.transformer(s3, flow_field)
        if self.verbose: print(f'Registered image shape: {registered.shape}')
        
        return registered, flow_field

class MinNESRegNet(nn.Module):

    def __init__(self, input_size, verbose=False):
        """
        Input size have to by a tuple of (channels, height, width) representing
        the shape of s3.
        """
        channels, heigth, width = input_size
        super(MinNESRegNet, self).__init__()        
        self.verbose = verbose                      
        self.unet_model = MinNEDualUNet(channels)

        self.flow = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.transformer = SpatialTransformer((heigth, width))

    def forward(self, s2, s3):
        
        # input format -> (bands, heigth, width)
        # s3 waited size (1, x, y)
        # s2 waited size (1, x*15, y*15)
        # ------------ HEAD ------------ #
            
        # ------------ BODY ------------ #
        unet_out = self.unet_model(s2, s3)
        if self.verbose: print(f'Dual U-NET output shape: {unet_out.shape}')
        # ------------ TAIL ------------ #
        flow_field = self.flow(unet_out)
        if self.verbose: print(f'Flow field shape: {flow_field.shape}')
        registered = self.transformer(s3, flow_field)
        if self.verbose: print(f'Registered image shape: {registered.shape}')
        
        return registered, flow_field