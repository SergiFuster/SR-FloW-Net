import torch
import torch.nn.functional as F
import numpy as np
import math
    
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, scale=1, kernel=9):
        self.kernel_size = kernel
        self.scale = scale

    def loss(self, y_true, y_pred, verbose=True):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.scale==1:
            I = y_true
        else:
            I = F.interpolate(y_true, scale_factor=1/self.scale, mode='bicubic', align_corners=False, recompute_scale_factor=False)
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.kernel_size] * ndims
        channel = 1
        # compute filters
        sum_filt = torch.ones([1, channel, *win]).to(device)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # Check if CUDA (GPU) is available, and move tensors accordingly

        # Move tensors to the same device
        sum_filt = sum_filt.to(device)
        I = I.to(device)
        J = J.to(device)


        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class CC3D():
    def __init__(self, kernel_size=[9, 9, 9]):
        self.kernel = kernel_size
    
    def loss(self, I, J):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if I.dim == 4: I, J = I.unsqueeze(1), J.unsqueeze(1)

        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = torch.ones([1, 1, *(self.kernel)]).to(device)

        I_sum = F.conv3d(I, filt, padding='same')
        J_sum = F.conv3d(J, filt, padding='same')
        I2_sum = F.conv3d(I2, filt, padding='same')
        J2_sum = F.conv3d(J2, filt, padding='same')
        IJ_sum = F.conv3d(IJ, filt, padding='same')

        win_size = np.prod(self.kernel)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights
        return -torch.mean(cc)

class LNCC2D():
    def __init__(self, kernels_size=9):
        self.kernel_size = kernels_size

    def loss(self, I, J):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.kernel_size] * ndims
        channels = I.shape[1]
        # compute filters
        sum_filt = torch.ones([1, channels, *win]).to(device)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)


        # Move tensors to the same device
        sum_filt = sum_filt.to(device)
        I = I.to(device)
        J = J.to(device)


        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, sum_filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, sum_filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class L2():

    def __init__(self):
        pass

    def loss(self, I, J):

        if I.shape[2] > J.shape[2]:
            I = F.interpolate(I, (J.shape[2], J.shape[3]), mode='bicubic', align_corners=False, recompute_scale_factor=False)
        else:
            J = F.interpolate(J, (I.shape[2], I.shape[3]), mode='bicubic', align_corners=False, recompute_scale_factor=False)

        # Calculamos la diferencia entre los dos tensores
        diff = I - J
        
        # Calculamos la norma L2 para cada elemento del batch
        norma_L2 = torch.norm(diff.view(diff.size(0), -1), dim=1)

        return norma_L2.mean()
    
class LNCC3D():
    def __init__(self, kernel_size=9):
        self.kernel = [kernel_size] * 2 + [1]
    
    def loss(self, I, J):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if I.dim == 4: I, J = I.unsqueeze(1), J.unsqueeze(1)

        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = torch.ones([1, 1, *(self.kernel)]).to(device)

        I_sum = F.conv3d(I, filt, padding='same')
        J_sum = F.conv3d(J, filt, padding='same')
        I2_sum = F.conv3d(I2, filt, padding='same')
        J2_sum = F.conv3d(J2, filt, padding='same')
        IJ_sum = F.conv3d(IJ, filt, padding='same')

        win_size = np.prod(self.kernel)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cross = torch.sum(cross, axis=1)
        I_var = torch.sum(I_var, axis=1)
        J_var = torch.sum(J_var, axis=1)
        
        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights
        return -torch.mean(cc)

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        #dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        #dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        #dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])         

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            #dz = dz * dz

        #d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        #grad = d / 3.0
        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad











