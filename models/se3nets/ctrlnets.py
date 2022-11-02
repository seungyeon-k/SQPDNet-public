import torch
import torch.nn as nn
import torch.nn.functional as F
#from  torch.autograd import Variable
import models.se3nets.se3layers as se3nn
from functions import utils_torch
#import util

################################################################################
'''
    Helper functions
'''
# Choose non-linearities
def get_nonlinearity(nonlinearity):
    if nonlinearity == 'prelu':
        return nn.PReLU()
    elif nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity == 'tanh':
        return nn.Tanh()
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid()
    elif nonlinearity == 'elu':
        return nn.ELU(inplace=True)
    elif nonlinearity == 'selu':
        return nn.SELU()
    elif nonlinearity == 'none':
        return lambda x: x # Return input as is
    else:
        assert False, "Unknown non-linearity: {}".format(nonlinearity)

# Get SE3 dimension based on se3 type & pivot
def get_se3_dimension(se3_type, use_pivot):
    # Get dimension (6 for se3aa, se3euler, se3spquat, se3aar)
    se3_dim = 6
    if se3_type == 'se3quat' or se3_type == 'se3aa4':
        se3_dim = 7
    elif se3_type == 'affine':
        se3_dim = 12
    # Add pivot dimensions
    if use_pivot:
        se3_dim += 3
    return se3_dim

### Hook for backprops of variables, just prints min, max, mean of grads along with num of "NaNs"
def variable_hook(grad, txt):
    print(txt, grad.max().item(), grad.min().item(), grad.mean().item(), grad.ne(grad).sum().item())

### Initialize the SE3 prediction layer to identity
def init_se3layer_identity(layer, num_se3=8, se3_type='se3aa'):
    layer.weight.data.uniform_(-0.0001, 0.0001)  # Initialize weights to near identity
    layer.bias.data.uniform_(-0.0001, 0.0001)  # Initialize biases to near identity
    # Special initialization for specific SE3 types
    if se3_type == 'affine':
        bs = layer.bias.data.view(num_se3, 3, -1)
        bs.narrow(2, 0, 3).copy_(torch.eye(3).view(1, 3, 3).expand(num_se3, 3, 3))
    elif se3_type == 'se3quat':
        bs = layer.bias.data.view(num_se3, -1)
        bs.narrow(1, 6, 1).fill_(1)  # ~ [0,0,0,1]

### Initialize the last deconv layer such that the mask predicts BG for first channel & zero for all other channels
def init_sigmoidmask_bg(layer, num_se3=8):
    # Init weights to be close to zero, so that biases affect the result primarily
    layer.weight.data.uniform_(-1e-3,1e-3) # sigmoid(0) ~= 0.5

    # Init first channel bias to large +ve value
    layer.bias.data.narrow(0,0,1).uniform_(4.1,3.9) # sigmoid(4) ~= 1

    # Init other layers biases to large -ve values
    layer.bias.data.narrow(0,1,num_se3-1).uniform_(-4.1,-3.9)   # sigmoid(-4) ~= 0

################################################################################
'''
    NN Modules / Loss functions
'''
# TODO: The weights are only supposed to be binary here. Non binary weights will do weird things

# MSE Loss that gives gradients w.r.t both input & target
# Input/Target: Bsz x nSE3 x 3 x 4
# NOTE: This scales the loss by 0.5 while the default nn.MSELoss does not
def PoseConsistencyLoss(input, target, size_average=True):
    delta = se3nn.ComposeRtPair()(input, se3nn.RtInverse()(target)) # input * target^-1
    rot, trans = delta.narrow(3,0,3), delta.narrow(3,3,1)
    costheta = 0.5 * ((rot[:,:,0,0] + rot[:,:,1,1] + rot[:,:,2,2]) - 1.0) # torch.acos(0.5*(tr(R)-1)); Eqn 9.19: https://pixhawk.org/_media/dev/know-how/jlblanco2010geometry3d_techrep.pdf
    if size_average:
        loss = trans.norm(2,2).mean() + (costheta-1.0).abs().mean()
    else:
        loss = trans.norm(2,2).sum() + (costheta-1.0).abs().sum()
    return loss

# MSE Loss that gives gradients w.r.t both input & target
# NOTE: This scales the loss by 0.5 while the default nn.MSELoss does not
def BiMSELoss(input, target, size_average=True, wts=None):
    weights = wts.expand_as(input) if wts is not None else 1 # Per-pixel scalar
    loss = 0.5 * torch.sum(((input - target) * weights)**2)
    if size_average:
        npts = weights.sum() if wts is not None else input.nelement() # Normalize by visibility mask
        return loss / npts
    else:
        return loss

# ABS Loss that gives gradients w.r.t both input & target
def BiAbsLoss(input, target, size_average=True, wts=None):
    weights = wts.expand_as(input) if wts is not None else 1 # Per-pixel scalar
    loss = ((input - target).abs() * weights).view(-1).sum()
    if size_average:
        npts = weights.sum() if wts is not None else input.nelement() # Normalize by visibility mask
        return loss / npts
    else:
        return loss

# NORMMSESQRT Loss that gives gradients w.r.t both input & target
def BiNormMSESqrtLoss(input, target, size_average=True, norm_per_pt=False, wts=None):
    weights = wts.expand_as(input) if wts is not None else 1 # Per-pixel scalar
    if norm_per_pt:
        sigma = (0.5 * target.pow(2).sum(1).sqrt()).clamp(min=2e-3).unsqueeze(1).expand_as(target) # scale * ||y||_2, clamp for numerical stability (changed clamp to 2mm)
    else:
        sigma = (0.5 * target.abs()).clamp(min=2e-3) # scale * y, clamp for numerical stability (changed clamp to 2mm)
    diff = (input - target) * weights
    loss = 0.5 * diff.pow(2).div(sigma).sum()  # (x - y)^2 / 2*s^2 where s^2 = sigma, the variance
    if size_average:
        npts = weights.sum() if wts is not None else input.nelement() # Normalize by visibility mask
        return loss / npts
    else:
        return loss

# Loss for 3D point errors
def Loss3D(input, target, loss_type='mse', wts=None):
    if loss_type == 'mse':
        return BiMSELoss(input, target, wts=wts)
    elif loss_type == 'abs':
        return BiAbsLoss(input, target, wts=wts)
    elif loss_type == 'normmsesqrt':
        return BiNormMSESqrtLoss(input, target, wts=wts)
    elif loss_type == 'normmsesqrtpt':
        return BiNormMSESqrtLoss(input, target, norm_per_pt=True, wts=wts)
    else:
        assert False, "Unknown loss type: " + loss_type

# Loss normalized by the number of points that move in the GT flow
def MotionNormalizedLoss3D(input, target, motion, loss_type='mse',
                           thresh=2.5e-3, wts=None):
    # Get number of "visible" points that move in each example of the batch
    bsz = input.size(0) # batch size
    assert input.is_same_size(target), "Input and Target sizes need to match"
    if wts is not None:
        wts = wts.to(input)
    wtmask = wts if wts is not None else 1
    nummotionpts = ((motion.abs().sum(1) > thresh).type_as(wts) * wtmask).float().view(bsz, -1).sum(1).clamp(min=100) # Takes care of numerical instabilities & acts as margin
    # Compute loss
    weights = wts.expand_as(input).clone().view(bsz, -1) if wts is not None else 1 # Per-pixel scalar
    if loss_type == 'mse':
        diff = (input - target).view(bsz, -1) * weights
        loss = diff.pow(2).sum(1).div(2*nummotionpts).mean() # 0.5 * sum_i (err_i/N_i); err_i = sum_j (pred_ij - tar_ij)^2
    elif loss_type == 'abs':
        diff = (input - target).view(bsz, -1) * weights
        loss = diff.abs().sum(1).div(nummotionpts).mean()  # sum_i (err_i/N_i); err_i = sum_j |pred_ij - tar_ij|_1
    elif loss_type.find('normmsesqrt') >= 0:
        # This loss is a scale invariant version of the mse loss
        # Scales the squared error by a variance term based on the target magnitude
        # TODO: DO we need args for the scale & default sigma?
        if loss_type == 'normmsesqrtpt':
            norm  = target.pow(2).sum(1).sqrt()
            sigma = (0.5 * norm).clamp(min=2e-3).unsqueeze(1).expand_as(target).contiguous().view(bsz,-1)  # scale * ||y||_2, clamp for numerical stability (changed clamp to 2mm)
        else:
            sigma = (0.5 * target.abs()).clamp(min=2e-3).reshape(bsz,-1)  # scale * y, clamp for numerical stability (changed clamp to 2mm)
        diff  = (input - target).reshape(bsz, -1) * weights
        loss  = diff.pow(2).div(sigma).sum(1).div(2*nummotionpts).mean() # (x - y)^2 / 2*s^2 where s^2 = sigma, the variance
    else:
        assert False, "Unknown loss type: " + loss_type
    # Return
    return loss

####################################
### Basic Conv + Pool + BN + Non-linearity structure
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu',
                 coord_conv=False, **kwargs):
        super(BasicConv2D, self).__init__()
        if coord_conv:
            self.conv = CoordConv(in_channels, out_channels, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # Convolution -> Pool -> BN -> Non-linearity
    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)

### Basic Deconv + (Optional Skip-Add) + BN + Non-linearity structure
class BasicDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, nonlinearity='prelu',
                 coord_conv=False, **kwargs):
        super(BasicDeconv2D, self).__init__()
        if coord_conv:
            self.deconv = CoordConvT(in_channels, out_channels, **kwargs)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, x, y=None):
        if y is not None:
            x = self.deconv(x) + y  # Skip-Add the extra input
        else:
            x = self.deconv(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)

###  BN + Non-linearity + Basic Conv + Pool structure
class PreConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu',
                 coord_conv=False, **kwargs):
        super(PreConv2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)
        if coord_conv:
            self.conv = CoordConv(in_channels, out_channels, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None

    # BN -> Non-linearity -> Convolution -> Pool
    def forward(self, x):
        if self.bn:
            x = self.bn(x)
        x = self.nonlin(x)
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        return x

### Basic Deconv + (Optional Skip-Add) + BN + Non-linearity structure
class PreDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, nonlinearity='prelu',
                 coord_conv=False, **kwargs):
        super(PreDeconv2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)
        if coord_conv:
            self.deconv = CoordConvT(in_channels, out_channels, **kwargs)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)

    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, x, y=None):
        if self.bn:
            x = self.bn(x)
        x = self.nonlin(x)
        if y is not None:
            x = self.deconv(x) + y  # Skip-Add the extra input
        else:
            x = self.deconv(x)
        return x

###################
####### CoordConv from Uber: https://eng.uber.com/coordconv/
class CoordConvBase(nn.Module):
    def __init__(self):
        super(CoordConvBase, self).__init__()
        self.conv = None
        self.coord = None
        self.batch_size = None
        self.coord_batch = None

    def forward(self, inp):
        # Assumes input is B x C x H x W
        bsz, _, ht, wd = inp.size()
        if self.coord is None:
            ## Create the coordinates
            def trange(dim1, dim2):
                r = torch.arange(0, dim1) / (dim1 - 1)
                a = r.view(dim1,1).expand(dim1,dim2)
                return a
            # Setup coords from -1 to 1
            xrng = trange(wd, ht).transpose(0,1)[None, None, :, :] # 0 to 1 along cols
            yrng = trange(ht, wd)[None, None, :, :] # 0 to 1 along rows
            self.coord = (torch.cat([xrng, yrng], dim=1).type_as(inp) * 2) - 1 # -1 to 1

        # If the batch size changes we need to recompute some things. Then
        # we can save the batch size info + precomputed info.
        if self.batch_size is None or self.batch_size != bsz:
            self.batch_size = bsz
            self.coord_batch = self.coord.expand(bsz,2,ht,wd) # No gradients

        # Cat coords to input and apply the conv layer
        x = torch.cat([inp, self.coord_batch], dim=1) # Concat along channels dimension
        return self.conv(x)

class CoordConv(CoordConvBase):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_dim + 2, out_dim, **kwargs)

class CoordConvT(CoordConvBase):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(CoordConvT, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dim + 2, out_dim, **kwargs)

###################
### Basic Conv + Pool + BN + Non-linearity structure
class BasicLinear(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=True, nonlinearity='prelu'):
        super(BasicLinear, self).__init__()
        self.lin    = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # Linear -> BN -> Non-linearity
    def forward(self, x):
        x = self.lin(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)

### Apply weight-sharpening to the masks across the channels of the input
### output = Normalize( (sigmoid(input) + noise)^p + eps )
### where the noise is sampled from a 0-mean, sig-std dev distribution (sig is increased over time),
### the power "p" is also increased over time and the "Normalize" operation does a 1-norm normalization
def sharpen_masks(input, add_noise=True, noise_std=0, pow=1):
    input = torch.sigmoid(input)
    if (add_noise and noise_std > 0):
        noise = input.new_zeros(*input.size()).normal_(mean=0.0, std=noise_std) # Sample gaussian noise
        input = input + noise
    input = (torch.clamp(input, min=0, max=100000) ** pow) + 1e-12  # Clamp to non-negative values, raise to a power and add a constant
    return F.normalize(input, p=1, dim=1, eps=1e-12)  # Normalize across channels to sum to 1

#############################
### NEW SE3 LAYER (se3toSE3 with some additional transformation for the translation vector)

# Create a skew-symmetric matrix "S" of size [(Bk) x 3 x 3] (passed in) given a [(Bk) x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    output = vector.view(N,3,1).expand(N,3,3).clone()
    output[:, 0, 0] = 0
    output[:, 1, 1] = 0
    output[:, 2, 2] = 0
    output[:, 0, 1] = -vector[:, 2]
    output[:, 1, 0] =  vector[:, 2]
    output[:, 0, 2] =  vector[:, 1]
    output[:, 2, 0] = -vector[:, 1]
    output[:, 1, 2] = -vector[:, 0]
    output[:, 2, 1] =  vector[:, 0]
    return output

# Compute the rotation matrix R & translation vector from the axis-angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# (t = I + (1-cos(theta))/theta^2 * K + ((theta-sin(theta)/theta^3) * K^2
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# Eqns: 77-84 (PAGE: 10) of http://ethaneade.com/lie.pdf
def se3ToRt(input):
    # Check dimensions
    bsz, nse3, ndim = input.size()
    N = bsz*nse3
    eps = 1e-12
    assert (ndim == 6)

    # Trans | Rot params
    input_v = input.view(N, ndim, 1)
    rotparam   = input_v.narrow(1,3,3) # so(3)
    transparam = input_v.narrow(1,0,3) # R^3

    # Get the un-normalized axis and angle
    axis = rotparam.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle  = torch.sqrt(angle2)  # Angle
    small  = (angle2 < eps).detach() # Don't need gradient w.r.t this operation (also because of pytorch error: https://discuss.pytorch.org/t/get-error-message-maskedfill-cant-differentiate-the-mask/9129/4)

    # Create Identity matrix
    I = angle2.expand(N,3,3).clone()
    I[:] = 0 # Fill will all zeros
    I[:,0,0], I[:,1,1], I[:,2,2] = 1.0, 1.0, 1.0 # Diagonal elements = 1

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute A = (sin(theta)/theta)
    A = torch.sin(angle) / angle
    A[small] = 1.0 # sin(0)/0 ~= 1

    # Compute B = (1 - cos(theta)/theta^2)
    B = (1 - torch.cos(angle)) / angle2
    B[small] = 1/2 # lim 0-> 0 (1 - cos(0))/0^2 = 1/2

    # Compute C = (theta - sin(theta))/theta^3
    C = (1 - A) / angle2
    C[small] = 1/6 # lim 0-> 0 (0 - sin(0))/0^3 = 1/6

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    R = I + K * A.expand(N, 3, 3) + K2 * B.expand(N, 3, 3)

    # Compute the translation tfm matrix: V = I + ((1-cos(theta))/theta^2) * K + ((theta-sin(theta))/theta^3)*K^2
    V = I + K * B.expand(N, 3, 3) + K2 * C.expand(N, 3, 3)

    # Compute translation vector
    t = torch.bmm(V, transparam) # V*u

    # Final tfm
    return torch.cat([R, t], 2).view(bsz,nse3,3,4).clone() # B x K x 3 x 4


################################################################################
'''
    Single step models
'''

####################################
### Pose Encoder
# Model that takes in "depth/point cloud" to generate "k" poses represented as [R|t]
class PoseEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu', init_se3_iden=False,
                 wide=False, use_jt_angles=False, num_state=7, full_res=False):
        super(PoseEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('[PoseEncoder] Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType   = PreConv2D if pre_conv else BasicConv2D
        DeconvType = PreDeconv2D if pre_conv else BasicDeconv2D

        ###### Encoder
        # Create conv-encoder (large net => 5/6 conv layers with pooling)
        chn = [32, 64, 128, 256, 256, 256] if wide else [8, 16, 32, 64, 128, 128] # Num channels
        self.conv1 = ConvType(input_channels, chn[0], kernel_size=9, stride=1, padding=4,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 240x320 -> 120x160
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv5 = ConvType(chn[3], chn[4], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
        if full_res:
            self.conv6 = ConvType(chn[4], chn[5], kernel_size=3, stride=1, padding=1,
                                  use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
            self.celem = chn[5]*7*10
        else:
            self.conv6 = None
            self.celem = chn[4]*7*10

        ###### Encode jt angles
        self.use_jt_angles = use_jt_angles
        jdim = 0
        if self.use_jt_angles:
            jdim = 256
            self.jtencoder = nn.Sequential(
                nn.Linear(num_state, 128),
                get_nonlinearity(nonlinearity),
                nn.Linear(128, 256),
                get_nonlinearity(nonlinearity),
            )

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3
        sdim = 256 if wide else 128
        self.se3decoder  = nn.Sequential(
                                nn.Linear(self.celem + jdim, sdim),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(sdim, self.num_se3 * self.se3_dim) # Predict the SE3s from the conv-output
                           )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the pose-mask model to predict identity transform")
            layer = self.se3decoder[2]  # Get final SE3 prediction module
            init_se3layer_identity(layer, num_se3, se3_type)  # Init to identity

        # Create pose decoder (convert to r/t)
        self.se3_type = se3_type
        self.posedecoder = nn.Sequential()
        #if se3_type != 'se3aar':
        self.posedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, use_pivot)) # Convert to Rt
        if use_pivot:
            self.posedecoder.add_module('pivotrt', se3nn.CollapseRtPivots()) # Collapse pivots

    def forward(self, z):
        if self.use_jt_angles:
            x,j = z # Pts, Jt angles
        else:
            x = z

        # Run conv-encoder to generate embedding
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if self.conv6 is not None:
            x = self.conv6(x)

        # Run jt-encoder & concatenate the embeddings
        if self.use_jt_angles:
            j = self.jtencoder(j)
            p = torch.cat([x.view(-1, self.celem), j], 1)
        else:
            p = x.view(-1, self.celem)

        # Run pose-decoder to predict poses
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Return poses
        return p

####################################
### Mask Encoder (single encoder that takes a depth image and predicts segmentation masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks
class MaskEncoder(nn.Module):
    def __init__(self, num_se3, pre_conv=False, input_channels=3, use_bn=True, nonlinearity='prelu',
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 wide=False, full_res=False, noise_stop_iter=1e6):
        super(MaskEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType = PreConv2D if pre_conv else BasicConv2D
        DeconvType = PreDeconv2D if pre_conv else BasicDeconv2D

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        chn = [32, 64, 128, 256, 256, 256] if wide else [8, 16, 32, 64, 128, 128]  # Num channels
        self.conv1 = ConvType(input_channels, chn[0], kernel_size=9, stride=1, padding=4,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 240x320 -> 120x160
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv5 = ConvType(chn[3], chn[4], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10

        ###### Switch between full-res vs not
        self.full_res = full_res
        if self.full_res:
            ###### Conv 6
            self.conv6 = ConvType(chn[4], chn[5], kernel_size=3, stride=1, padding=1,
                                  use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10

            ###### Mask Decoder
            # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
            self.conv1x1 = ConvType(chn[5], chn[5], kernel_size=1, stride=1, padding=0,
                                    use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
            self.deconv1 = DeconvType(chn[5], chn[4], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
            self.deconv2 = DeconvType(chn[4], chn[3], kernel_size=4, stride=2, padding=1,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
            self.deconv3 = DeconvType(chn[3], chn[2], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
            self.deconv4 = DeconvType(chn[2], chn[1], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
            self.deconv5 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 120x160 -> 240x320
            if pre_conv:
                # Can fit an extra BN + Non-linearity
                self.deconv6 = DeconvType(chn[0], num_se3, kernel_size=8, stride=2, padding=3,
                                          use_bn=use_bn, nonlinearity=nonlinearity)  # 8x8, 240x320 -> 480x640
            else:
                self.deconv6 = nn.ConvTranspose2d(chn[0], num_se3, kernel_size=8, stride=2,
                                                  padding=3)  # 8x8, 120x160 -> 240x320
        else:
            ###### Mask Decoder
            # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
            self.conv1x1 = ConvType(chn[4], chn[4], kernel_size=1, stride=1, padding=0,
                                    use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
            self.deconv1 = DeconvType(chn[4], chn[3], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
            self.deconv2 = DeconvType(chn[3], chn[2], kernel_size=4, stride=2, padding=1,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
            self.deconv3 = DeconvType(chn[2], chn[1], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
            self.deconv4 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
            if pre_conv:
                # Can fit an extra BN + Non-linearity
                self.deconv5 = DeconvType(chn[0], num_se3, kernel_size=8, stride=2, padding=3,
                                          use_bn=use_bn, nonlinearity=nonlinearity)  # 8x8, 120x160 -> 240x320
            else:
                self.deconv5 = nn.ConvTranspose2d(chn[0], num_se3, kernel_size=8, stride=2, padding=3)  # 8x8, 120x160 -> 240x320

        # Normalize to generate mask (wt-sharpening vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter  # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate  # Rate for sharpening
            self.maskdecoder = sharpen_masks  # Use the weight-sharpener
            self.noise_stop_iter = noise_stop_iter # Stop adding noise after this many iters
        else:
            self.maskdecoder = nn.Softmax2d()  # SoftMax normalization

    def compute_wt_sharpening_stats(self, train_iter=0):
        citer = 1 + (train_iter - self.sharpen_start_iter)
        noise_std, pow = 0, 1
        if (citer > 0):
            noise_std = min((citer / 125000.0) * self.sharpen_rate,
                            0.1)  # Should be 0.1 by ~12500 iters from start (if rate=1)
            if hasattr(self, 'noise_stop_iter') and train_iter > self.noise_stop_iter:
                noise_std = 0
            pow = min(1 + (citer / 500.0) * self.sharpen_rate,
                      100)  # Should be 26 by ~12500 iters from start (if rate=1)
        return noise_std, pow

    def forward(self, x, train_iter=0):
        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        ### Full-res vs Not
        if self.full_res:
            # Conv-6
            c6 = self.conv6(c5)

            # Run mask-decoder to predict a smooth mask
            m = self.conv1x1(c6)
            m = self.deconv1(m, c5)
            m = self.deconv2(m, c4)
            m = self.deconv3(m, c3)
            m = self.deconv4(m, c2)
            m = self.deconv5(m, c1)
            m = self.deconv6(m)
        else:
            # Run mask-decoder to predict a smooth mask
            m = self.conv1x1(c5)
            m = self.deconv1(m, c4)
            m = self.deconv2(m, c3)
            m = self.deconv3(m, c2)
            m = self.deconv4(m, c1)
            m = self.deconv5(m)

        # Predict a mask (either wt-sharpening or sigmoid-mask or soft-mask approach)
        # Normalize to sum across 1 along the channels (only for weight sharpening or soft-mask)
        if self.use_wt_sharpening:
            noise_std, pow = self.compute_wt_sharpening_stats(train_iter=train_iter)
            m = self.maskdecoder(m, add_noise=self.training, noise_std=noise_std, pow=pow)
        else:
            m = self.maskdecoder(m)

        # Return masks
        return m

####################################
### Pose-Mask Encoder (single encoder that predicts both poses and masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks and "k" poses represented as [R|t]
# NOTE: We can set a conditional flag that makes it predict both poses/masks or just poses
class PoseMaskEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu', init_se3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                wide=False, use_jt_angles=False, num_state=7,
                 full_res=False, noise_stop_iter=1e6):
        super(PoseMaskEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType   = PreConv2D if pre_conv else BasicConv2D
        DeconvType = PreDeconv2D if pre_conv else BasicDeconv2D

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        chn = [32, 64, 128, 256, 256, 256] if wide else [8, 16, 32, 64, 128, 128]  # Num channels
        self.conv1 = ConvType(input_channels, chn[0], kernel_size=9, stride=1, padding=4,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 240x320 -> 120x160
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv5 = ConvType(chn[3], chn[4], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
        self.celem = chn[4] * 7 * 10

        ###### Switch between full-res vs not
        self.full_res = full_res
        if self.full_res:
            ###### Conv 6
            self.conv6 = ConvType(chn[4], chn[5], kernel_size=3, stride=1, padding=1,
                                  use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
            self.celem = chn[5] * 7 * 10

            ###### Mask Decoder
            # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
            self.conv1x1 = ConvType(chn[5], chn[5], kernel_size=1, stride=1, padding=0,
                                    use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
            self.deconv1 = DeconvType(chn[5], chn[4], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
            self.deconv2 = DeconvType(chn[4], chn[3], kernel_size=4, stride=2, padding=1,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
            self.deconv3 = DeconvType(chn[3], chn[2], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
            self.deconv4 = DeconvType(chn[2], chn[1], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
            self.deconv5 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 120x160 -> 240x320
            if pre_conv:
                # Can fit an extra BN + Non-linearity
                self.deconv6 = DeconvType(chn[0], num_se3, kernel_size=8, stride=2, padding=3,
                                          use_bn=use_bn, nonlinearity=nonlinearity)  # 8x8, 240x320 -> 480x640
            else:
                self.deconv6 = nn.ConvTranspose2d(chn[0], num_se3, kernel_size=8, stride=2,
                                                  padding=3)  # 8x8, 120x160 -> 240x320
        else:
            ###### Mask Decoder
            # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
            self.conv1x1 = ConvType(chn[4], chn[4], kernel_size=1, stride=1, padding=0,
                                    use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
            self.deconv1 = DeconvType(chn[4], chn[3], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
            self.deconv2 = DeconvType(chn[3], chn[2], kernel_size=4, stride=2, padding=1,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
            self.deconv3 = DeconvType(chn[2], chn[1], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
            self.deconv4 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
            if pre_conv:
                # Can fit an extra BN + Non-linearity
                self.deconv5 = DeconvType(chn[0], num_se3, kernel_size=8, stride=2, padding=3,
                                          use_bn=use_bn, nonlinearity=nonlinearity)  # 8x8, 120x160 -> 240x320
            else:
                self.deconv5 = nn.ConvTranspose2d(chn[0], num_se3, kernel_size=8, stride=2,
                                                  padding=3)  # 8x8, 120x160 -> 240x320

        # Normalize to generate mask (wt-sharpening vs sigmoid-mask vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate # Rate for sharpening
            self.maskdecoder = sharpen_masks # Use the weight-sharpener
            self.noise_stop_iter = noise_stop_iter
        else:
            self.maskdecoder = nn.Softmax2d() # SoftMax normalization

        ###### Encode jt angles
        self.use_jt_angles = use_jt_angles
        jdim = 0
        if self.use_jt_angles:
            jdim = 256
            self.jtencoder = nn.Sequential(
                nn.Linear(num_state, 128),
                get_nonlinearity(nonlinearity),
                nn.Linear(128, 256),
                get_nonlinearity(nonlinearity),
            )

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3
        sdim = 256 if wide else 128 # NOTE: This was 64 before! #128
        self.se3decoder  = nn.Sequential(
                                nn.Linear(self.celem + jdim, sdim),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(sdim, self.num_se3 * self.se3_dim)  # Predict the SE3s from the conv-output
                           )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the pose-mask model to predict identity transform")
            layer = self.se3decoder[2]  # Get final SE3 prediction module
            init_se3layer_identity(layer, num_se3, se3_type)  # Init to identity

        # Create pose decoder (convert to r/t)
        self.se3_type = se3_type
        def func(x):
            x = x.view(-1, self.se3_dim)
            return utils_torch.exp_se3(x).view(-1, self.num_se3, 4, 4)
        self.posedecoder = func
        # self.posedecoder = nn.Sequential()
        # #if se3_type != 'se3aar':
        # self.posedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, use_pivot)) # Convert to Rt
        # if use_pivot:
        #     self.posedecoder.add_module('pivotrt', se3nn.CollapseRtPivots()) # Collapse pivots

    def compute_wt_sharpening_stats(self, train_iter=0):
        citer = 1 + (train_iter - self.sharpen_start_iter)
        noise_std, pow = 0, 1
        if (citer > 0):
            noise_std = min((citer/125000.0) * self.sharpen_rate, 0.1) # Should be 0.1 by ~12500 iters from start (if rate=1)
            if hasattr(self, 'noise_stop_iter') and train_iter > self.noise_stop_iter:
                noise_std = 0
            pow = min(1 + (citer/500.0) * self.sharpen_rate, 100) # Should be 26 by ~12500 iters from start (if rate=1)
        return noise_std, pow

    def forward(self, z, predict_masks=True, train_iter=0):
        if self.use_jt_angles or type(z) == list:
            x,j = z # Pts, Jt angles
        else:
            x = z

        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        if self.full_res:
            c6 = self.conv6(c5)
            cout = c6
        else:
            cout = c5

        # Run jt-encoder & concatenate the embeddings
        if self.use_jt_angles:
            j = self.jtencoder(j)
            p = torch.cat([cout.view(-1, self.celem), j], 1)
        else:
            p = cout.view(-1, self.celem)

        # Run pose-decoder to predict poses
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        self.se3output = p.clone() # SE3 prediction
        #if self.se3_type == 'se3aar':
        #    p = se3ToRt(p) # Use new se3ToRt layer!
        p = self.posedecoder(p)

        # Run mask-decoder to predict a smooth mask
        # NOTE: Conditional based on input flag
        if predict_masks:
            ### Full-res vs Not
            if self.full_res:
                # Run mask-decoder to predict a smooth mask
                m = self.conv1x1(c6)
                m = self.deconv1(m, c5)
                m = self.deconv2(m, c4)
                m = self.deconv3(m, c3)
                m = self.deconv4(m, c2)
                m = self.deconv5(m, c1)
                m = self.deconv6(m)
            else:
                # Run mask-decoder to predict a smooth mask
                m = self.conv1x1(c5)
                m = self.deconv1(m, c4)
                m = self.deconv2(m, c3)
                m = self.deconv3(m, c2)
                m = self.deconv4(m, c1)
                m = self.deconv5(m)

            # Predict a mask (either wt-sharpening or sigmoid-mask or soft-mask approach)
            # Normalize to sum across 1 along the channels (only for weight sharpening or soft-mask)
            if self.use_wt_sharpening:
                noise_std, pow = self.compute_wt_sharpening_stats(train_iter=train_iter)
                m = self.maskdecoder(m, add_noise=self.training, noise_std=noise_std, pow=pow)
            else:
                m = self.maskdecoder(m)

            # Return both
            return p, m
        else:
            return p


####################################
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class TransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, delta_pivot='', se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False, use_jt_angles=False, num_state=7):
        super(TransitionModel, self).__init__()
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=(delta_pivot == 'pred')) # Only if we are predicting directly
        self.num_se3 = num_se3

        # Pose encoder
        pdim = [128, 256]
        self.poseencoder = nn.Sequential(
                                nn.Linear(self.num_se3 * 16, pdim[0]),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(pdim[0], pdim[1]),
                                get_nonlinearity(nonlinearity)
                            )

        # Control encoder
        cdim = [128, 256] #[64, 128]
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, cdim[0]),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(cdim[0], cdim[1]),
                                get_nonlinearity(nonlinearity)
                            )

        # Jt angle encoder
        jdim = [0, 0]
        self.use_jt_angles = use_jt_angles
        if use_jt_angles:
            jdim = [128, 256]  # [64, 128]
            self.jtangleencoder = nn.Sequential(
                nn.Linear(num_state, jdim[0]),
                get_nonlinearity(nonlinearity),
                nn.Linear(jdim[0], jdim[1]),
                get_nonlinearity(nonlinearity)
            )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            nn.Linear(pdim[1]+cdim[1]+jdim[1], 256),
            get_nonlinearity(nonlinearity),
            nn.Linear(256, 128),
            get_nonlinearity(nonlinearity),
            nn.Linear(128, self.num_se3 * self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            layer = self.deltase3decoder[4]  # Get final SE3 prediction module
            init_se3layer_identity(layer, num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        self.se3_type    = se3_type
        self.delta_pivot = delta_pivot
        self.inp_pivot   = (self.delta_pivot != '') and (self.delta_pivot != 'pred') # Only for these 2 cases, no pivot is passed in as input
        def func(x):
            x = x.view(-1, self.se3_dim)
            return utils_torch.exp_se3(x).view(-1, self.num_se3, 4, 4)
        self.deltaposedecoder = func
        # self.deltaposedecoder = nn.Sequential()
        # self.deltaposedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, (self.delta_pivot != '')))  # Convert to Rt
        # if (self.delta_pivot != ''):
        #     self.deltaposedecoder.add_module('pivotrt', se3nn.CollapseRtPivots())  # Collapse pivots

        # Compose deltas with prev pose to get next pose
        # It predicts the delta transform of the object between time "t1" and "t2": p_t2_to_t1: takes a point in t2 and converts it to t1's frame of reference
	    # So, the full transform is: p_t2 = p_t1 * p_t2_to_t1 (or: p_t2 (t2 to cam) = p_t1 (t1 to cam) * p_t2_to_t1 (t2 to t1))
        # self.posedecoder = se3nn.ComposeRtPair()

    def forward(self, x):
        # Run the forward pass
        if self.use_jt_angles:
            if self.inp_pivot:
                p, j, c, pivot = x # Pose, Jtangles, Control, Pivot
            else:
                p, j, c = x # Pose, Jtangles, Control
        else:
            if self.inp_pivot:
                p, c, pivot = x # Pose, Control, Pivot
            else:
                p, c = x # Pose, Control
        pv = p.view(-1, self.num_se3*16) # Reshape pose
        pe = self.poseencoder(pv)    # Encode pose
        ce = self.ctrlencoder(c)     # Encode ctrl
        if self.use_jt_angles:
            je = self.jtangleencoder(j)  # Encode jt angles
            x = torch.cat([pe,je,ce], 1) # Concatenate encoded vectors
        else:
            x = torch.cat([pe,ce], 1)    # Concatenate encoded vectors
        x = self.deltase3decoder(x)  # Predict delta-SE3
        x = x.view(-1, self.num_se3, self.se3_dim)
        if self.inp_pivot: # For these two cases, we don't need to handle anything
            x = torch.cat([x, pivot.view(-1, self.num_se3, 3)], 2) # Use externally provided pivots
        x = self.deltaposedecoder(x)  # Convert delta-SE3 to delta-Pose (can be in local or global frame of reference)
        # Predicted delta is already in the global frame of reference, use it directly (from global to global)
        # z = self.posedecoder(x, p) # Compose predicted delta & input pose to get next pose (SE3_2 = SE3_2 * SE3_1^-1 * SE3_1)
        z = torch.einsum('bnij, bnjk -> bnik', x, p)
        y = x # D = SE3_2 * SE3_1^-1 (global to global)

        # Return
        return [y, z] # Return both the deltas (in global frame) and the composed next pose


####################################
### Multi-step version of the SE3-Pose-Model
### Currently, this has a separate pose & mask predictor as well as a transition model
### NOTE: The forward pass is not currently implemented, this needs to be done outside
class MultiStepSE3PoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False, delta_pivot='',
                 input_channels=3, use_bn=True, pre_conv=False, decomp_model=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7,
                 full_res=False, noise_stop_iter=1e6, trans_type='default',
                 posemask_type='default'):
        super(MultiStepSE3PoseModel, self).__init__()

        # Initialize the pose & mask model
        self.decomp_model = decomp_model
        if self.decomp_model:
            print('Using separate networks for pose and mask prediction')
            self.posemodel = PoseEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                         input_channels=input_channels,
                                         init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                         nonlinearity=nonlinearity, wide=wide,
                                         use_jt_angles=use_jt_angles, num_state=num_state,
                                         full_res=full_res)
            self.maskmodel = MaskEncoder(num_se3=num_se3, input_channels=input_channels,
                                         use_bn=use_bn, pre_conv=pre_conv,
                                         nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                         sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                         wide=wide,
                                         full_res=full_res, noise_stop_iter=noise_stop_iter)
        else:
            self.posemaskmodel = PoseMaskEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                            input_channels=input_channels,
                                            init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                            nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                            sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                            wide=wide,
                                            use_jt_angles=use_jt_angles, num_state=num_state,
                                            full_res=full_res, noise_stop_iter=noise_stop_iter)

        # Initialize the transition model
        self.transitionmodel = TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, delta_pivot=delta_pivot,
                                       se3_type=se3_type,
                                       nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden,
                                       local_delta_se3=local_delta_se3,
                                       use_jt_angles=use_jt_angles_trans, num_state=num_state)

        # Options
        self.use_jt_angles = use_jt_angles
        self.use_jt_angles_trans = use_jt_angles_trans

    # Predict pose only
    def forward_only_pose(self, x):
        ptcloud, jtangles = x
        inp = [ptcloud, jtangles] if self.use_jt_angles else ptcloud
        if self.decomp_model:
            return self.posemodel(inp)
        else:
            return self.posemaskmodel(inp, predict_masks=False) # returns only pose

    # Predict both pose and mask
    def forward_pose_mask(self, x, train_iter=0):
        ptcloud, jtangles = x
        inp = [ptcloud, jtangles] if self.use_jt_angles else ptcloud
        if self.decomp_model:
            pose = self.posemodel(inp)
            mask = self.maskmodel(ptcloud, train_iter=train_iter)
            return pose, mask
        else:
            return self.posemaskmodel(inp, train_iter=train_iter, predict_masks=True) # Predict both

    # Predict next pose based on current pose and control
    def forward_next_pose(self, pose, ctrl, jtangles=None, pivots=None):
        inp = [pose, jtangles, ctrl] if self.use_jt_angles_trans else [pose, ctrl]
        if self.transitionmodel.inp_pivot:
            inp.append(pivots)
        return self.transitionmodel(inp)

    # Forward pass through the model
    def forward(self, x):
        print('Forward pass for Multi-Step SE3-Pose-Model is not yet implemented')
        raise NotImplementedError