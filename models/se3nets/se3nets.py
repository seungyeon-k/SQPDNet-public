import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import models.se3nets.se3layers as se3nn
from models.se3nets import ctrlnets
from functions import utils_torch
################################################################################
'''
    Single step / Recurrent models
'''
####################################
### Encoder
# Model that takes in "depth/point cloud" & "controls" to generate an encoded state
class Encoder(nn.Module):
    def __init__(self, num_ctrl, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu',
                 wide=False, num_state=0,
                 use_jt_angles=False):
        super(Encoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('[PoseEncoder] Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType   = ctrlnets.PreConv2D if pre_conv else ctrlnets.BasicConv2D
        DeconvType = ctrlnets.PreDeconv2D if pre_conv else ctrlnets.BasicDeconv2D

        ###### Img encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        self.chn = [32, 64, 128, 256, 256] if wide else [8, 16, 32, 64, 128] # Num channels
        self.conv1 = ConvType(input_channels, self.chn[0], kernel_size=9, stride=1, padding=4,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 240x320 -> 120x160
        self.conv2 = ConvType(self.chn[0], self.chn[1], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv3 = ConvType(self.chn[1], self.chn[2], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv4 = ConvType(self.chn[2], self.chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv5 = ConvType(self.chn[3], self.chn[4], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
        self.celem = self.chn[4]*7*10

        ###### Ctrl encoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.ctn = [128,256] if wide else [64, 128]
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, self.ctn[0]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(self.ctn[0], self.ctn[1]) # Predict the SE3s from the conv-output
                           )

        ###### Jt angle encoder
        self.use_jt_angles = use_jt_angles
        jdim = 0
        if self.use_jt_angles:
            jdim = 256
            self.jtangleencoder = nn.Sequential(
                nn.Linear(num_state, 128),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(128, 256),
                ctrlnets.get_nonlinearity(nonlinearity),
            )

        ###### Hidden layer
        self.combined1 = nn.Sequential(
            nn.Linear(self.celem + self.ctn[1] + jdim, 256),
            ctrlnets.get_nonlinearity(nonlinearity)
        )
        self.hiddenlayer = nn.Linear(256, 256)
        self.combined2 = nn.Sequential(
            nn.Linear(256, self.celem),
            ctrlnets.get_nonlinearity(nonlinearity),
        )

        # ###### Joint encoder
        # self.jtencoder = nn.Sequential(
        #                         nn.Linear(self.celem + self.ctn[1], 256),
        #                         ctrlnets.get_nonlinearity(nonlinearity),
        #                         nn.Linear(256, self.celem),
        #                         ctrlnets.get_nonlinearity(nonlinearity),
        #                    )

    def forward(self, x):
        # Run conv-encoder to generate embedding
        p, j, c = x # pts, ctrl

        c1 = self.conv1(p)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        pe = c5.view(-1, self.celem)

        # Run ctrl-encoder
        ce = self.ctrlencoder(c)

        # Run jtangle-encoder & combine encoding
        if self.use_jt_angles:
            je = self.jtangleencoder(j)  # Encode jt angles
            e = torch.cat([pe, je, ce], 1)  # Concatenate encoded vectors
        else:
            e = torch.cat([pe, ce], 1)  # Concatenate encoded vectors

        # Use the hidden layer network
        h1 = self.combined1(e)
        ls = self.hiddenlayer(h1)
        h2 = self.combined2(ls)

        # Return jt encoder output
        return h2.view(-1,self.chn[4],7,10), [c1,c2,c3,c4,c5]

####################################
### Mask Encoder (single encoder that takes a depth image and predicts segmentation masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks
class MaskDecoder(nn.Module):
    def __init__(self, num_se3, pre_conv=False, use_bn=True, nonlinearity='prelu',
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1, wide=False):
        super(MaskDecoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType = ctrlnets.PreConv2D if pre_conv else ctrlnets.BasicConv2D
        DeconvType = ctrlnets.PreDeconv2D if pre_conv else ctrlnets.BasicDeconv2D

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        chn = [32, 64, 128, 256, 256] if wide else [8, 16, 32, 64, 128] # Num channels
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
            self.maskdecoder = ctrlnets.sharpen_masks  # Use the weight-sharpener
        else:
            self.maskdecoder = nn.Softmax2d()  # SoftMax normalization

    def compute_wt_sharpening_stats(self, train_iter=0):
        citer = 1 + (train_iter - self.sharpen_start_iter)
        noise_std, pow = 0, 1
        if (citer > 0):
            noise_std = min((citer / 125000.0) * self.sharpen_rate,
                            0.1)  # Should be 0.1 by ~12500 iters from start (if rate=1)
            pow = min(1 + (citer / 500.0) * self.sharpen_rate,
                      100)  # Should be 26 by ~12500 iters from start (if rate=1)
        return noise_std, pow

    def forward(self, x, train_iter=0):
        # Run mask-decoder to predict a smooth mask
        p, [c1,c2,c3,c4,c5] = x
        m = self.conv1x1(p)
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
### Delta-Decoder model (predicts change in poses based on the applied control)
# Takes in state_t and generates delta pose between t & t+1
class DeltaSE3Decoder(nn.Module):
    def __init__(self, num_se3, input_dim, use_pivot=False, se3_type='se3aa',
                 nonlinearity='prelu', init_se3_iden=False):
        super(DeltaSE3Decoder, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3

        # SE3 decoder
        self.input_dim = input_dim
        self.se3decoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            ctrlnets.get_nonlinearity(nonlinearity),
            nn.Linear(256, 128),
            ctrlnets.get_nonlinearity(nonlinearity),
            nn.Linear(128, self.num_se3 * self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            layer = self.se3decoder[4]  # Get final SE3 prediction module
            ctrlnets.init_se3layer_identity(layer, num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        if se3_type == 'se3aa':
            def func(x):
                x = x.view(-1, self.se3_dim)
                R = utils_torch.exp_so3(x[:, :3]).view(-1, self.num_se3, 3, 3)
                p = x[:, 3:].view(-1, self.num_se3, 3, 1)
                T = torch.cat([R, p], dim=-1)
                append_array = torch.tensor([0, 0, 0, 1]).view(1, 4).unsqueeze(0).unsqueeze(0).to(x).repeat(len(T), self.num_se3, 1, 1)
                return torch.cat([T, append_array], dim=2)
            self.posedecoder = func
        elif se3_type == 'affine':
            def func(x):
                x = x.view(len(x), self.num_se3, 3, 4)
                append_array = torch.tensor([0, 0, 0, 1]).view(1, 4).unsqueeze(0).unsqueeze(0).to(x).repeat(len(x), self.num_se3, 1, 1)
                return torch.cat([x, append_array], dim=2)
            self.posedecoder = func
        # self.posedecoder = nn.Sequential()
        # self.posedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, use_pivot))  # Convert to Rt
        # if use_pivot:
        #     self.posedecoder.add_module('pivotrt', se3nn.CollapseRtPivots())  # Collapse pivots

    def forward(self, x):
        # Run the forward pass
        s = x.view(-1, self.input_dim) # Reshape state
        x = self.se3decoder(s)  # Predict delta-SE3
        x = x.view(-1, self.num_se3, self.se3_dim)
        x = self.posedecoder(x)  # Convert delta-SE3 to delta-Pose (can be in local or global frame of reference)

        # Return
        return x # Return both the deltas (in global frame) and the composed next pose

####################################
### SE3-Model (single-step model that takes [depth_t, ctrl-t] to predict
### [delta-pose, mask_t]
class SE3Model(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1, wide=False,
                 num_state=0, use_jt_angles=False):
        super(SE3Model, self).__init__()

        # Initialize the pose-mask model
        self.encoder     = Encoder(num_ctrl=num_ctrl, input_channels=input_channels, use_bn=use_bn, pre_conv=pre_conv,
                                   nonlinearity=nonlinearity, wide=wide, num_state=num_state,
                                   use_jt_angles=use_jt_angles)
        self.maskdecoder = MaskDecoder(num_se3=num_se3, pre_conv=pre_conv, use_bn=use_bn, nonlinearity=nonlinearity,
                                       use_wt_sharpening=use_wt_sharpening, sharpen_start_iter=sharpen_start_iter,
                                       sharpen_rate=sharpen_rate, wide=wide)
        self.deltase3decoder  = DeltaSE3Decoder(input_dim=self.encoder.celem, num_se3=num_se3, use_pivot=use_pivot,
                                                se3_type=se3_type,
                                                nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden)

        self.num_se3 = num_se3

    # Forward pass through the model
    def forward(self, x, reset_hidden_state=False, mask_2d_old=None, train_iter=0):
        # Get input vars
        input_1, jtangles_1, ctrl_1 = x

        # Get delta-pose & mask predictions
        state_1        = self.encoder([input_1, jtangles_1, ctrl_1])
        # if mask_2d_old is None:
        mask_1         = self.maskdecoder(state_1, train_iter=train_iter)
        # else:
        #     mask_2d_old = mask_2d_old.unsqueeze(1) # convert to Nx1xHxW
        #     one_hot = torch.zeros(mask_2d_old.size(0), self.num_se3, mask_2d_old.size(2), mask_2d_old.size(3)).to(mask_2d_old)
        #     mask_1 = one_hot.scatter_(1, mask_2d_old.to(torch.int64), 1).to(torch.float32) 
        deltapose_t_12 = self.deltase3decoder(state_1[0]) # (bs, num_se3, 4, 4)

        # Predict 3D points
        ptcloud_1 = input_1.narrow(1,0,3) # First 3 channels only (input can be xyzrgb or xyzhue)
        # predpts_1 = se3nn.NTfm3D()(ptcloud_1, mask_1, deltapose_t_12)
        def func(pc, mask, deltapose):
            batch_size, num_channels, data_height, data_width = pc.size()
            pc_ = torch.cat([
                pc,
                torch.ones(batch_size, 1, data_height, data_width).to(pc)
            ], dim=1)

            num_se3 = mask.size()[1]
            assert (num_channels == 3)
            assert (mask.size() == torch.Size([batch_size, num_se3, data_height, data_width]))
            assert (deltapose.size() == torch.Size([batch_size, num_se3, 4, 4])) 

            pc_next_ = torch.einsum('bnij, bnhw, bjhw -> bihw', deltapose, mask, pc_)

            # pc_next_ = torch.einsum('bnij, bjhw -> bnihw', deltapose, pc_)
            # pc_next_ = torch.einsum('bnihw, bnhw -> bihw', pc_next_, mask)

            pc_next = pc_next_[:, :3, :, :]
            return pc_next

        predpts_1 = func(ptcloud_1, mask_1, deltapose_t_12)
        flows_12  = predpts_1 - ptcloud_1 # Return flows

        # Return outputs
        return flows_12, [deltapose_t_12, mask_1]


def label_to_one_hot_label(labels, num_se3, device=f'cuda:0', eps=1e-6):
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_se3) + shape[1:], device=device)
    ret = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps    
    return ret