import torch
import torch.nn as nn
from models.se3nets import ctrlnets 

### Initialize the SE3 prediction layer to identity
def init_flowlayer_identity(layer):
    layer.weight.data.uniform_(-0.001, 0.001)  # Initialize weights to near identity
    layer.bias.data.uniform_(-0.01, 0.01)  # Initialize biases to near identity

################################################################################
'''
    Single step / Recurrent models
'''
####################################
### Encoder
# Model that takes in "depth/point cloud", "controls" & "jtangles" to generate an encoded state
class FlowNet(nn.Module):
    def __init__(self, num_ctrl, num_state=0, input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_flow_iden = False,
                 use_jt_angles=False):
        super(FlowNet, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('[Flow-Encoder] Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType   = ctrlnets.PreConv2D if pre_conv else ctrlnets.BasicConv2D
        DeconvType = ctrlnets.PreDeconv2D if pre_conv else ctrlnets.BasicDeconv2D

        ###### Img encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        self.chn = [32, 64, 128, 256, 256] # Num channels
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
        cdim = 256
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, 128),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(128, 256) # Get encoded state
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
            nn.Linear(self.celem + cdim + jdim, 256),
            ctrlnets.get_nonlinearity(nonlinearity)
        )
        self.hiddenlayer = nn.Linear(256, 256)
        self.combined2 = nn.Sequential(
            nn.Linear(256, self.celem),
            ctrlnets.get_nonlinearity(nonlinearity),
        )

        ###### Deconv-pipeline to predict output flow
        dchn = [32, 64, 128, 256, 256]  # Num channels
        #self.conv1x1 = ConvType(dchn[4], dchn[4], kernel_size=1, stride=1, padding=0,
        #                        use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
        self.deconv1 = DeconvType(dchn[4], dchn[3], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
        self.deconv2 = DeconvType(dchn[3], dchn[2], kernel_size=4, stride=2, padding=1,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
        self.deconv3 = DeconvType(dchn[2], dchn[1], kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
        self.deconv4 = DeconvType(dchn[1], dchn[0], kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
        if pre_conv:
            # Can fit an extra BN + Non-linearity
            self.deconv5 = DeconvType(dchn[0], 3, kernel_size=8, stride=2, padding=3,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 8x8, 120x160 -> 240x320
        else:
            self.deconv5 = nn.ConvTranspose2d(dchn[0], 3, kernel_size=8, stride=2,
                                              padding=3)  # 8x8, 120x160 -> 240x320

        # Init to identity
        if init_flow_iden:
            init_flowlayer_identity(self.deconv5)

    def forward(self, x):
        # Run the forward pass
        p, j, c = x  # Pose, Jtangles, Control

        # Run conv-encoder to generate embedding
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
            e = torch.cat([pe,je,ce], 1) # Concatenate encoded vectors
        else:
            e = torch.cat([pe,ce], 1)    # Concatenate encoded vectors

        # Use the hidden layer network
        h1 = self.combined1(e)
        ls = self.hiddenlayer(h1)
        h2 = self.combined2(ls)

        # Deconv net to produce flows
        di = h2.view(-1, self.chn[4], 7, 10) # View it properly
        m = self.deconv1(di, c4)
        m = self.deconv2(m, c3)
        m = self.deconv3(m, c2)
        m = self.deconv4(m, c1)
        flow = self.deconv5(m)

        # Return flow
        return flow