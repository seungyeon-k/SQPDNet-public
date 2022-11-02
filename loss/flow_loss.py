import torch
import torch.nn as nn
from models.se3nets.ctrlnets import MotionNormalizedLoss3D, Loss3D

def MSELoss(input, target):
    return torch.sum(((input-target)**2).reshape(len(input), -1), dim=1).mean()

class FlowLoss(nn.Module):
    def __init__(self, **kwargs):
        super(FlowLoss, self).__init__()
        self.loss_type = kwargs.get('type', 'motionnormalizedloss3d')
        if self.loss_type == 'motionnormalizedloss3d':
            self.loss_func = MotionNormalizedLoss3D
        elif self.loss_type == 'mse':
            self.loss_func = MSELoss

    def forward(self, input, target, wts=None):
        if self.loss_type == 'motionnormalizedloss3d':
            loss = self.loss_func(input, target, target, loss_type='normmsesqrt', wts=wts)
        elif self.loss_type == 'mse':
            loss = self.loss_func(input, target)
        return loss
