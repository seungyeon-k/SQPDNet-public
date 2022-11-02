import torch
import torch.nn as nn

from functions.utils_torch import quats_to_matrices_torch

class SuperquadricLoss(nn.Module):
    def __init__(self, **kargs):
        super(SuperquadricLoss, self).__init__()

    def forward(self, output, y):
        """
        Args:
            output (b x 16): deformable superquadric parameters.
                             (network output)
            y (b x 3 x n): ground-truth point cloud.

        Returns:
            loss (float): Gross and Boult loss
        """

        # network output processing
        position = output[:, :3]
        orientation = output[:, 3:7]
        rotation = quats_to_matrices_torch(orientation)
        parameters = output[:, 7:]

        # ground-truth point cloud
        y_position = y[:, :3, :]
        rotation_t = rotation.permute(0,2,1)
        y_transformed = - rotation_t @ position.unsqueeze(2) + rotation_t @ y_position

        # Gross and Boult superquadric loss
        loss = torch.mean(self.sq_distance(y_transformed, parameters)**2)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError('The loss function is inf or nan.')

        return loss

    def sq_distance(self, pts_gt, parameters):
        # parameter decomposition
        a1 = parameters[:, 0:1]
        a2 = parameters[:, 1:2]
        a3 = parameters[:, 2:3]
        e1 = parameters[:, 3:4]
        e2 = parameters[:, 4:5]

        # epsilon for numerical stability
        eps = 1e-10

        # inverse deformation
        X = pts_gt[:, 0, :]
        Y = pts_gt[:, 1, :]
        Z = pts_gt[:, 2, :]

        beta = (
            (
            torch.abs(X/a1)**(2/e2)
            + torch.abs(Y/a2)**(2/e2)
            )**(e2/e1) + torch.abs(Z/a3)**(2/e1)
        + eps)**(-e1/2)

        F = torch.norm(pts_gt, dim=1) * (1 - beta)

        return F   