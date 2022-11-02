import torch
from functions.utils_torch import quats_to_matrices_torch


class MotionPredictionLoss(torch.nn.Module):
	def __init__(self, device=None, **kargs):
		super(MotionPredictionLoss, self).__init__()
		self.device = device
		self.motion_dim = kargs['motion_dim']
		self.weight = kargs['weight']

	def forward(self, target, output):
		if self.motion_dim == '2D':
			# decompose   
			position_target = target[:, 0:2]
			theta_target = target[:, 2]
			position_output = output[:, 0:2]
			theta_output = output[:, 2]

			# position error
			position_error = torch.norm(position_target - position_output, dim=1) ** 2

			# orientation error
			orientation_error = (1 - torch.cos(theta_target - theta_output)) ** 2

		elif self.motion_dim == '3D':
			# decompose   
			position_target = target[:, 0:3]
			quaternion_target = target[:, 3:]
			position_output = output[:, 0:3]
			quaternion_output = output[:, 3:]

			# position error
			position_error = torch.norm(position_target - position_output, dim=1) ** 2
			
			# quaternions to rotation matrices
			matrices_target = quats_to_matrices_torch(quaternion_target)
			matrices_output_t = quats_to_matrices_torch(quaternion_output).permute(0, 2, 1)

			# rotation matrices error
			matrices_error = torch.eye(3).unsqueeze(0).to(matrices_target) - matrices_target @ matrices_output_t
			orientation_error = torch.norm(matrices_error, p='fro', dim=(1, 2)) ** 2

		else:
			raise ValueError('invalid space type for motion loss function')
			
		return torch.mean(position_error + self.weight * orientation_error)
