import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(s_act):
	if s_act == 'relu':
		return nn.ReLU(inplace=True)
	elif s_act == 'sigmoid':
		return nn.Sigmoid()
	elif s_act == 'softplus':
		return nn.Softplus()
	elif s_act == 'linear':
		return None
	elif s_act == 'tanh':
		return nn.Tanh()
	elif s_act == 'leakyrelu':
		return nn.LeakyReLU(0.2, inplace=True)
	elif s_act == 'softmax':
		return nn.Softmax(dim=1)
	else:
		raise ValueError(f'Unexpected activation: {s_act}')

class MotionPredictionNetwork(nn.Module):
	def __init__(self, backbone, **kwargs):
		super(MotionPredictionNetwork, self).__init__()

		self.args = kwargs
		self.motion_dim = kwargs['motion_dim']

		# action network
		self.net_action = MLP(input_dim= kwargs['action_dim'], **kwargs['action'])

		# ego feature network
		self.net_ego = MLP(input_dim= kwargs['dsq_dim'], **kwargs['ego'])

		# global scene feature network
		self.net_global_1 = PointwiseMLP(input_dim= kwargs['dsq_dim'], **kwargs['global_1'])
		self.net_global_2_input_dim = kwargs['global_1']['output_dim']
		self.net_global_2 = MLP(input_dim=self.net_global_2_input_dim, **kwargs['global_2'])

		# motion network
		self.net_motion_input_dim = (kwargs['action']['output_dim'] +  
									kwargs['ego']['output_dim'] + 
									kwargs['global_2']['output_dim'])
		if self.motion_dim == '2D':
			self.net_motion_position = MLP(input_dim=self.net_motion_input_dim, output_dim=2, **kwargs["motion_position"])
			self.net_motion_orientation = MLP(input_dim=self.net_motion_input_dim, output_dim=2, **kwargs["motion_orientation"])    			
		elif self.motion_dim == '3D':
			self.net_motion_position = MLP(input_dim=self.net_motion_input_dim, output_dim=3, **kwargs["motion_position"])
			self.net_motion_orientation = MLP(input_dim=self.net_motion_input_dim, output_dim=4, **kwargs["motion_orientation"])

	def forward(self, x, a):
		# find fake primitives whose confidences are 0
		mask_fake = x[:, 0, :] == 0

		# ego feature
		x_ego = self.net_ego(x[:, :, 0])

		# global scene feature
		x_global = self.net_global_1(x)
		
		# make fake primitives' feature to -inf
		x_global = x_global - torch.nan_to_num(torch.inf * mask_fake, 0, posinf=torch.inf).unsqueeze(1)
		x_global = F.adaptive_max_pool1d(x_global, 1).squeeze(-1)
		x_global = self.net_global_2(x_global)

		# action
		a = self.net_action(a)

		# concatenate feature vectors
		x_cat = torch.cat([x_ego, x_global, a], dim=1)

		# motion prediction
		x_pos = self.net_motion_position(x_cat)
		x_ori = self.net_motion_orientation(x_cat)
		x_ori = F.normalize(x_ori, p=2, dim=1)

		if self.motion_dim == '2D':
			x_ori = torch.atan2(x_ori[:, 0:1], x_ori[:, 1:2])

		return torch.cat([x_pos, x_ori], dim=1)

	def train_step(self, x, a, y, optimizer, loss_function, clip_grad=None, **kwargs):
		optimizer.zero_grad()

		pred = self(x, a)

		loss = loss_function(y, pred)

		if clip_grad is not None:
			torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
		loss.backward()

		optimizer.step()

		return {
			"loss": loss.item(),
			"scene": x.detach().cpu().numpy(),
			"action": a.detach().cpu().numpy(),
			"motion_pred": pred.detach().cpu().numpy(),
			"motion_gt": y.detach().cpu().numpy()
		}

	def validation_step(self, x, a, y, loss_function, **kwargs):
		pred = self(x, a)

		loss = loss_function(y, pred)
		loss.backward()

		return {
			"loss": loss.item(),
			"scene": x.detach().cpu().numpy(),
			"action": a.detach().cpu().numpy(),
			"motion_pred": pred.detach().cpu().numpy(),
			"motion_gt": y.detach().cpu().numpy()
		}


class MLP(nn.Module):
	def __init__(self, **args):
		super(MLP, self).__init__()
		self.l_hidden = args['l_hidden']
		self.output_dim = args['output_dim']
		self.input_dim = args['input_dim']
		l_neurons = self.l_hidden + [self.output_dim]
		
		l_layer = []
		prev_dim = self.input_dim
		for i, n_hidden in enumerate(l_neurons):
			l_layer.append(nn.Linear(prev_dim, n_hidden))
			if i < len(l_neurons) - 1:
				l_layer.append(nn.LeakyReLU(0.2))
			prev_dim = n_hidden

		self.net = nn.Sequential(*l_layer)

	def forward(self, x):
		x = self.net(x)

		return x

class PointwiseMLP(nn.Module):
	def __init__(self, **args):
		super(PointwiseMLP, self).__init__()
		self.l_hidden = args['l_hidden']
		self.output_dim = args['output_dim']
		self.input_dim = args['input_dim']
		l_neurons = self.l_hidden + [self.output_dim]
		
		l_layer = []
		prev_dim = self.input_dim
		for i, n_hidden in enumerate(l_neurons):
			l_layer.append(nn.Conv1d(prev_dim, 
									 n_hidden, 
									kernel_size=1, 
									bias=False)
						  )
			if i < len(l_neurons) - 1:
				l_layer.append(nn.LeakyReLU(0.2))
			prev_dim = n_hidden

		self.net = nn.Sequential(*l_layer)

	def forward(self, x):
		x = self.net(x)
		
		return x