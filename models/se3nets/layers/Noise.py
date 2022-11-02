import torch
from torch.autograd import Function
from torch.nn import Module

'''
From: https://github.com/willwhitney/understanding-visual-concepts/blob/master/Noise.lua
Adds gaussian noise to the inputs during training, leaves inputs unchanged during testing
Gradients are passed back as is
'''

## FWD/BWD pass function
class NoiseFunction(Function):
	def __init__(self, max_std, slope_std, iter_count, start_iter, training):
		assert (max_std >= 0 and slope_std >= 0 and start_iter >= 0);
		assert (iter_count.nelement() == 1);
		self.max_std	= max_std
		self.slope_std	= slope_std
		self.iter_count = iter_count
		self.start_iter = start_iter
		self.training	= training

	def get_std(self):
		iter = 1 + (self.iter_count[0] - self.start_iter)
		if (iter > 0) :
			return min((iter/125000)*self.slope_std, self.max_std)
		else:
			return 0

	def forward(self, input):
		std = self.get_std()
		if self.training:
			noise = input.new().resize_as_(input)
			if std == 0:
				noise.fill_(0) # no noise
			else:
				noise.normal_(0, std) # Gaussian noise with 0-mean, std-standard deviation
			output = input + noise
		else:
			output = input

		# Return
		return output

	def backward(self, grad_output):
		return grad_output.clone()


## FWD/BWD pass module
class Noise(Module):
	def __init__(self, max_std, slope_std, iter_count, start_iter):
		super(Noise, self).__init__()
		assert(max_std >= 0 and slope_std >= 0);
		assert(iter_count.nelement() == 1);
		self.max_std	= max_std
		self.slope_std	= slope_std
		self.iter_count = iter_count
		self.start_iter = start_iter

	def forward(self, input):
		return NoiseFunction(self.max_std, self.slope_std,
					 		 self.iter_count, self.start_iter, self.training)(input)