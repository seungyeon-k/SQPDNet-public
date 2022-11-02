import torch
from torch.autograd import Function
from torch.nn import Module

'''
   --------------------- Huber loss ------------------------------
'''
'''
## FWD/BWD pass function
class HuberLossFunction(Function):
	def __init__(self, size_average, delta):
		self.size_average = size_average
		self.delta		  = delta

	def forward(self, input, target):
		# Compute loss
		self.save_for_backward(input, target)
		absDiff = (input - target).abs_()
		minDiff = torch.clamp(absDiff, max=self.delta) # cmin
		output  = 0.5 * (minDiff * (2*absDiff - minDiff)).view(1,-1).sum(1) # Seems like autograd Functions need to return tensors
		if self.size_average:
			output *= (1.0/input.nelement())

		# Return
		return output

	def backward(self, grad_output):
		# Get saved tensors & setup scale factor
		input, target = self.saved_tensors
		scale = (1.0/input.nelement()) if self.size_average else 1.0

		# Compute gradient (note that there's no need for the grad_output here)
		minDiff = torch.clamp((input - target).abs_(), max=self.delta) # cmin
		grad_input = (input - target).sign() * minDiff * scale

		# Return
		return grad_input # Need to return grads for both inputs here
'''

## FWD/BWD pass module
class HuberLoss(Module):
	def __init__(self, size_average, delta):
		super(HuberLoss, self).__init__()
		self.size_average = size_average
		self.delta		  = delta
		assert(delta >= 0);

	def forward(self, input, target):
		# Compute loss
		absDiff = (input - target).abs()
		minDiff = torch.clamp(absDiff, max=self.delta)  # cmin
		output = 0.5 * (minDiff * (2 * absDiff - minDiff)).sum()
		if self.size_average:
			output *= (1.0 / input.nelement())

		# Return
		return output
