import torch
from torch.autograd import Function
from torch.nn import Module

'''
   --------------------- Normalized MSE loss ------------------------------
    This loss computes a normalized version of the MSE loss with the normalizing constant based on the target (each dimension is independent). 
    We compute:
		Loss = 1/N * sum_(i = 1 to N) 0.5 * ((x - mu)^2/sigma)
    where:
		x  	= input
		mu 	= target
		sigma	= scale * abs(target)  + defsigma
	where scale is a user-set parameter, with a default value of 0.5 & defsigma is also a user set parameter (default: 0.005) 
	to reduce numerical instabilities. In effect, we try to normalize the loss to be scale invariant. 
	NOTE: This loss will only work if the input and target values are approx. zero mean
'''
## FWD/BWD pass module
class NormalizedMSESqrtLoss(Module):
	def __init__(self, size_average, scale=0.5, defsigma=0.005):
		super(NormalizedMSESqrtLoss, self).__init__()
		self.size_average = size_average
		self.scale		  = scale
		self.defsigma	  = defsigma

	def forward(self, input, target):
		# Compute loss
		sigma    = (self.scale * target.abs()) + self.defsigma # sigma	= scale * abs(target)  + defsigma
		residual = (input - target).pow(2)	# res = (x - mu)^2
		output   = 0.5 * (residual / sigma).sum() # -- SUM [ 0.5 * ((x-mu)^2/sigma) ]
		if self.size_average:
			output *= (1.0 / input.nelement())

		# Return
		return output