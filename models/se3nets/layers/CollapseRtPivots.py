import torch
from torch.autograd import Function
from torch.nn import Module

'''
	--------------------- Collapses [R|t|pivots] to [R|t'] where t' = p + t - Rp ------------------------------
   CollapseRtPivots() :
   CollapseRtPivots.forward(input)
   CollapseRtPivots.backward(grad_output)

   CollapseRtPivots will transform the given input transforms [R|t|p] (B x N x 3 x 5) to a set of outputs [R'|t'] (B x N x 3 x 4) where 
	R' = R & t' = t + p - Rp
   Each 3D transform is a (3x5) matrix [R|t|p],
   where "R" is a (3x3) affine matrix, "t" is a translation (3x1) and "p" is a pivot (3x1).
'''

## FWD/BWD pass function
class CollapseRtPivotsFunction(Function):
	def forward(self, input):
		# Check dimensions
		batch_size, num_se3, num_rows, num_cols = input.size()
		assert (num_rows == 3 and num_cols == 5);

		# Init for FWD pass
		self.save_for_backward(input)
		input_v = input.view(-1, 3, 5);
		r = input_v.narrow(2, 0, 3);
		t = input_v.narrow(2, 3, 1);
		p = input_v.narrow(2, 4, 1);

		# Compute output = [r, t + p - Rp]
		output = input.new().resize_(batch_size, num_se3, 3, 4);
		output.narrow(3, 0, 3).copy_(r); # r
		output.narrow(3, 3, 1).copy_(t+p).add_(-1, torch.bmm(r, p)); # t + p - Rp

		# Return
		return output;

	def backward(self, grad_output):
		# Get saved tensors & setup vars
		input = self.saved_tensors[0]
		input_v = input.view(-1, 3, 5);
		r = input_v.narrow(2, 0, 3);
		p = input_v.narrow(2, 4, 1);
		ro_g = grad_output.view(-1, 3, 4).narrow(2, 0, 3);
		to_g = grad_output.view(-1, 3, 4).narrow(2, 3, 1);

		# Initialize grad input
		input_g = input.new().resize_as_(input);
		input_g.narrow(3,0,3).copy_(ro_g).add_(-1, torch.bmm(to_g, p.transpose(1,2))); # r_g = ro_g - (to_g * p^T)
		input_g.narrow(3,3,1).copy_(to_g); 											  # t_g = to_g
		input_g.narrow(3,4,1).copy_(to_g).add_(-1, torch.bmm(r.transpose(1,2), to_g)); # p_g = to_g - (R^T * to_g)

		# Return
		return input_g;


## FWD/BWD pass module
class CollapseRtPivots(Module):
	def __init__(self):
		super(CollapseRtPivots, self).__init__()

	def forward(self, input):
		return CollapseRtPivotsFunction()(input)