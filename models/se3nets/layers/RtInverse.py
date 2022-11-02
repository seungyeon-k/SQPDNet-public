import torch
from torch.autograd import Function
from torch.nn import Module

'''
	--------------------- Invert transformations of type R|t ------------------------------
   RtInverse() :
   RtInverse.forward(input)
   RtInverse.backward(grad_output)

   Inverts the transform: Given [R|t], returns [R^T | -R^T*t]
   Size of inputs/outputs: (B x k x 3 x 4)
'''

## FWD/BWD pass function
class RtInverseFunction(Function):
	def forward(self, input):
		# Check dimensions
		_, _, num_rows, num_cols = input.size()
		assert (num_rows == 3 and num_cols == 4);

		# Init for FWD pass
		self.save_for_backward(input)
		input_v = input.view(-1, 3, 4);
		r = input_v.narrow(2, 0, 3);
		t = input_v.narrow(2, 3, 1);

		# Compute output = [r^T -r^T * t]
		output = input.new().resize_as_(input);
		output.narrow(3, 0, 3).copy_(r.transpose(1,2)); # r^T
		output.narrow(3, 3, 1).copy_(torch.bmm(r.transpose(1,2), t).mul_(-1)); # -r^T * t

		# Return
		return output;

	def backward(self, grad_output):
		# Get saved tensors & setup vars
		input = self.saved_tensors[0]
		input_v = input.view(-1, 3, 4);
		r = input_v.narrow(2, 0, 3);
		t = input_v.narrow(2, 3, 1);
		ro_g = grad_output.contiguous().view(-1, 3, 4).narrow(2, 0, 3);
		to_g = grad_output.contiguous().view(-1, 3, 4).narrow(2, 3, 1);

		# Initialize grad input
		input_g = input.new().resize_as_(input);

		# Grad "R"
		input_g.narrow(3,0,3).copy_(torch.bmm(t, to_g.transpose(1,2)).mul_(-1)).add_(ro_g.transpose(1,2)); # r_g = ro_g ^ T - (t * to_g^T)

		# Grad "t"
		input_g.narrow(3,3,1).copy_(torch.bmm(r, to_g).mul_(-1)); # t_g = -r * to_g

		# Return
		return input_g;


## FWD/BWD pass module
class RtInverse(Module):
	def __init__(self):
		super(RtInverse, self).__init__()

	def forward(self, input):
		return RtInverseFunction()(input)
