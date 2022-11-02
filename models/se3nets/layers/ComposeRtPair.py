import torch
from torch.autograd import Function
from torch.nn import Module

'''
	--------------------- Compose multiple transformations ------------------------------
   ComposeRtPair() :
   ComposeRtPair.forward(input1, input2)
   ComposeRtPair.backward(grad_output)

   ComposeRtPair will compose the given pair of input transforms (T_n, T_n-1) => of size (B x N x 3 x 4) 
   resulting in a set of composed transforms T_n * T_n-1 (also of size B x N x 3 x 4). 
   Each 3D transform is a (3x4) matrix [R|t],
   where "R" is a (3x3) affine matrix and "t" is a translation (3x1).
'''

## FWD/BWD pass function
class ComposeRtPairFunction(Function):
	def forward(self, A, B):
		# Check dimensions
		batch_size, num_se3, num_rows, num_cols = A.size()
		assert(num_rows == 3 and num_cols == 4);
		assert(A.is_same_size(B));
		
		# Init for FWD pass
		self.save_for_backward(A,B); # Save for BWD pass
		Av = A.view(-1,3,4);
		Bv = B.view(-1,3,4);
		rA = Av.narrow(2,0,3);
		rB = Bv.narrow(2,0,3);
		tA = Av.narrow(2,3,1);
		tB = Bv.narrow(2,3,1);

		# Init output
		output = A.new().resize_as_(A);
		output.view(-1,3,4).narrow(2,0,3).copy_(torch.bmm(rA, rB));
		output.view(-1,3,4).narrow(2,3,1).copy_(torch.baddbmm(tA, rA, tB));

		# Return
		return output;

	def backward(self, grad_output):
		# Get saved tensors & setup vars
		A, B = self.saved_tensors
		Av = A.view(-1, 3, 4);
		Bv = B.view(-1, 3, 4);
		rA = Av.narrow(2, 0, 3);
		rB = Bv.narrow(2, 0, 3);
		tA = Av.narrow(2, 3, 1);
		tB = Bv.narrow(2, 3, 1);
		r_g = grad_output.view(-1,3,4).narrow(2,0,3);
		t_g = grad_output.view(-1,3,4).narrow(2,3,1);

		# Initialize grad input
		A_g = grad_output.new().resize_as_(grad_output);
		B_g = grad_output.new().resize_as_(grad_output);

		# Compute gradients w.r.t translations tA & tB
		# t = rA * tB + tA
		A_g.view(-1,3,4).narrow(2,3,1).copy_(t_g); # tA_g = t_g
		B_g.view(-1,3,4).narrow(2,3,1).copy_(torch.bmm(rA.transpose(1,2), t_g)); # tB_g = rA ^ T * t_g

		# Compute gradients w.r.t rotations rA & rB
		# r = rA * rB
		A_g.view(-1,3,4).narrow(2,0,3).copy_(torch.bmm(r_g, rB.transpose(1,2)).baddbmm_(t_g, tB.transpose(1,2))); # rA_g = r_g * rB ^ T + (t_g * tB ^ T)
		B_g.view(-1,3,4).narrow(2,0,3).copy_(torch.bmm(rA.transpose(1,2), r_g)); # rB_g = rA ^ T * r_g (similar to translation grad, but with 3 column vectors)

		# Return
		return A_g, B_g;

## FWD/BWD pass module
class ComposeRtPair(Module):
	def __init__(self):
		super(ComposeRtPair, self).__init__()

	def forward(self, A, B):
		return ComposeRtPairFunction()(A, B)