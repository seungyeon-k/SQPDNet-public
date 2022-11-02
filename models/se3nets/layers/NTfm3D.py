import torch
from torch.autograd import Function
from torch.nn import Module
from _ext import se3layers

'''
	--------------------- Non-Rigidly deform the input point cloud by transforming and blending across multiple SE3s ------------------------------
   NTfm3D() :
   NTfm3D.forward(3D points, masks, Rt)
   NTfm3D.backward(grad_output)

   NTfm3D will transform the given input points "x" (B x 3 x N x M) and "k" masks (B x k x N x M) via a set of 3D affine transforms (B x k x 3 x 4), 
	resulting in a set of transformed 3D points (B x 3 x N x M). The transforms will be applied to all the points 
	and their outputs are interpolated based on the mask weights:
		output = mask(1,...) .* (R_1 * x + t_1) + mask(2,...) .* (R_2 * x + t_2) + .....
	Each 3D transform is a (3x4) matrix [R|t], where "R" is a (3x3) affine matrix and "t" is the translation (3x1).
	Note: The mask values have to sum to 1.0
		sum(mask,2) = 1
'''

## FWD/BWD pass function
class NTfm3DFunction(Function):
	def __init__(self, use_mask_gradmag=True):
		super(NTfm3DFunction, self).__init__()
		self.use_mask_gradmag = use_mask_gradmag  # Default this is true

	def forward(self, points, masks, transforms):
		# Check dimensions
		batch_size, num_channels, data_height, data_width = points.size()
		num_se3 = masks.size()[1]
		assert(num_channels == 3)
		assert(masks.size() == torch.Size([batch_size, num_se3, data_height, data_width]))
		assert(transforms.size() == torch.Size([batch_size, num_se3, 3, 4])) # Transforms [R|t]

		# Create output (or reshape)
		output = points.new_zeros(*points.size())

		# Run the FWD pass
		if points.is_cuda:
			se3layers.NTfm3D_forward_cuda(points, masks, transforms, output)
		elif points.type() == 'torch.DoubleTensor':
			se3layers.NTfm3D_forward_double(points, masks, transforms, output)
		else:
			se3layers.NTfm3D_forward_float(points, masks, transforms, output)
		self.save_for_backward(points, masks, transforms, output) # Save for BWD pass

		# Return
		return output

	def backward(self, grad_output):
		# Get saved tensors
		points, masks, transforms, output = self.saved_tensors
		assert(grad_output.is_same_size(output))

		# Initialize grad input
		grad_points 	= points.new_zeros(*points.size())
		grad_masks      = masks.new_zeros(*masks.size())
		grad_transforms = transforms.new_zeros(*transforms.size())

		# Run the BWD pass
		if grad_output.is_cuda:
			se3layers.NTfm3D_backward_cuda(points, masks, transforms, output,
										   grad_points, grad_masks, grad_transforms, grad_output,
										   self.use_mask_gradmag)
		elif grad_output.type() == 'torch.DoubleTensor':
			se3layers.NTfm3D_backward_double(points, masks, transforms, output,
											 grad_points, grad_masks, grad_transforms, grad_output,
											 self.use_mask_gradmag)
		else:
			se3layers.NTfm3D_backward_float(points, masks, transforms, output,
											grad_points, grad_masks, grad_transforms, grad_output,
											self.use_mask_gradmag)

		# Return
		return grad_points, grad_masks, grad_transforms

## FWD/BWD pass module
class NTfm3D(Module):
	def __init__(self, use_mask_gradmag=True):
		super(NTfm3D, self).__init__()
		self.use_mask_gradmag = use_mask_gradmag  # Default this is true

	def forward(self, points, masks, transforms):
		return NTfm3DFunction(use_mask_gradmag=self.use_mask_gradmag)(points, masks, transforms)
