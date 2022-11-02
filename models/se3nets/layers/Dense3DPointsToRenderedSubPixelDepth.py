import torch
from torch.autograd import Function
from torch.nn import Module
from _ext import se3layers

'''
	Dense3DPointsToRenderedSubPixelDepth(fy,fx,cy,cx) :
   	Dense3DPointsToRenderedSubPixelDepth.forward(points)
   	Dense3DPointsToRenderedSubPixelDepth.backward(grad_output)

   	DensePoints3DToRenderedSubPixelDepth takes in a set of dense 3D points (B x 3 x N x M) and outputs a 3D vector (B x 3 x N x M) 
   	for every 3D point where (X,Y) are the sub-pixel corresponding to the 3D point (x,y,z) based on the camera-parameters and Z = z
   	NOTE: This layer also does the depth test - points which are not visible are set to all zeros
      	X = fx * (x/z) + cx
      	Y = fy * (y/z) + cy
      	Z = z
      	where (fx,fy) are the focal length of the camera in (x,y) => (col,row) and
            (cx,cy) are the centers of projection in (x,y) => (col,row)
      	Points with z = 0 are set to (X,Y,Z) = (0,0,0)
      	Points that are not visible are set to (X,Y,Z) = (0,0,0)
   	The parameters (fy,fx,cy,cx) are optional - they default to values for a 240 x 320 image
'''

## FWD/BWD pass function
class Dense3DPointsToRenderedSubPixelDepthFunction(Function):
	def __init__(self,
				 fy = 589.3664541825391 * 0.5,
				 fx = 589.3664541825391 * 0.5,
				 cy = 240.5 * 0.5,
				 cx = 320.5 * 0.5):
		# Declare params(default for a 240 x 320 image)
		self.fy, self.fx, self.cy, self.cx = fy, fx, cy, cx
		self.indexmap = None
		assert (self.fy > 0 and self.fx > 0 and self.cy >= 0 and self.cx >= 0)

	def forward(self, points):
		# Check dimensions
		batch_size, num_channels, data_height, data_width = points.size()
		assert (num_channels == 3);
		assert (self.cx < data_width and self.cy < data_height)

		# Create output & temp data (3D)
		output = points.new().resize_as_(points)
		if not self.indexmap:
			self.indexmap = points.new()
		self.indexmap.resize_(batch_size,1,data_height,data_width) # 1-channel

		# Run the FWD pass
		if points.is_cuda:
			se3layers.Project3DPointsToSubPixelDepth_forward_cuda(points, self.indexmap, output,
																  self.fy, self.fx, self.cy, self.cx)
		elif points.type() == 'torch.DoubleTensor':
			se3layers.Project3DPointsToSubPixelDepth_forward_double(points, self.indexmap, output,
																  self.fy, self.fx, self.cy, self.cx)
		else:
			se3layers.Project3DPointsToSubPixelDepth_forward_float(points, self.indexmap, output,
																  self.fy, self.fx, self.cy, self.cx)

		# Return
		self.save_for_backward(points)  # Save for BWD pass
		return output

	def backward(self, grad_output):
		# Get saved tensors
		points = self.saved_tensors[0]
		assert (grad_output.is_same_size(points))

		# Initialize grad input
		grad_points = points.new().resize_as_(points)

		# Run the BWD pass
		if grad_output.is_cuda:
			se3layers.Project3DPointsToSubPixelDepth_backward_cuda(points, self.indexmap, grad_points, grad_output,
																  self.fy, self.fx, self.cy, self.cx)
		elif grad_output.type() == 'torch.DoubleTensor':
			se3layers.Project3DPointsToSubPixelDepth_backward_double(points, self.indexmap, grad_points, grad_output,
																  self.fy, self.fx, self.cy, self.cx)
		else:
			se3layers.Project3DPointsToSubPixelDepth_backward_float(points, self.indexmap, grad_points, grad_output,
																  self.fy, self.fx, self.cy, self.cx)

		# Return
		return grad_points

## FWD/BWD pass module
class Dense3DPointsToRenderedSubPixelDepth(Module):
	def __init__(self,
				 fy=589.3664541825391 * 0.5,
				 fx=589.3664541825391 * 0.5,
				 cy=240.5 * 0.5,
				 cx=320.5 * 0.5):
		super(Dense3DPointsToRenderedSubPixelDepth, self).__init__()
		self.fy, self.fx, self.cy, self.cx = fy, fx, cy, cx
		assert (self.fy > 0 and self.fx > 0 and self.cy >= 0 and self.cx >= 0)

	def forward(self, points):
		return Dense3DPointsToRenderedSubPixelDepthFunction(self.fy, self.fx, self.cy, self.cx)(points)
