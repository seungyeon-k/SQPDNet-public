import torch
from torch.autograd import Function
from torch.nn import Module

'''
   DepthImageToDense3DPoints(height,width,fy,fx,cy,cx) :
   DepthImageToDense3DPoints.forward(depth_img)
   DepthImageToDense3DPoints.backward(grad_output)

   DepthImageToDense3DPoints takes in a set of depth maps (B x 1 x N x M) and outputs a 3D point (B x 3 x N x M) for every pixel in the image:
	The 3D points are defined w.r.t a frame centered in the center of the image. The frame has the following convention:
		+ve X is increasing columns, +ve Y is increasing rows, +ve Z is moving closer to the objects in the scene
	X and Y co-ordinates are computed based on the passed in camera parameters using the following formula:
		X = (xp - cx)/fx 
		Y = (yp - cy)/fy
		where (fx,fy) are the focal length in (x,y) => (col,row) and
				(cx,cy) are the centers of projection in (x,y) => (col,row)
  	  	Z   = 0 is the image plane
	  	Z   > 0 goes forward from the image plane
	The parameters (fy,fx,cy,cx) are optional - they default to values for a 240 x 320 image from an Asus Xtion Pro
'''

## FWD/BWD pass function
class DepthImageToDense3DPointsFunction(Function):
	def __init__(self, height, width, base_grid,
				 fy = 589.3664541825391 * 0.5,
				 fx = 589.3664541825391 * 0.5,
				 cy = 240.5 * 0.5,
				 cx = 320.5 * 0.5):
		# Image dimensions
		assert (height > 1 and width > 1)
		self.height    = height
		self.width 	   = width
		self.base_grid = base_grid

		# Declare params(default for a 240 x 320 image)
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		assert (self.fx > 0 and self.fy > 0 and
				self.cx >= 0 and self.cx < self.width and
				self.cy >= 0 and self.cy < self.height)

	def forward(self, depth):
		# Check dimensions (B x 1 x H x W)
		batch_size, num_channels, num_rows, num_cols = depth.size()
		assert(num_channels == 1)
		assert(num_rows == self.height)
		assert(num_cols == self.width)

		# Compute output = (x, y, depth)
		output = depth.new().resize_(batch_size,3,num_rows,num_cols)
		xy, z = output.narrow(1,0,2), output.narrow(1,2,1)
		z.copy_(depth) # z = depth
		xy.copy_(depth.expand_as(xy) * self.base_grid.narrow(1,0,2).expand_as(xy)) # [x*z, y*z, z]

		# Return
		return output

	def backward(self, grad_output):
		# Get saved tensors & check dimensions
		_, _, num_rows, num_cols = self.base_grid.size()
		assert((grad_output.size(2) == num_rows) and (grad_output.size(3) == num_cols));

		# Compute grad input
		# g_z = x * go_x + y * go_y + go_z
		grad_input = (self.base_grid.expand_as(grad_output) * grad_output).sum(1)

		# Return
		return grad_input

## FWD/BWD pass module
class DepthImageToDense3DPoints(Module):
	def __init__(self, height, width,
				 fy = 589.3664541825391 * 0.5,
				 fx = 589.3664541825391 * 0.5,
				 cy = 240.5 * 0.5,
				 cx = 320.5 * 0.5):
		super(DepthImageToDense3DPoints, self).__init__()

		# Image dimensions
		assert(height > 1 and width > 1)
		self.height = height
		self.width  = width
		self.base_grid = None

		# Declare params(default for a 240 x 320 image)
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		assert (self.fx > 0 and self.fy > 0 and
				self.cx >= 0 and self.cx < self.width and
				self.cy >= 0 and self.cy < self.height)

	def forward(self, input):
		# Generate basegrid only once
		# TODO: This explicitly assumes that a "Variable" is passed in. Will fail if a tensor is passed in
		if self.base_grid is None:
			self.base_grid = torch.ones(1,3,self.height,self.width).type_as(input.data)  # (x,y,1)
			for j in xrange(0, self.width):  # +x is increasing columns
				self.base_grid[:, 0, :, j].fill_((j - self.cx) / self.fx)
			for i in xrange(0, self.height):  # +y is increasing rows
				self.base_grid[:, 1, i, :].fill_((i - self.cy) / self.fy)

		# Run the rest of the FWD pass
		return DepthImageToDense3DPointsFunction(self.height, self.width, self.base_grid,
												 self.fy, self.fx, self.cy, self.cx)(input)