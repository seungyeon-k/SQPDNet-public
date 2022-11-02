import numpy as np
import open3d as o3d

from functions.utils import get_SE3s, exp_so3
from functions.primitives import Superquadric
from copy import deepcopy


class Gripper:
	def __init__(self, SE3, width=0):
		self.hand_SE3 = SE3
		self.gripper_width = width
		if width < 0:
			print("gripper width exceeds minimum width. gripper width is set to 0")
			self.gripper_width = 0
		if width > 0.08:
			print("gripper width exceeds maximum width. gripper width is set to 0.08")
			self.gripper_width = 0.08

		self.hand = o3d.io.read_triangle_mesh("assets/gripper/hand.ply")
		self.hand.compute_vertex_normals()
		self.hand.paint_uniform_color([0.9, 0.9, 0.9])
		self.finger1 = o3d.io.read_triangle_mesh("assets/gripper/finger.ply")
		self.finger1.compute_vertex_normals()
		self.finger1.paint_uniform_color([0.7, 0.7, 0.7])
		self.finger2 = o3d.io.read_triangle_mesh("assets/gripper/finger.ply")
		self.finger2.compute_vertex_normals()
		self.finger2.paint_uniform_color([0.7, 0.7, 0.7])

		self.finger1_M = get_SE3s(np.identity(3), np.array([0, self.gripper_width/2, 0.1654/3]))
		self.finger2_M = get_SE3s(exp_so3(np.asarray([0, 0, 1]) * np.pi), np.array([0, -self.gripper_width/2, 0.1654/3]))

		self.finger1_SE3 = np.dot(self.hand_SE3, self.finger1_M)
		self.finger2_SE3 = np.dot(self.hand_SE3, self.finger2_M)
			
		self.hand.transform(self.hand_SE3)
		self.finger1.transform(self.finger1_SE3)
		self.finger2.transform(self.finger2_SE3)
		self.mesh = self.hand + self.finger1 + self.finger2

def grasp_planner_simple(SE3, parameters, n_gripper=10, d=0.08, max_width=0.07, ratio=0.8, augment=True, use_only_topdown=False):

	eps = 1e-1
	gripper_SE3s = []

	if abs(parameters['e1'] - 0.2) < eps and abs(parameters['e2'] - 0.2) < eps: 

		# top-down grasp
		for idx in range(n_gripper):
			if parameters['a2'] < max_width / 2:
				p = np.array([parameters['a1'] * (2 * ratio * idx / (n_gripper - 1) - ratio), 0, d + parameters['a3']])
				theta = np.pi / 2
				gripper_SO3 = np.array([[np.sin(theta), -np.cos(theta), 0],
								[-np.cos(theta), -np.sin(theta), 0],
								[0, 0, -1]])
				gripper_SE3 = get_SE3s(gripper_SO3, p)
				gripper_SE3s.append(SE3.dot(gripper_SE3))

			if parameters['a1'] < max_width / 2:
				p = np.array([0, parameters['a2'] * (2 * ratio * idx / (n_gripper - 1) - ratio), d + parameters['a3']])
				theta = 0
				gripper_SO3 = np.array([[np.sin(theta), -np.cos(theta), 0],
								[-np.cos(theta), -np.sin(theta), 0],
								[0, 0, -1]])
				gripper_SE3 = get_SE3s(gripper_SO3, p)
				gripper_SE3s.append(SE3.dot(gripper_SE3))

		if not use_only_topdown:
			# side grasp
			for idx in range(n_gripper):
				if parameters['a2'] < max_width / 2:
					p = np.array([parameters['a1'] + d, 0, parameters['a3'] * (2 * ratio * idx / (n_gripper - 1) - ratio)])
					theta = np.pi
					gripper_SO3 = np.array([[0, np.sin(theta), np.cos(theta)],
									[0, np.cos(theta), -np.sin(theta)],
									[-1, 0, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))
					p = np.array([-parameters['a1'] - d, 0, parameters['a3'] * (2 * ratio * idx / (n_gripper - 1) - ratio)])
					theta = 0
					gripper_SO3 = np.array([[0, np.sin(theta), np.cos(theta)],
									[0, np.cos(theta), -np.sin(theta)],
									[-1, 0, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))

				if parameters['a1'] < max_width / 2:
					p = np.array([0, parameters['a2'] + d, parameters['a3'] * (2 * ratio * idx / (n_gripper - 1) - ratio)])
					theta = np.pi / 2
					gripper_SO3 = np.array([[0, np.sin(theta), np.cos(theta)],
									[0, np.cos(theta), -np.sin(theta)],
									[-1, 0, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))
					p = np.array([0, -parameters['a2'] - d, parameters['a3'] * (2 * ratio * idx / (n_gripper - 1) - ratio)])
					theta = -np.pi / 2
					gripper_SO3 = np.array([[0, np.sin(theta), np.cos(theta)],
									[0, np.cos(theta), -np.sin(theta)],
									[-1, 0, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))

			# edge grasp 
			for idx in range(n_gripper):
				if parameters['a3'] < max_width / 2:
					p = np.array([parameters['a1'] + d, parameters['a2'] * (2 * ratio * idx / (n_gripper - 1) - ratio), 0])
					theta = - np.pi / 2
					gripper_SO3 = np.array([[np.cos(theta), 0, np.sin(theta)],
									[np.sin(theta), 0, -np.cos(theta)],
									[0, 1, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))		
					p = np.array([- parameters['a1'] - d, parameters['a2'] * (2 * ratio * idx / (n_gripper - 1) - ratio), 0])
					theta = np.pi / 2
					gripper_SO3 = np.array([[np.cos(theta), 0, np.sin(theta)],
									[np.sin(theta), 0, -np.cos(theta)],
									[0, 1, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3)) 	
					p = np.array([parameters['a1'] * (2 * ratio * idx / (n_gripper - 1) - ratio), parameters['a2'] + d, 0])
					theta = 0
					gripper_SO3 = np.array([[np.cos(theta), 0, np.sin(theta)],
									[np.sin(theta), 0, -np.cos(theta)],
									[0, 1, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))	
					p = np.array([parameters['a1'] * (2 * ratio * idx / (n_gripper - 1) - ratio), - parameters['a2'] - d, 0])
					theta = np.pi
					gripper_SO3 = np.array([[np.cos(theta), 0, np.sin(theta)],
									[np.sin(theta), 0, -np.cos(theta)],
									[0, 1, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))	 		

	elif abs(parameters['e1'] - 0.2) < eps and abs(parameters['e2'] - 1.0) < eps and abs(parameters['a2'] - parameters['a1']) < eps:

		# top-down grasp
		for idx in range(n_gripper * 2):
			if parameters['a2'] < max_width / 2:
				p = np.array([0, 0, d + parameters['a3']])
				theta = 2 * np.pi * idx / (n_gripper * 2)
				gripper_SO3 = np.array([[np.sin(theta), -np.cos(theta), 0],
								[-np.cos(theta), -np.sin(theta), 0],
								[0, 0, -1]])
				gripper_SE3 = get_SE3s(gripper_SO3, p)
				gripper_SE3s.append(SE3.dot(gripper_SE3))

		if not use_only_topdown:

			# side grasp
			for idx_theta in range(n_gripper):
				for idx_z in range(n_gripper):
					if parameters['a2'] < max_width / 2:
						theta = 2 * np.pi * idx_theta / (n_gripper)
						p = np.array([(parameters['a1'] + d) * np.cos(theta), (parameters['a2'] + d) * np.sin(theta), parameters['a3'] * (2 * ratio * idx_z / (n_gripper - 1) - ratio)])
						gripper_SO3 = np.array([[0, np.sin(theta), -np.cos(theta)],
										[0, -np.cos(theta), -np.sin(theta)],
										[-1, 0, 0]])
						gripper_SE3 = get_SE3s(gripper_SO3, p)
						gripper_SE3s.append(SE3.dot(gripper_SE3))

			# edge grasp
			for idx in range(n_gripper * 2):
				if parameters['a3'] < max_width / 2:
					theta = 2 * np.pi * idx / (n_gripper * 2)
					p = np.array([(parameters['a1'] + d) * np.cos(theta), (parameters['a2'] + d) * np.sin(theta), 0])
					gripper_SO3 = np.array([[np.cos(theta - np.pi / 2), 0, np.sin(theta - np.pi / 2)],
									[np.sin(theta - np.pi / 2), 0, -np.cos(theta - np.pi / 2)],
									[0, 1, 0]])
					gripper_SE3 = get_SE3s(gripper_SO3, p)
					gripper_SE3s.append(SE3.dot(gripper_SE3))	

	if augment:
		gripper_SE3s_augment = deepcopy(gripper_SE3s)
		for i in range(len(gripper_SE3s)):
			rotated_gripper_SE3 = gripper_SE3s[i].dot(np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
			gripper_SE3s_augment.append(rotated_gripper_SE3)

		return np.array(gripper_SE3s_augment)
	else:
		return gripper_SE3s


if __name__ == "__main__":

	# parameters
	SE3 = np.eye(4)
	parameters = dict()
	parameters['a1'] = 0.03
	parameters['a2'] = 0.03
	parameters['a3'] = 0.09
	parameters['e1'] = 0.2
	parameters['e2'] = 1.0
	n_gripper = 5

	# load deformable superquadric mesh
	shape = Superquadric(SE3, parameters, resolution=100)
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
	mesh_list = [frame, shape.mesh]

	# sample 
	grasp_SE3s = grasp_planner_simple(SE3, parameters, n_gripper=n_gripper)
	for idx in range(len(grasp_SE3s)):
		gripper = Gripper(grasp_SE3s[idx], 0.08)
		mesh_list.append(gripper.mesh)

	o3d.visualization.draw_geometries(mesh_list)
