import numpy as np
import open3d as o3d
import torch
from copy import deepcopy
from itertools import combinations

from control.grasp_planner import grasp_planner_simple, Gripper
from functions.primitives import Superquadric
from functions.utils import get_SE3s

def sq_function(pc, parameters):
	# parameter decomposition
	a1 = parameters[:, 0:1]
	a2 = parameters[:, 1:2]
	a3 = parameters[:, 2:3]
	e1 = parameters[:, 3:4]
	e2 = parameters[:, 4:5]

	X = pc[:, 0, :]
	Y = pc[:, 1, :]
	Z = pc[:, 2, :]

	F = (
		torch.abs(X/a1)**(2/e2)
		+ torch.abs(Y/a2)**(2/e2)
		)**(e2/e1) + torch.abs(Z/a3)**(2/e1)

	return F

def check_collision(scene, gripper_SE3, gripper_pc, visualize=False, object_collision_scale = 1.1):

	# decompose SE3 and DSQ parameters
	Ts = scene[0]
	shape_parameters = scene[1]
	
	# bigger
	shape_parameters_big = deepcopy(shape_parameters)
	shape_parameters_big[:, 0] *= object_collision_scale
	shape_parameters_big[:, 1] *= object_collision_scale
	shape_parameters_big[:, 2] *= object_collision_scale

	# predefined grasp pose generation
	grasp_pc = gripper_SE3[:3, :3].dot(gripper_pc.transpose()) + np.expand_dims(gripper_SE3[:3, 3], 1)
	grasp_pc = torch.Tensor(grasp_pc).unsqueeze(0) # 1 x 3 x n_pc

	# surrounding objects
	position = torch.tensor(Ts[:, :3, 3]) # n x 3
	rotation = torch.tensor(Ts[:, :3, :3]) # n x 3 x 3
	parameters = torch.tensor(shape_parameters_big)

	# ground-truth point cloud
	num_points = grasp_pc.shape[2]
	num_objects = position.shape[0]
	rotation_t = rotation.permute(0,2,1)
	grasp_pcs_transformed = - rotation_t @ position.unsqueeze(2) + rotation_t @ grasp_pc
	
	# calculate score
	scores = sq_function(grasp_pcs_transformed, parameters)
	scores = scores.reshape(num_objects, num_points)
	scores = torch.min(scores, dim=1)[0]
	scores = torch.min(scores, dim=0)[0]
	score = scores.item()
	if score < 1:
		bool_collision = True
	else:
		bool_collision = False

	# visualize=True

	if visualize:
  		# superquadric meshes
		mesh_scene = []
		target_color=[0, 0, 1]

		# draw meshes
		for shape_idx, (SE3, shape_parameter) in enumerate(zip(Ts, shape_parameters_big)):
			
			# parameter processing
			parameters = dict()
			parameters['a1'] = shape_parameter[0]
			parameters['a2'] = shape_parameter[1]
			parameters['a3'] = shape_parameter[2]
			parameters['e1'] = shape_parameter[3]
			parameters['e2'] = shape_parameter[4]
			mesh = Superquadric(SE3, parameters, resolution=10).mesh

			if shape_idx == 0:
				mesh.paint_uniform_color(target_color)
			else:
				mesh.paint_uniform_color([0.7, 0.7, 0.7])

			mesh_scene.append(mesh) 	
		
		# draw grippers
		gripper_mesh = Gripper(gripper_SE3, 0.0).mesh
		if bool_collision:
			gripper_mesh.paint_uniform_color([1, 0, 0])
		else:
			gripper_mesh.paint_uniform_color([0, 1, 0])

		# draw gripper point cloud
		grasp_pc_numpy = grasp_pc.detach().cpu().numpy().transpose()
		gripper_pc_o3d = o3d.geometry.PointCloud()
		gripper_pc_o3d.points = o3d.utility.Vector3dVector(grasp_pc_numpy)

		o3d.visualization.draw_geometries(mesh_scene + [gripper_mesh] + [gripper_pc_o3d])

	return bool_collision

def moving_criterior(scene, goal_positions):
	Ts = scene[0]
	parameters = scene[1]
	positions = Ts[:, :2, 3]
	orientations = Ts[:, :3, :3]

	# goal positions
	goal_positions_list = []
	goal_orientations_list = []
	for key, value in goal_positions.items():
		goal_positions_list.append(value['position_xy'])
		goal_orientations_list.append(value['orientation'])

	dists = []
	for i_pri in range(parameters.shape[0]):
		dist_to_goals = []
		for goal_position, goal_orientation in zip(goal_positions_list, goal_orientations_list):
			dist_position = (positions[i_pri, 1] - goal_position[1])**2
			dist_to_goals.append(dist_position)
		dists += [min(dist_to_goals)]

	score = - sum(dists) / len(dists)
	
	return score

def moving_interactive_criterior(scene, goal_position):
	Ts = scene[0]
	parameters = scene[1]
	position = Ts[0, :2, 3]
	orientation = Ts[0, :3, :3]

	desired_position = goal_position['position_xy']
	desired_orientation = goal_position['orientation']

	dist_position = sum((position - desired_position)**2)
	
	return -dist_position

def singulation_criterior(scene, tau):
	Ts = scene[0]
	positions = Ts[:, :2, 3]

	dists = []
	pos_combinations = list(combinations(positions, 2))
	for pos_combination in pos_combinations:
		diff_position = np.sqrt(np.sum((pos_combination[0] - pos_combination[1]) ** 2))
		dists += [min(diff_position - tau, 0)]

	score = min(dists)
	
	return score

def grasp_criterior(scene, 
					gripper_pc=None, 
					visualize=False, 
					table=None, 
					get_valid_grasp_poses=False, 
					use_only_topdown=False, 
					use_ik=False, 
					object_collision_scale=1.0):

	# select real objects whose confidene is 1
	Ts = scene[0]
	shape_parameters = scene[1]

	# target object
	SE3 = Ts[0]
	parameters = dict()
	parameters['a1'] = shape_parameters[0, 0]
	parameters['a2'] = shape_parameters[0, 1]
	parameters['a3'] = shape_parameters[0, 2]
	parameters['e1'] = shape_parameters[0, 3]
	parameters['e2'] = shape_parameters[0, 4]

	# predefined grasp pose generation
	grasp_SE3s = grasp_planner_simple(SE3, parameters, n_gripper=10, ratio=0.9, use_only_topdown=use_only_topdown)
	if len(grasp_SE3s) == 0:
		return 'failed', 'failed'
	grasp_pcs = grasp_SE3s[:, :3, :3]@np.expand_dims(gripper_pc.transpose(), 0) + np.expand_dims(grasp_SE3s[:, :3, 3], 2)
	grasp_pcs = torch.tensor(grasp_pcs)

	# surrounding objects
	position = torch.tensor(Ts[1:, :3, 3])
	rotation = torch.tensor(Ts[1:, :3, :3])
	parameters = torch.tensor(shape_parameters[1:])
	parameters_big = deepcopy(parameters)
	parameters_big[:, 0] *= object_collision_scale
	parameters_big[:, 1] *= object_collision_scale
	parameters_big[:, 2] *= object_collision_scale

	if table is not None:
		table_position = table[1]
		table_rotation = np.eye(3)
		table_parameter = [table[0][0] / 2, table[0][1] / 2, table[0][2] / 2, 0.2, 0.2]
		position = torch.cat([position, torch.tensor(table_position).unsqueeze(0)], dim=0)
		rotation = torch.cat([rotation, torch.tensor(table_rotation).unsqueeze(0)], dim=0)
		parameters_big = torch.cat([parameters_big, torch.tensor(table_parameter).unsqueeze(0)], dim=0)

	# ground-truth point cloud
	num_SE3s = grasp_pcs.shape[0]
	num_points = grasp_pcs.shape[2]
	num_objects = position.shape[0]
	grasp_pcs = grasp_pcs.permute(1,0,2).reshape(3, -1).unsqueeze(0)
	rotation_t = rotation.permute(0,2,1)
	grasp_pcs_transformed = - rotation_t @ position.unsqueeze(2) + rotation_t @ grasp_pcs
	
	# calculate score
	scores = sq_function(grasp_pcs_transformed, parameters_big)
	scores = scores.reshape(num_objects, num_SE3s, num_points)
	scores = torch.min(scores, dim=2)[0]
	scores = torch.min(scores, dim=0)[0]
	score = torch.max((scores > 1).type(torch.float64))

	# valid grasp poses
	valid_grasp_SE3s = grasp_SE3s[scores > 1, :, :]

	if (score == 1) and use_ik:

		# heuristic-based
		ik_available_indices = (valid_grasp_SE3s[:, 1, 3] > -0.55) * \
							   (np.inner(valid_grasp_SE3s[:, :3, 2], np.array([1, 0, 0])) > 0.866)
		valid_grasp_SE3s = valid_grasp_SE3s[ik_available_indices]
		if len(valid_grasp_SE3s) > 0:
			score = 1
		else:
			score = 0	

	if visualize:
  		# superquadric meshes
		mesh_scene = []
		target_color=[0, 0, 1]

		# draw meshes
		for shape_idx, (SE3, shape_parameter) in enumerate(zip(Ts, shape_parameters)):
			
			# parameter processing
			parameters = dict()
			parameters['a1'] = shape_parameter[0]
			parameters['a2'] = shape_parameter[1]
			parameters['a3'] = shape_parameter[2]
			parameters['e1'] = shape_parameter[3]
			parameters['e2'] = shape_parameter[4]
			mesh = Superquadric(SE3, parameters, resolution=10).mesh

			if shape_idx == 0:
				mesh.paint_uniform_color(target_color)
			else:
				mesh.paint_uniform_color([0.7, 0.7, 0.7])

			mesh_scene.append(mesh) 

		# draw table
		if table is not None:	
			parameters = dict()
			parameters['a1'] = table_parameter[0]
			parameters['a2'] = table_parameter[1]
			parameters['a3'] = table_parameter[2]
			parameters['e1'] = table_parameter[3]
			parameters['e2'] = table_parameter[4]
			mesh = Superquadric(get_SE3s(table_rotation, table_position), parameters, resolution=10).mesh

			mesh.paint_uniform_color([0.4, 0.4, 0.4])
			mesh_scene.append(mesh) 

		# draw coordinate
		coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
		mesh_scene.append(coordinate)

		# draw grippers
		for grasp_idx in range(len(grasp_SE3s)):
			print(grasp_SE3s[grasp_idx])
			gripper_mesh = Gripper(grasp_SE3s[grasp_idx], 0.08).mesh
			if scores[grasp_idx] < 1:
				gripper_mesh.paint_uniform_color([1, 0, 0])
			else:
				gripper_mesh.paint_uniform_color([0, 1, 0])

			o3d.visualization.draw_geometries(mesh_scene + [gripper_mesh])

	if get_valid_grasp_poses:
		return score, valid_grasp_SE3s
	else:
		return score

if __name__ == "__main__":

	# parameters
	SE3 = np.eye(4)
	parameters = dict()
	parameters['a1'] = 0.03
	parameters['a2'] = 0.03
	parameters['a3'] = 0.09
	parameters['e1'] = 0.2
	parameters['e2'] = 1.0

	# gripper and sampled points
	gripper = Gripper(np.eye(4), 0.08).mesh
	gripper.compute_vertex_normals()
	gripper_pc = gripper.sample_points_uniformly(number_of_points=256)
	gripper_pc.paint_uniform_color([0.5, 0.5, 0.5])
	gripper_pc = np.asarray(gripper_pc.points)

	# scene example
	scene = np.array([[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00],
					[-5.5511e-17,  8.8534e-02,  5.6913e-02,  2.6891e-02,  0.0000e+00],
					[ 0.0000e+00, -1.3660e-01,  1.8925e-02, -1.1511e-01,  0.0000e+00],
					[ 0.0000e+00, -1.1993e-02,  1.8005e-02,  6.0024e-03,  0.0000e+00],
					[ 5.3496e-22, -9.7300e-06,  1.3134e-05,  7.7196e-06,  0.0000e+00],
					[-2.3279e-21, -3.7540e-05, -2.1653e-05, -3.4680e-05,  0.0000e+00],
					[-3.2762e-18, -1.4746e-01, -3.0475e-01, -1.0875e-01,  0.0000e+00],
					[ 1.0000e+00,  9.8907e-01,  9.5243e-01,  9.9407e-01,  0.0000e+00],
					[ 1.5000e-02,  3.2000e-02,  4.0000e-02,  2.8000e-02,  0.0000e+00],
					[ 3.8000e-02,  1.4000e-02,  4.0000e-02,  8.0000e-02,  0.0000e+00],
					[ 4.2000e-02,  3.0000e-02,  6.0000e-02,  4.8000e-02,  0.0000e+00],
					[ 2.0000e-01,  2.0000e-01,  2.0000e-01,  2.0000e-01,  0.0000e+00],
					[ 2.0000e-01,  2.0000e-01,  1.0000e+00,  2.0000e-01,  0.0000e+00],
					[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
					[ 1.0000e-02,  1.0000e-02,  1.0000e-02,  1.0000e-02,  0.0000e+00],
					[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00],
					[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]])
	scene = scene[:, [1, 2, 0, 3, 4]]

	score = grasp_criterior(scene, gripper_pc=gripper_pc, visualize=True)