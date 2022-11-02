import numpy as np
import torch
import random
import time
from copy import deepcopy
import os
import pybullet as p
from datetime import datetime
import open3d as o3d

from control.grasp_planner import Gripper
from control.criteriors import check_collision, moving_criterior, moving_interactive_criterior, grasp_criterior, singulation_criterior
from loader.segmentation_dataset import normalize_pointcloud
from functions.point_clouds import noise_augmentation, upsample_pointcloud
from functions.open3d_renderer import objects_renderer_with_pc
from functions.utils import quats_to_matrices, get_SE3s, matrices_to_quats
from functions.communicator_server import Listener
from trainers.recognition_trainer import mesh_generator_recog_pred

class Controller:
	def __init__(self, cfg_controller, model, device, realworld=False, simulator=False):
		self.model = model
		self.model.device = device
		self.device = device
		self.env = self.model.env
		self.visualization = False
		self.realworld = realworld
		self.realworld_debug = False
		self.simulator = simulator

		# for debugging
		self.realworld_debug = False
		self.visualize_recognized_objects = False
		self.visualize_gripper_pc = False
		self.visualize_grasp_candidates = False
		self.visualize_collision = False

		# default setting
		self.num_directions = 8
		self.num_z = 5

		# table setting
		self.table_size = [0.5, 0.895, 0.05]
		self.table_x_augment = 2.0
		self.run_id = datetime.now().strftime('%Y%m%d-%H%M')

		# control settings
		self.sample_num = cfg_controller.sample_num
		self.action_horizon = cfg_controller.action_horizon
		self.debug = cfg_controller.debug
		self.criterior = cfg_controller.criterior
		if self.criterior == 'moving':
			self.goal_positions = cfg_controller.goal_positions
		elif self.criterior == 'moving_interactive':
			self.goal_position = cfg_controller.goal_position
		elif self.criterior == 'singulation':
			self.tau = cfg_controller.tau
		elif self.criterior == 'grasp':
			self.use_only_topdown = False
			self.use_ik = True
		elif self.criterior == 'grasp_top':
			self.criterior = 'grasp'
			self.use_only_topdown = True
			self.use_ik = False

		# target object
		self.target_object = cfg_controller.target_object # 'largest', 'smallest', 'cylinder'

		# communication setting
		if self.realworld:
			self.ip = cfg_controller.ip
			self.port = cfg_controller.port
			self.table_offest = 0.002

		# collision scale
		self.camera_collision_scale = 1.1
		self.gripper_collision_scale = 1.1
		self.object_collision_scale = 1.1

		# camera
		camera_size = np.array([0.04, 0.095, 0.125]) * self.camera_collision_scale
		camera = o3d.geometry.TriangleMesh.create_box(width = camera_size[0], height = camera_size[1], depth = camera_size[2])
		camera.translate([0.0625 - camera_size[0]/2, - camera_size[1]/2, 0.0 - camera_size[2]/2])
		camera.compute_vertex_normals()

		# gripper point cloud (closed)
		gripper_closed = Gripper(np.eye(4), 0.0).mesh
		gripper_closed.compute_vertex_normals()
		gripper_closed += camera
		self.gripper_closed_pc = gripper_closed.sample_points_uniformly(number_of_points=1024)
		self.gripper_closed_pc.paint_uniform_color([0.5, 0.5, 0.5])
		if self.visualize_gripper_pc:
			frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
			o3d.visualization.draw_geometries([frame, gripper_closed, self.gripper_closed_pc])	
		self.gripper_closed_pc = np.asarray(self.gripper_closed_pc.points) * self.gripper_collision_scale

		# gripper point cloud (open)
		gripper_open = Gripper(np.eye(4), 0.08).mesh
		gripper_open.compute_vertex_normals()
		gripper_open += camera
		self.gripper_open_pc = gripper_open.sample_points_uniformly(number_of_points=2048)
		self.gripper_open_pc.paint_uniform_color([0.5, 0.5, 0.5])
		if self.visualize_gripper_pc: 
			frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
			o3d.visualization.draw_geometries([frame, gripper_open, self.gripper_open_pc])
		self.gripper_open_pc = np.asarray(self.gripper_open_pc.points)

	def control(self):
		self.model.eval()

		# initialize data
		data = {}

		# control
		iteration = 0
		while True:

			# iteration number
			iteration += 1
			print(f"************************* Iteration {iteration} *************************")

			# obtain partially observed point cloud data			
			if self.realworld_debug:
				pc = np.load('test.npy')
			
			elif self.realworld:
				server = Listener(self.ip, self.port)
				print(f'waiting image ...')
				image = server.recv_vision()
				image = np.array(image)
				print(f'recieved image shape: {image.shape}')
				pc = image.T
				del image
				np.save('test', pc)

			else:
				# load point cloud from simulator, add noise and normalize
				pc = self.env._get_pc().T   

			# set the number of point cloud
			pc = pc[:3, :]
			if pc.shape[1] > 2048:
				pc = pc[:, random.sample(range(pc.shape[1]), 2048)]
			else:
				pc = upsample_pointcloud(2048, pc.T).T

			# normalize point cloud
			pc_input = deepcopy(pc)
			if (not self.realworld) and (not self.realworld_debug):
				pc_input = noise_augmentation(pc_input, noise_std=0.001)
			pc_input, mean_xyz_global, diagonal_len_global = normalize_pointcloud(pc_input)

			# sent to torch from numpy
			data['pc'] = torch.Tensor(pc_input).unsqueeze(0)
			data['mean_xyz'] = torch.Tensor(mean_xyz_global).unsqueeze(0)
			data['diagonal_len'] = torch.tensor(diagonal_len_global).unsqueeze(0)

			# segmentation
			data = self.model.segmentation(data)

			# segment result calibration
			data = self.model.calibration(data)

			# recognition
			data = self.model.recognition(data)

			# unnormalize data
			data = self.model.unnormalize(data)

			# set the largest object to target object
			data = self.target_selector(data, self.target_object)

			# delete deformation
			data['parameters'] = data['parameters'][:, :, :5]

			# sent to numpy from torch
			Ts_numpy = data['Ts_pred'].squeeze(0).cpu().numpy()
			parameters_numpy = data['parameters'].squeeze(0).cpu().numpy()

			if self.visualize_recognized_objects:
				objects_renderer_with_pc(pc, Ts_numpy, parameters_numpy)

			# calculate score
			if self.criterior == 'grasp':
				current_score, valid_grasp_SE3s = self.criterior_score(Ts_numpy, parameters_numpy, gripper_pc=self.gripper_open_pc, get_valid_grasp_poses=True)
			else:
				current_score = self.criterior_score(Ts_numpy, parameters_numpy, gripper_pc=self.gripper_open_pc)
			print(f"current score: {current_score}")

			if current_score == 'failed':
				print('No valid grasp pose is available (grasp planning failed).')
				if self.realworld:
					dict_data = {'task': self.criterior,
								'valid_grasp_SE3': 'failed',
								'list_sq_poses': Ts_numpy,
								'list_sq_parameters': parameters_numpy,
								}
					server.send_grasp(dict_data)
					print('sending completed.')
					server.close_connection()	
				self.task_success = 'failed'
				break			
			
			# terminate condition
			if self.terminate_condition(current_score=current_score, iteration=iteration) or (iteration > 10):
				if (self.criterior == 'grasp') and (current_score == 1):	
					if self.realworld:
						dict_data = {'task': self.criterior,
								'valid_grasp_SE3': valid_grasp_SE3s[0],
								'list_sq_poses': Ts_numpy,
								'list_sq_parameters': parameters_numpy,
								}
						server.send_grasp(dict_data)
						print('sending completed.')
						server.close_connection()
					self.task_success = 'success'
				elif (self.criterior == 'grasp') and (current_score == 0):
					if self.realworld:
						dict_data = {'task': self.criterior,
							'valid_grasp_SE3': 'failed',
							'list_sq_poses': Ts_numpy,
							'list_sq_parameters': parameters_numpy,
							}
						server.send_grasp(dict_data)
						print('sending completed.')
						server.close_connection()
					self.task_success = 'failed'    					
				else:
					self.task_success = 'success'
				break

			# sample action
			actions_all, actions_valid, scenes, arranged_parameters, scores = self.action_sampler(Ts_numpy, parameters_numpy, current_score)

			# none of the pushing action is available
			if len(actions_valid) == 0:
				if self.realworld:
					print('None of action is available')
					if self.realworld:
						dict_data = {
							'task': self.criterior,
							'status': False,
							'list_sq_poses': Ts_numpy,
							'list_sq_parameters': parameters_numpy,
							'list_qd': None,
							'list_acc': None,
							'list_vel': None,
						}
				print('all of the sampled actions are not better than current sate.')	
				break

			# sort action w.r.t. score values
			sorted_actions, sorted_indices, sorted_scores = self.action_sorter(actions_valid, scores)

			# action execution
			for action_idx in range(len(sorted_actions)):
				best_action_first = sorted_actions[action_idx][0]
				if self.realworld:
					best_action_first[3] += self.table_offest
				best_score = sorted_scores[action_idx]

				# execute action            
				if self.realworld:
					success_planning, joint_stats_list = self.env.sim.push_action_for_real_world(
															position=[best_action_first[1], best_action_first[2], best_action_first[3]],
															rotation_angle=best_action_first[0],
															speed=0.05,
															distance=0.10
														)
					if success_planning:
						if self.realworld:
							# compute desired action
							dict_data = {
								'task': self.criterior,
								'status': True,
								'list_sq_poses': Ts_numpy,
								'list_sq_parameters': parameters_numpy,
								'list_qd': joint_stats_list,
								'list_acc': [0.5] + [0.1] * len(joint_stats_list) + [0.5],
								'list_vel': [0.5] + [0.1] * len(joint_stats_list) + [0.5],
							}
							if not os.path.exists('real_experiment_data/'):
								os.makedirs('real_experiment_data/')
							np.save(f'real_experiment_data/{self.criterior}_{self.run_id}_iter_{iteration}_Ts', Ts_numpy)
							np.save(f'real_experiment_data/{self.criterior}_{self.run_id}_iter_{iteration}_parameters', parameters_numpy)
									
							# execute action
							print(f'planning successed with score {best_score}, sending dict data...')
							server.send_grasp(dict_data)
							print('sending completed.')
							server.close_connection()
							break
					else:
						print('planning failed. try next best action (real).')
						continue

				else:
					old_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.env.object_ids])
					old_ors = np.array([p.getBasePositionAndOrientation(object_id)[1] for object_id in self.env.object_ids])
					success_planning_1 = self.env.sim.down_action(
						position=[best_action_first[1], best_action_first[2], best_action_first[3]],
						rotation_angle=best_action_first[0]
					)
					success_planning_2 = self.env.sim.push_action(
						position=[best_action_first[1], best_action_first[2], best_action_first[3]],
						rotation_angle=best_action_first[0],
						speed=0.05,
						distance=0.10
					)
					self.env.sim.robot_go_home()   

					if success_planning_1 and success_planning_2:
						print(f'planning successed with score {best_score}')
						break
					else:
						print('planning failed. try next best action (simulation).')
						self.env.sim.robot_go_home()
						self.env._reset_objects(old_pos, old_ors)
						continue 

			# if all pushing actions is not available
			if not ((self.realworld and success_planning) or ((not self.realworld) and success_planning_1 and success_planning_2)):
				print('all actions are failed!')
				if self.realworld:
					dict_data = {
						'task': self.criterior,
						'status': False,
						'list_sq_poses': Ts_numpy,
						'list_sq_parameters': parameters_numpy,
						'list_qd': None,
						'list_acc': None,
						'list_vel': None,
					}
					server.send_grasp(dict_data)
					print('sending completed.')
					server.close_connection()
				self.task_success = 'failed'
				break					
		
		# wait
		time.sleep(3)
		print(f"Task {self.criterior} {self.task_success}!!")

	def target_selector(self, data, method):
		Ts = data['Ts_pred']
		parameters = data['parameters']
		num_primitives = parameters.shape[1]

		if num_primitives == 1:
			pass
		else:            
			if method == 'largest':
				idxs = list(range(num_primitives))
				
				vols = parameters[0, :, 0] * parameters[0, :, 1] * parameters[0, :, 2]
				idx_largest = torch.argmax(vols).item()
				idxs.remove(idx_largest)
				
				Ts = Ts[:, [idx_largest] + idxs]
				parameters = parameters[:, [idx_largest] + idxs]

			elif method == 'smallest':
				idxs = list(range(num_primitives))
				
				vols = parameters[0, :, 0] * parameters[0, :, 1] * parameters[0, :, 2]
				idx_smallest = torch.argmin(vols).item()
				idxs.remove(idx_smallest)
				
				Ts = Ts[:, [idx_smallest] + idxs]
				parameters = parameters[:, [idx_smallest] + idxs]

			elif method == 'cylinder':
				idxs = list(range(num_primitives))
				e2 = parameters[0, :, 4]
				idx_largest = torch.argmax(e2).item()
				idxs.remove(idx_largest)
				
				Ts = Ts[:, [idx_largest] + idxs]
				parameters = parameters[:, [idx_largest] + idxs]

			data['Ts_pred'] = Ts
			data['parameters'] = parameters

		return data

	def action_sampler(self, Ts, parameters, current_score):
		num_primitives = parameters.shape[0]
		
		# parameters
		if self.criterior == 'moving_interactive':
			objects = np.random.choice(num_primitives - 1, size=self.sample_num * self.action_horizon, replace=True) + 1
		else:
			objects = np.random.choice(num_primitives, size=self.sample_num * self.action_horizon, replace=True)

		# choose direction
		if self.realworld:
			direction = np.random.choice(self.num_directions - 2, size=self.sample_num * self.action_horizon, replace=True) + 2
		else:
			direction = np.random.choice(self.num_directions, size=self.sample_num * self.action_horizon, replace=True)
			
		# choose z coordinate
		position_z = Ts[objects, 2, 3]
		obj_height = np.clip((2 * position_z - self.env.workspace_bounds[2, 0]) - 0.02, self.env.workspace_bounds[2, 0], 1)
		z_max_heights = np.floor(((obj_height - self.env.workspace_bounds[2, 0]) / (self.env.workspace_bounds[2, 1] - self.env.workspace_bounds[2, 0])) * (self.num_z - 1)).astype(np.int)
		z = np.array([np.random.choice(i) for i in z_max_heights + 1])

		# to prevent object fall-over
		z[z >= 2] = 1

		# calculate diagonal length and choose initial distance
		min_lens = np.zeros(num_primitives)
		for i in range(num_primitives):
			min_lens[i] = np.min(parameters[i, :2])
		# init_dists = np.ceil(min_lens * 100) / 100 + 0.01
		init_dists = np.ceil(min_lens * 100) / 100 + 0.03
		init_dists = init_dists[objects] + 0.02 * np.random.randint(4, size=len(objects))
		
		# sample random action sequences
		direction_angle = direction / self.num_directions * 2 * np.pi
		push_initial_delta = - np.asarray([np.cos(direction_angle), np.sin(direction_angle)]).T * np.expand_dims(init_dists, 1)
		if self.realworld or self.realworld_debug:
			if not self.simulator:
				z_heights = (z / (self.num_z - 1)) * (self.env.workspace_bounds[2, 1] - self.env.workspace_bounds[2, 0]) + 0.263
			else:
				z_heights = (z / (self.num_z - 1)) * (self.env.workspace_bounds[2, 1] - self.env.workspace_bounds[2, 0]) + 0.273 - 0.01
		else:
			z_heights = (z / (self.num_z - 1)) * (self.env.workspace_bounds[2, 1] - self.env.workspace_bounds[2, 0]) + self.env.workspace_bounds[2, 0]
		actions = np.concatenate((np.expand_dims(direction_angle, 1), push_initial_delta, np.expand_dims(z_heights, 1)), axis=1)
		
		actions = actions.reshape(self.sample_num, self.action_horizon, 4)
		objects = objects.reshape(self.sample_num, self.action_horizon)

		# check collision
		actions_valid = []
		scenes = []
		scores = []
		arranged_parameters = []
		for seq_idx in range(len(actions)):
			Ts_prev = deepcopy(Ts)
			score_prev = current_score
			Ts_seq = np.zeros((self.action_horizon, num_primitives, 4, 4), dtype=np.float32)
			flag_stop = False

			for t_idx in range(self.action_horizon):

				# action at the sequence and the timestep
				a = actions[seq_idx, t_idx, :]
				a[1:3] += Ts_prev[objects[seq_idx, t_idx], :2, 3]

				# gripper SE3
				gripper_position = np.asarray([a[1], a[2], a[3] + (self.env.sim.gripper_height + self.env.sim.gripper_length)]) \
					- np.tan(self.env.sim.gripper_tilt) * (self.env.sim.gripper_height + self.env.sim.gripper_length) * np.asarray([np.cos(a[0]), np.sin(a[0]), 0])
				gripper_orientation = p.getQuaternionFromEuler([0, np.pi - self.env.sim.gripper_tilt, a[0]])
				gripper_SE3 = get_SE3s(quats_to_matrices(np.asarray(gripper_orientation)), gripper_position)

				# check collision
				if check_collision([Ts_prev, parameters], 
									gripper_SE3, 
									self.gripper_closed_pc, 
									visualize=self.visualize_collision,
									object_collision_scale=self.object_collision_scale):
					flag_stop = True
					break

				# motion prediction
				if not self.simulator:
					data = {}
					data['Ts_pred'] = torch.Tensor(Ts_prev).unsqueeze(0).to(self.device)
					data['action'] = torch.Tensor(a).unsqueeze(0).to(self.device)
					data['parameters'] = torch.Tensor(parameters[:, :5]).unsqueeze(0).to(self.device)
					data['num_primitives_batch'] = torch.Tensor([len(parameters)]).int()

					data = self.model.motion_prediction(data)

					motions = data['motion_preds'].detach().squeeze(0).cpu().numpy()
					Ts_prev = data['Ts_pred'].detach().squeeze(0).cpu().numpy()
					parmaeters_prev = data['parameters'].squeeze(0).cpu().numpy()

					# calculate motion SE3
					if motions.shape[1] == 3:    # 2D
						motion_SE3s = np.array([[[np.cos(motion[2]), -np.sin(motion[2]), 0, motion[0]],
												[np.sin(motion[2]), np.cos(motion[2]), 0, motion[1]],
												[0, 0, 1, 0],
												[0, 0, 0, 1]] for motion in motions])
					elif motions.shape[1] == 7:  # 3D
						motion_SE3s = get_SE3s(quats_to_matrices(motions[:, 3:7]), motions[:, :3])
					else:
						raise ValueError("check the dimension of the motion")

					# update SE3
					Ts_seq[t_idx] = Ts_prev @ motion_SE3s

				else:
					parmaeters_prev = parameters
					positions_old = Ts_prev[:, :3, 3]
					orientations_old = matrices_to_quats(Ts_prev[:, :3, :3])
					meshes = mesh_generator_recog_pred(np.concatenate([positions_old, orientations_old, parameters], axis=1), transform=False)

					for i, (mesh, position_old, orientation_old) in enumerate(zip(meshes, positions_old, orientations_old)):
						filename = f"primitive {i}.stl"

						o3d.io.write_triangle_mesh(filename, mesh)

						collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=filename)
						obj_id = p.createMultiBody(0.05, collision_id, -1, position_old, orientation_old)

						self.model.env.object_ids += [obj_id]

						os.remove(filename)

					push_initial = a[1:]
					direction_angle = a[0]

					self.model.env.sim.down_action(position=push_initial, rotation_angle=direction_angle)
					self.model.env.sim.push_action(position=push_initial, rotation_angle=direction_angle, speed=0.05, distance=0.10)

					new_po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.model.env.object_ids]

					positions_new = np.array([new_po_or[0] for new_po_or in new_po_ors])
					orientations_new = np.array([new_po_or[1] for new_po_or in new_po_ors])

					Ts_seq[t_idx] = get_SE3s(quats_to_matrices(orientations_new), positions_new)

					for obj_id in self.model.env.object_ids:
						p.removeBody(obj_id)
					self.model.env.object_ids = []

				# # check if objects are on table
				bool_on_table = np.logical_and(Ts[:, 1, 3] >= -self.table_size[1], Ts[:, 1, 3] <= self.table_size[1]).all()
				if not bool_on_table:
					flag_stop = True
					break

				# calculate score
				score = self.criterior_score(Ts_seq[t_idx], parameters, gripper_pc=self.gripper_open_pc)
				if score < score_prev or (self.criterior == 'grasp' and score_prev == 1):
					flag_stop = True
					break
				
				Ts_prev = deepcopy(Ts_seq[t_idx])
				score_prev = deepcopy(score)

			if flag_stop:
				if t_idx != 0:
					actions_valid.append(actions[seq_idx, :t_idx])
					scenes.append(Ts_seq[:t_idx])
					scores.append(score_prev)
					arranged_parameters.append(parmaeters_prev)
			else:
				actions_valid.append(actions[seq_idx])
				scenes.append(Ts_seq)
				scores.append(score)
				arranged_parameters.append(parmaeters_prev)

		return actions, actions_valid, scenes, arranged_parameters, scores

	def criterior_score(self, Ts_pri, parameters_pri, gripper_pc=None, get_valid_grasp_poses=False):
		if self.criterior == 'moving':
			output = moving_criterior([Ts_pri, parameters_pri], self.goal_positions)
		if self.criterior == 'moving_interactive':
			output = moving_interactive_criterior([Ts_pri, parameters_pri], self.goal_position)
		if self.criterior == 'singulation':
			output = singulation_criterior([Ts_pri, parameters_pri], self.tau)
		elif self.criterior == 'grasp':
			table_size_virtual = deepcopy(self.table_size)
			table_size_virtual[0] += self.table_x_augment
			output = grasp_criterior([Ts_pri, parameters_pri], 
									gripper_pc, 
									table=[table_size_virtual, np.array(self.env.sim.high_table_position)], 
									visualize=self.visualize_grasp_candidates, 
									get_valid_grasp_poses=get_valid_grasp_poses,
									use_only_topdown=self.use_only_topdown,
									use_ik=self.use_ik)

		return output

	def terminate_condition(self, **kwargs):
		if self.criterior == 'moving':
			terminate = (kwargs['current_score'] >= -0.001)
		elif self.criterior == 'moving_interactive':
			terminate = (kwargs['current_score'] >= -0.001)
		elif self.criterior == 'singulation':
			terminate = (kwargs['current_score'] >= -0.01)
		elif self.criterior == 'grasp':
			terminate = (kwargs['current_score'] == 1)

		return terminate

	def action_selector(self, actions, scores):

		if self.criterior == 'grasp':
			success_indices = np.where(np.array(scores) == 1.0)[0]
			if len(success_indices) == 0:
				best_idx = np.random.choice(len(actions))
				best_action = actions[best_idx]
				best_score = scores[best_idx]
			else:
				for i, success_idx in enumerate(success_indices):
					if i == 0:
						best_idx = deepcopy(success_idx)
						best_action = deepcopy(actions[success_idx])
					else:
						if len(actions[success_idx]) < len(best_action):
							best_idx = deepcopy(success_idx)
							best_action = deepcopy(actions[success_idx])
				best_score = scores[best_idx]
				
		else:
			best_idx = np.argmax(np.array(scores))
			best_action = actions[best_idx]
			best_score = scores[best_idx]

		print(f"score of the best action: {best_score}")

		return best_action, best_idx

	def action_sorter(self, actions, scores):
	
		if self.criterior == 'grasp':
			success_indices = np.where(np.array(scores) == 1.0)[0]
			fail_indices = np.where(np.array(scores) == 0.0)[0]
			if len(success_indices) == 0:
				sorted_indices = range(len(actions))
				sorted_actions = deepcopy(actions)
				sorted_scores = deepcopy(scores)
			else:
				len_action_list = []
				for i, success_idx in enumerate(success_indices):
					len_action = len(actions[success_idx])
					len_action_list.append(len_action)
				sorted_len_action = success_indices[np.argsort(np.array(len_action_list))]
				sorted_indices = np.concatenate((sorted_len_action, fail_indices))
				sorted_actions = [actions[i] for i in sorted_indices]
				sorted_scores = [scores[i] for i in sorted_indices]
				
		else:
			sorted_indices = np.argsort(-np.array(scores))
			sorted_actions = [actions[i] for i in sorted_indices]
			sorted_scores = [scores[i] for i in sorted_indices]

		return sorted_actions, sorted_indices, sorted_scores