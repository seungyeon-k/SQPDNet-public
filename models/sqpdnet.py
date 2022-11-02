import torch
import open3d as o3d
from copy import deepcopy
import torch.nn.functional as F
from collections import OrderedDict

from .segmenation_network import SegmentationNetwork
from .dgcnn import DGCNN
from .sqnet import SuperquadricNetwork
from .motion_prediction_network import MotionPredictionNetwork
from control.control_env import ControlSimulationEnv
from loss.segmentation_loss import hungarian_matching, batch_reordering
from loader.recognition_dataset import normalize_pointcloud as normalize_pointcloud
from functions.utils_torch import quats_to_matrices_torch, get_SE3s_torch, matrices_to_quats_torch
from loader.baseline_dataset import coordinate_processing, SE2_from_SE3

class SuperquadricMotionPredictionNetwork(torch.nn.Module):
	def __init__(self, backbone, enable_gui=False, **kwargs):
		super(SuperquadricMotionPredictionNetwork, self).__init__()

		# control environment
		self.env = ControlSimulationEnv(enable_gui=enable_gui)

		modules = []
		cfg_modules = []

		if 'seg_module' in kwargs:
			# segmentation module
			cfg_seg_module = kwargs['seg_module']
			cfg_seg_backbone = cfg_seg_module.pop('backbone')
			seg_backbone = get_backbone_instance(cfg_seg_backbone['arch'])(**cfg_seg_backbone)
			self.seg_module = SegmentationNetwork(seg_backbone, **cfg_seg_module)
			modules += [self.seg_module]
			cfg_modules += [cfg_seg_module]
			self.calibration_k = kwargs['calibration_k']
			self.num_pts_recog = kwargs['num_pts_recog']

		if 'recog_module' in kwargs:
			# recognition module
			cfg_recog_module = kwargs['recog_module']
			cfg_recog_backbone = cfg_recog_module.pop('backbone')
			recog_backbone = get_backbone_instance(cfg_recog_backbone['arch'])(**cfg_recog_backbone)
			if cfg_recog_module['arch'] == 'sqnet':
				self.recog_module = SuperquadricNetwork(recog_backbone, **cfg_recog_module)
			modules += [self.recog_module]
			cfg_modules += [cfg_recog_module]

		if 'motion_module' in kwargs:
			# motion prediction module
			cfg_motion_module = kwargs['motion_module']
			self.motion_module = MotionPredictionNetwork(None, **cfg_motion_module)
			modules += [self.motion_module]
			cfg_modules += [cfg_motion_module]

		# load pretrained models
		for module, cfg_module in zip(modules, cfg_modules):
			if cfg_module.get('pretrained', None):
				pretrained_model_path = cfg_module['pretrained']

				ckpt = torch.load(pretrained_model_path, map_location='cpu')['model_state']
				new_ckpt = OrderedDict()

				flag_replace = False
				for key, val in ckpt.items():
					if 'motion_module.' in key:
						key = key.replace('motion_module.', '')
						new_ckpt[key] = val
						flag_replace = True
				if flag_replace:
					ckpt = new_ckpt

				module.load_state_dict(ckpt)

	def train_step(self, data, optimizer, loss_function, device, clip_grad=None, **kwargs):
		x_scene_batch = data['x_scene'].to(device)
		a_scene_batch = data['a_scene'].to(device)
		y_scene_batch = data['y_scene'].to(device)

		x_scene = x_scene_batch.reshape(-1, x_scene_batch.shape[2], x_scene_batch.shape[3])
		a_scene = a_scene_batch.reshape(-1, a_scene_batch.shape[2])
		y_scene = y_scene_batch.reshape(-1, y_scene_batch.shape[2])

		real_idxs = x_scene[:, 0, 0] == 1

		x_scene = x_scene[real_idxs]
		a_scene = a_scene[real_idxs]
		y_scene = y_scene[real_idxs]

		preds = self.motion_module(x_scene, a_scene)

		loss = loss_function(y_scene, preds)

		optimizer.zero_grad()

		if clip_grad is not None:
			torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
		loss.backward()
		optimizer.step()

		return {'loss': loss.item(),
				'scene': x_scene.detach().cpu().numpy(),
				'action': a_scene.detach().cpu().numpy(),
				'motion_pred': preds.detach().cpu().numpy(),
				'motion_gt': y_scene.detach().cpu().numpy()
		}

	def validation_step(self, data, loss_function, device, **kwargs):
		x_scene_batch = data['x_scene'].to(device)
		a_scene_batch = data['a_scene'].to(device)
		y_scene_batch = data['y_scene'].to(device)

		x_scene = x_scene_batch.reshape(-1, x_scene_batch.shape[2], x_scene_batch.shape[3])
		a_scene = a_scene_batch.reshape(-1, a_scene_batch.shape[2])
		y_scene = y_scene_batch.reshape(-1, y_scene_batch.shape[2])

		real_idxs = x_scene[:, 0, 0] == 1

		x_scene = x_scene[real_idxs]
		a_scene = a_scene[real_idxs]
		y_scene = y_scene[real_idxs]

		preds = self.motion_module(x_scene, a_scene)

		loss = loss_function(y_scene, preds)

		return {'loss': loss.item(),
				'scene': x_scene.detach().cpu().numpy(),
				'action': a_scene.detach().cpu().numpy(),
				'motion_pred': preds.detach().cpu().numpy(),
				'motion_gt': y_scene.detach().cpu().numpy()
		}

	def segmentation(self, data):
		# test
		pc = data['pc'].to(self.device)

		seg_pred = self.seg_module(pc)
		seg_pred = seg_pred.detach()

		data['pc'] = pc
		data['seg_pred'] = seg_pred

		return data

	def reordering(self, data):
		seg_pred = data['seg_pred']
		labels = data['labels'].to(self.device)

		labels = F.one_hot(labels.long(), num_classes=self.seg_module.num_classes)

		seg_pred = torch.clamp(seg_pred, min=1e-7, max=1 - 1e-7)

		matching_idxs = hungarian_matching(seg_pred, labels)
		seg_pred = batch_reordering(seg_pred, matching_idxs)

		data['seg_pred'] = seg_pred

		return data

	def calibration(self, data):
		pc_batch = data['pc']
		seg_pred_batch = data['seg_pred']

		num_classes = seg_pred_batch.shape[2]

		for batch_idx, (pc, seg_pred) in enumerate(zip(pc_batch, seg_pred_batch)):
			pc = pc.cpu().numpy().T
			seg_pred = seg_pred.argmax(axis=1)

			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(pc)
			partial_pcd_tree = o3d.geometry.KDTreeFlann(pcd)

			seg_pred_raw = torch.clone(seg_pred)

			for i in range(len(seg_pred_raw)):
				[_, idxs, _] = partial_pcd_tree.search_knn_vector_3d(pcd.points[i], self.calibration_k)
				seg_pred_near = seg_pred_raw[idxs]
				seg_pred_near = seg_pred_near[seg_pred_near != 0]
				seg_pred[i] = torch.bincount(seg_pred_near).argmax() if len(seg_pred_near) else seg_pred[i]

			seg_pred_1hot = torch.eye(num_classes)[seg_pred]

			seg_pred_batch[batch_idx] = seg_pred_1hot

		data['seg_pred'] = seg_pred_batch

		return data

	def recognition(self, data):
		pc_batch = data['pc']
		seg_pred_batch = data['seg_pred'].argmax(dim=2)

		batch_size = len(pc_batch)
		max_num_primitives = max([len(torch.unique(seg_pred)) for seg_pred in seg_pred_batch])
		recog_preds_batch = torch.zeros(batch_size, max_num_primitives, self.recog_module.output_dim_total).to(self.device)
		mean_xyz_pris_batch = torch.zeros(batch_size, max_num_primitives, 3, 1)
		diagonal_len_pris_batch = torch.zeros(batch_size, max_num_primitives)

		seg_ids_batch = torch.zeros(batch_size, self.seg_module.num_classes)
		num_primitives_batch = torch.zeros(batch_size, dtype=int)
		
		for batch_idx in range(len(pc_batch)):
			pc_scene = pc_batch[batch_idx]
			seg_pred = seg_pred_batch[batch_idx]

			seg_ids = torch.unique(seg_pred)
			seg_ids_batch[batch_idx, :len(seg_ids)] = seg_ids
			num_primitives_batch[batch_idx] = len(seg_ids)

			pc = []

			for primitive_idx, seg_id in enumerate(seg_ids):
				# get primitive-wise point cloud
				pc_pri = pc_scene[:, seg_pred == seg_id]
				pc_pri = torch.cat([pc_pri, torch.ones(1, pc_pri.shape[1]).to(self.device)])

				pc_surround = pc_scene[:, seg_pred != seg_id]
				pc_surround = torch.cat([pc_surround, torch.zeros(1, pc_surround.shape[1]).to(self.device)])

				pc_overall = torch.cat([pc_pri, pc_surround], dim=1)

				# normalize point cloud
				pc_overall, mean_xyz_pri, diagonal_len_pri = normalize_pointcloud(pc_overall.cpu().numpy(), pc_pri.cpu().numpy())

				pc += [torch.Tensor(pc_overall)]

				mean_xyz_pris_batch[batch_idx, primitive_idx] = torch.Tensor(mean_xyz_pri)
				diagonal_len_pris_batch[batch_idx, primitive_idx] = diagonal_len_pri.item()

			pc = torch.stack(pc, dim=0).to(self.device)

			# test
			recog_preds = self.recog_module(pc)
			recog_preds = recog_preds.detach()

			recog_preds_batch[batch_idx] = recog_preds

		data['recog_preds'] = recog_preds_batch
		data['seg_ids_batch'] = seg_ids_batch
		data['num_primitives_batch'] = num_primitives_batch
		data['mean_xyz_pris'] = mean_xyz_pris_batch
		data['diagonal_len_pris'] = diagonal_len_pris_batch

		return data

	def unnormalize(self, data):
		pc = data['pc'].permute([0, 2, 1])
		recog_preds = data['recog_preds']
		mean_xyz_objects = data['mean_xyz_pris'].to(self.device).squeeze(-1)
		diagonal_len_objects = data['diagonal_len_pris'].to(self.device).unsqueeze(-1)
		mean_xyz_global = data['mean_xyz'].to(self.device).permute([0, 2, 1])
		diagonal_len_global = data['diagonal_len'].to(self.device).unsqueeze(-1).unsqueeze(-1)

		# decompose output
		positions = deepcopy(recog_preds[:, :, :3])
		orientations = deepcopy(recog_preds[:, :, 3:7])
		parameters = deepcopy(recog_preds[:, :, 7:])

		# revise position and parameters primitive-wisely
		positions = positions * diagonal_len_objects + mean_xyz_objects
		parameters[:, :, :3] *= diagonal_len_objects

		# revise position and parameters globaly
		pc = pc * diagonal_len_global + mean_xyz_global
		positions = positions * diagonal_len_global + mean_xyz_global
		parameters[:, :, :3] *= diagonal_len_global

		parameters = parameters.permute([0, 2, 1])

		# get before-action-SE3 predictions
		Rs = quats_to_matrices_torch(orientations.reshape(-1, orientations.shape[2]))
		Ts = get_SE3s_torch(Rs, positions.reshape(-1, positions.shape[2]))
		Ts = Ts.reshape(recog_preds.shape[0], recog_preds.shape[1], Ts.shape[1], Ts.shape[2])

		data['pc'] = pc.permute([0, 2, 1])
		data['Ts_pred'] = Ts
		data['parameters'] = parameters.permute([0, 2, 1])

		return data

	def motion_prediction(self, data):
		Ts_batch = data['Ts_pred']
		action = data['action'].to(self.device)
		parameters_batch = data['parameters']
		num_primitives_batch = data['num_primitives_batch']

		batch_size, max_num_primitives = Ts_batch.shape[:2]
		if self.motion_module.motion_dim == '2D':
			motion_preds_batch = torch.zeros(batch_size, max_num_primitives, 3).to(self.device)
		elif self.motion_module.motion_dim == '3D':
			motion_preds_batch = torch.zeros(batch_size, max_num_primitives, 7).to(self.device)

		# get action position and vector
		action_positions = action[:, 1:]
		action_angles = action[:, 0]
		action_vectors = torch.stack([torch.cos(action_angles), torch.sin(action_angles), torch.zeros(batch_size).to(self.device)]).T

		# process primitive poses to revise x and z axes
		for batch_idx in range(batch_size):
			parameters = parameters_batch[batch_idx]
			action_vector = action_vectors[batch_idx]
			action_position = action_positions[batch_idx]
			num_primitives = num_primitives_batch[batch_idx].item()

			x = torch.zeros(num_primitives, 13, num_primitives).to(self.device)
			a = torch.zeros(num_primitives, 5).to(self.device)

			for primitive_idx in range(num_primitives):
				# load primitive pose and parameters
				parameter_before = parameters_batch[batch_idx, primitive_idx]
				T_before = Ts_batch[batch_idx, primitive_idx].cpu().numpy()

				# revise x and z axes
				parameter_after, T_after = coordinate_processing(parameter_before, T_before, action_vector.cpu().numpy())

				# replace to processed data
				parameters_batch[batch_idx, primitive_idx] = parameter_after
				Ts_batch[batch_idx, primitive_idx] = torch.Tensor(T_after)

			for primitive_idx in range(num_primitives):
				Ts = Ts_batch[batch_idx, :num_primitives]
				parameters = parameters_batch[batch_idx, :num_primitives, :5]

				# inverse pose of the target object
				# T_inverse = torch.inverse(Ts[primitive_idx])          
				C = SE2_from_SE3(Ts[primitive_idx].cpu().numpy())
				C_inverse = torch.inverse(torch.tensor(C)).to(torch.float32).to(self.device)  

				# compute inverse SE3 of the selected primitive and reorder
				primitive_idxs = list(range(num_primitives))
				primitive_idxs.remove(primitive_idx)
				Ts_centric = Ts[[primitive_idx] + primitive_idxs]
				parameters_centric = parameters[[primitive_idx] + primitive_idxs]

				# inversely transform with pose of the selected primitive
				Ts_wrt_ego = C_inverse @ Ts_centric 

				# compute positions and orientations w.r.t. the selected primitive
				positions_wrt_ego = Ts_wrt_ego[:, :3, 3]
				orientations_wrt_ego = matrices_to_quats_torch(Ts_wrt_ego[:, :3, :3])

				# express action w.r.t. the selected primitive
				action_position_wrt_ego = (C_inverse @ torch.cat([action_position, torch.ones(1).to(self.device)]))[:-1]    
				action_vector_wrt_ego = C_inverse[:3, :3] @ action_vector                                                  
				action_angle_wrt_ego = torch.atan2(action_vector_wrt_ego[1], action_vector_wrt_ego[0]).to(self.device)
				action_angle_wrt_ego = torch.Tensor([torch.cos(action_angle_wrt_ego), torch.sin(action_angle_wrt_ego)]).to(self.device)

				x[primitive_idx] = torch.cat([torch.ones(1, num_primitives).to(self.device), positions_wrt_ego.T, orientations_wrt_ego.T, parameters_centric.T[:5]])
				a[primitive_idx] = torch.cat([action_position_wrt_ego, action_angle_wrt_ego])

			# test
			motion_preds = self.motion_module(x, a)
			motion_preds = motion_preds.detach() if not self.training else motion_preds

			motion_preds_batch[batch_idx, :num_primitives] = motion_preds

		data['motion_preds'] = motion_preds_batch
		data['Ts_pred'] = Ts_batch
		data['parameters'] = parameters_batch

		return data

def get_backbone_instance(name):
	try:
		return {
			'dgcnn': DGCNN,
		}[name]
	except:
		raise (f"Backbone {name} not available")
