import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import h5py
import numpy as np
from copy import deepcopy

from functions.utils import get_SE3s, matrices_to_quats, quats_to_matrices, exp_so3
from functions.point_clouds import noise_augmentation

class BaselineDataset(Dataset):
    def __init__(self, split, cfg_data):
        self.model = cfg_data['loader']

        if self.model == 'sqpdnet':
            self.num_primitives = cfg_data['num_primitives']
            self.motion_dim = cfg_data['motion_dim']
            self.num_classes = cfg_data.get('num_classes', 5)
            self.num_points = cfg_data.get('num_points', 2048)
            self.seg_noise_augment = cfg_data.get('seg_noise_augment', True)
            self.seg_noise_std = cfg_data.get('seg_noise_std', 0.001)
            self.recognized_parameters = cfg_data.get('recognized_parameters', False)
            self.parameters_noise_augment = cfg_data.get('parameters_noise_augment', True)
            self.motion_noise_augment = cfg_data.get('motion_noise_augment', True)
            self.motion_noise_std = cfg_data.get('motion_noise_std', 0.02)
        elif self.model == 'dsr-net':
            self.num_directions = cfg_data['num_directions']

        data_path_list = cfg_data['paths']

        self.file_list = []
        for data_path in data_path_list:
            self.file_list += [osp.join(osp.join(data_path, split), file) for file in os.listdir(osp.join(data_path, split))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file = h5py.File(filename, 'r')

        if self.model == 'sqpdnet':
            data = self.get_sqpdnet_data(file)
        elif self.model == 'dsr-net':
            data = self.get_dsr_net_data(file)
        elif self.model == ('se3-nets' or 'se3-pose-nets'):
            data = self.get_se3_nets_data(file)
        else:
            raise TypeError(f"No model type {self.model}. Supported model types are 'sqpdnet', 'dsr-net', 'se3-nets' and 'se3-pose-nets")

        data['mask_3d_new'] = file['mask_3d_new'][()]
        data['mask_2d_old'] = file['mask_2d_old'][()].reshape(file['depth_image_old'].shape)
        data['mask_2d_new'] = file['mask_2d_new'][()].reshape(file['depth_image_old'].shape)
        data['depth_image_old'] = file['depth_image_old'][()]
        data['depth_image_new'] = file['depth_image_new'][()]

        return data

    def get_sqpdnet_data(self, file):
        # check the number of point
        assert (file['pc_down_wo_plane'].shape[0] == self.num_points), \
            f"Number of points in point cloud data should be {self.num_points}, but is {file['pc_down_wo_plane'].shape[0]}"
        assert (file['labels_down_wo_plane'].shape[0] == self.num_points), \
            f"Number of points in label data should be {self.num_points}, but is {file['labels_down_wo_plane'].shape[0]}"

        # check the number classes
        assert max(file['labels_down_wo_plane'][()])+1 <= self.num_classes, \
            f"Number of classes should be {max(file['labels_down_wo_plane'][()])+1}, but is {self.num_classes}"
        
        # check the number of objects and primitives
        assert (len(eval(file['object_info'][()])) <= self.num_primitives), \
            f"Number of primitives should be {len(eval(file['object_info'][()]))}, but is {self.num_primitives}"

        pc = file['pc_down_wo_plane'][()].transpose()
        labels = file['labels_down_wo_plane'][()]
        object_info = eval(file['object_info'][()])
        positions_old = file['positions_old'][()]
        orientations_old = file['orientations_old'][()]
        positions_new = file['positions_new'][()]
        orientations_new = file['orientations_new'][()]
        action_coord = file['action_coord'][()]
        pc_gt = file['pc_gt'][()].transpose()

        num_objects = len(object_info)

        # get DSQ parameters
        if self.recognized_parameters:
            parameters = file['recognized_parameters'][()].T
        else:
            parameters = get_DSQ_parameters(object_info)

        # get before-action-SE3s
        Rs_old = quats_to_matrices(orientations_old)
        Ts_old = get_SE3s(Rs_old, positions_old)

        # get after-action SE3s
        Rs_new = quats_to_matrices(orientations_new)
        Ts_new = get_SE3s(Rs_new, positions_new)

        # get action position and vector
        action_position = action_coord[1:]
        action_angle = action_coord[0]
        action_vector = np.array([np.cos(action_angle), np.sin(action_angle), 0])

        x_scene = np.zeros((self.num_primitives, 13, self.num_primitives))
        a_scene = np.zeros((self.num_primitives, 5))
        if self.motion_dim == '2D':
            y_scene = np.zeros((self.num_primitives, 3))
        elif self.motion_dim == '3D':
            y_scene = np.zeros((self.num_primitives, 7))
        object_types = np.zeros(self.num_primitives)
        object_sizes = np.zeros((self.num_primitives, 3))
        
        # process object posses to revise x and z axis
        for object_idx in range(num_objects):
            # load object pose and parameters
            parameters_before = parameters[:, object_idx]
            T_old_before = Ts_old[object_idx]
            T_new_before = Ts_new[object_idx]
            size_before = object_info[object_idx]['size']

            # coordinate processing
            parameter_after, T_old_after, T_new_after, size_after = coordinate_processing(parameters_before, T_old_before, action_vector, T_new_before, size_before)

            # replace to new processed data
            parameters[:, object_idx] = parameter_after
            Ts_old[object_idx] = T_old_after
            Ts_new[object_idx] = T_new_after
            object_info[object_idx]['size'] = size_after

        for object_idx in range(num_objects):
            # compute inverse SE2 of the selected object
            T_inverse = np.linalg.inv(Ts_old[object_idx])
            C = SE2_from_SE3(Ts_old[object_idx])
            C_inverse = np.linalg.inv(C)

            # change the order
            object_idxs = list(range(num_objects))
            object_idxs.remove(object_idx)
            Ts_obj_centric = Ts_old[[object_idx] + object_idxs]
            parameters_obj_centric = parameters[:, [object_idx] + object_idxs]

            # inversely transform with the selected object's SE2
            Ts_wrt_object = C_inverse @ Ts_obj_centric

            # compute positions and orientations w.r.t. the selected object
            positions_wrt_object = Ts_wrt_object[:, :3, 3]
            orientations_wrt_object = matrices_to_quats(Ts_wrt_object[:, :3, :3])

            # x: array with shape (17, num_primitives) where 17 dims consist of 1 dim confidences, 3 dims positions, 4 dims orientations and 9 dims parameters
            x = np.zeros((13, self.num_primitives))
            x[:, :num_objects] = np.concatenate([np.ones((1, num_objects)), positions_wrt_object.T, orientations_wrt_object.T, parameters_obj_centric[:5]])

            # express action w.r.t. the selected object
            action_position_wrt_object = (C_inverse @ np.append(action_position, 1))[:-1]
            action_vector_wrt_object = C_inverse[:3, :3] @ action_vector
            action_angle_wrt_object = np.arctan2(action_vector_wrt_object[1], action_vector_wrt_object[0])
            action_angle_wrt_object = np.array([np.cos(action_angle_wrt_object), np.sin(action_angle_wrt_object)])

            # a: array with shape (5, ) where 5 dims consist of x, y, z, cos(\angle), sin(\angle) whose dimensions are 1
            a = np.concatenate([action_position_wrt_object, action_angle_wrt_object])

            # get SE3 of the selected object after action and compute position and orientation difference
            T_new = Ts_new[object_idx]
            T_diff = T_inverse @ T_new
            if self.motion_dim == '2D':
                position_diff = T_diff[:2, 3]
                orientation_diff = np.array([np.arctan2(T_diff[1, 0], T_diff[0, 0])])
            elif self.motion_dim == '3D':
                position_diff = T_diff[:3, 3]
                orientation_diff = matrices_to_quats(T_diff[:3, :3])

            # y:
            #   2D: array with shape (3, ) where 3 dims consist of 2 dims position and 1 dims orientation
            #   3D: array with shape (7, ) where 7 dims consist of 3 dims position and 4 dims orientation
            y = np.concatenate([position_diff, orientation_diff])

            x = self.noise_augmentation_motion(x, parameters_noise=self.parameters_noise_augment, motion_noise=self.motion_noise_augment, std=self.motion_noise_std)

            x_scene[object_idx] = x
            a_scene[object_idx] = a
            y_scene[object_idx] = y
            if object_info[object_idx]['type'] == 'box':
                object_types[object_idx] = 1
            elif object_info[object_idx]['type'] == 'cylinder':
                object_types[object_idx] = 2
            else:
                raise KeyError("Unknown object type. Known object types are 'box' and 'cylinder'")
            object_sizes[object_idx] = object_info[object_idx]['size']
            Ts_old_scene = np.append(Ts_old, np.zeros((self.num_primitives-len(Ts_old), 4, 4)), axis=0)
        
        # gt point cloud
        pc_gt_appended = np.zeros((6, 512, self.num_primitives))
        pc_gt_appended[:, :, :pc_gt.shape[2]] = pc_gt

        if self.seg_noise_augment:
            pc = noise_augmentation(pc, self.seg_noise_std)
        pc, pc_gt_appended, mean_xyz, diagonal_len = normalize_pointcloud(pc, pc_gt_appended)

        data = {
            'pc': pc.astype(np.float32), 'labels': labels, 'action': action_coord.astype(np.float32),
            'x_scene': x_scene.astype(np.float32), 'a_scene': a_scene.astype(np.float32), 'y_scene': y_scene.astype(np.float32),
            'object_types': object_types, 'object_sizes': object_sizes,
            'Ts': Ts_old_scene.astype(np.float32), 'pc_gt': pc_gt_appended,
            'mean_xyz': mean_xyz.astype(np.float32), 'diagonal_len': diagonal_len.astype(np.float32),
        }

        return data

    def get_dsr_net_data(self, file):
        action_pixel = file['action_pixel'][()]
        tsdf = file['tsdf'][()]
        mask_3d_old = file['mask_3d_old'][()].astype(np.int64)
        scene_flow_3d = file['scene_flow_3d'][()].astype(np.float32).transpose([3, 0, 1, 2])

        self.volume_size = mask_3d_old.shape

        action_map = self.get_action(action_pixel)
        
        data = {
            'action': action_map,
            'tsdf': tsdf,
            'mask_3d_old': mask_3d_old,
            'scene_flow_3d': scene_flow_3d,
        }

        return data

    def get_se3_nets_data(self, file):
        organized_pc = file['organized_pc'][()]
        organized_flow = file['scene_flow_2d'][()]
        action_coord = file['action_coord'][()]
        mask_2d_old = file['mask_2d_old'][()]

        action_position = action_coord[1:]
        action_angle = action_coord[0]
        action_vector = np.array([np.cos(action_angle), np.sin(action_angle), 0])
        action = np.concatenate([action_position, action_vector])
        
        opc = torch.tensor((organized_pc).transpose(2, 0, 1), dtype=torch.float)
        act = torch.tensor(action, dtype=torch.float)
        flow = torch.tensor((organized_flow).transpose(2, 0, 1), dtype=torch.float)

        data = {
            'organized_pc': opc,
            'action': act,
            'organized_flow': flow,
            'mask_2d_old': mask_2d_old,
        }

        return data

    def noise_augmentation_motion(self, x, parameters_noise, motion_noise, std=0.02):
        if motion_noise:
            x[1:8, 1:] *= np.random.normal(1, std, (7, self.num_primitives-1))
        
        if parameters_noise:
            x[8:] *= np.random.normal(1, std, (5, self.num_primitives))
            
            x[11:13] = np.clip(x[11:13], 0.2, 2.0)

        return x

    def get_action(self, action_pixel):
        ##### Need to be imported in data generation #####
        [direction, r, c, z] = action_pixel
        action_map = np.zeros([self.num_directions] + list(self.volume_size), np.float32)
        action_map[direction, r, c, z] = 1

        return action_map

def normalize_pointcloud(pc, gt):
    mean_xyz = np.expand_dims(np.mean(pc, axis=1), 1)
    max_xyz = np.max(pc, axis=1)
    min_xyz = np.min(pc, axis=1)
    diagonal_len = np.linalg.norm(max_xyz-min_xyz)

    pc -= mean_xyz
    pc /= diagonal_len

    gt[:, :3] -= np.expand_dims(mean_xyz, 0)
    gt[:, :3] /= diagonal_len

    return pc, gt, mean_xyz, diagonal_len

def get_DSQ_parameters(object_infos):
    parameters = np.zeros((9, len(object_infos)))

    for object_id, object_info in enumerate(object_infos):
        if object_info['type'] == 'box':
            parameters[0, object_id] = object_info['size'][0] / 2   # a1
            parameters[1, object_id] = object_info['size'][1] / 2   # a2
            parameters[2, object_id] = object_info['size'][2] / 2   # a3
            parameters[3, object_id] = 0.2                          # e1
            parameters[4, object_id] = 0.2                          # e2
            parameters[5, object_id] = 0                            # k
            parameters[6, object_id] = 1e-2                         # b
            parameters[7, object_id] = 1                            # cos(\alpha)
            parameters[8, object_id] = 0                            # sin(\alpha)
        elif object_info['type'] == 'cylinder':
            parameters[0, object_id] = object_info['size'][0]       # a1
            parameters[1, object_id] = object_info['size'][1]       # a2
            parameters[2, object_id] = object_info['size'][2] / 2   # a3
            parameters[3, object_id] = 0.2                          # e1
            parameters[4, object_id] = 1                            # e2
            parameters[5, object_id] = 0                            # k
            parameters[6, object_id] = 1e-2                         # b
            parameters[7, object_id] = 1                            # cos(\alpha)
            parameters[8, object_id] = 0                            # sin(\alpha)
        else:
            raise NotImplementedError

    return parameters

def coordinate_processing(parameter_before, T_old_before, action_vector, T_new_before=None, size_before=None):
    T_old_after = deepcopy(T_old_before)
    parameter_after = deepcopy(parameter_before)
    if T_new_before is not None:
        T_new_after = deepcopy(T_new_before)
    if size_before is not None:
        size_after = deepcopy(size_before)
    axis_indices = [0, 1, 2]

    # align z-axis to gravity axis 
    if abs(parameter_before[3] - parameter_before[4]) < 1e-2:
        inner_product_values_with_z = np.squeeze(np.array([[0, 0, 1]]).dot(T_old_before[:3, axis_indices]))
        new_z_axis_idx = np.argmax(abs(inner_product_values_with_z))
        z_axis_sign = np.sign(inner_product_values_with_z[new_z_axis_idx])
        T_old_after[:3, 2] = T_old_before[:3, new_z_axis_idx] * z_axis_sign
        parameter_after[2] = parameter_before[new_z_axis_idx]
        if T_new_before is not None:
            T_new_after[:3, 2] = T_new_before[:3, new_z_axis_idx] * z_axis_sign
        if size_before is not None:
            size_after[2] = size_before[new_z_axis_idx]
    else:
        z_axis_sign = np.sign(np.squeeze(np.array([[0, 0, 1]]).dot(T_old_before[:3, 2])))
        new_z_axis_idx = 2
        T_old_after[:3, 2] = T_old_before[:3, 2] * z_axis_sign
        if T_new_before is not None:
            T_new_after[:3, 2] = T_new_before[:3, 2] * z_axis_sign

    # remove candidate
    axis_indices.remove(new_z_axis_idx)

    # align x-axis to action axis
    inner_product_values_with_action = np.squeeze(action_vector.dot(T_old_before[:3, axis_indices]))
    new_x_axis_idx = np.argmax(abs(inner_product_values_with_action))
    x_axis_sign = np.sign(inner_product_values_with_action[new_x_axis_idx])
    new_x_axis_idx = axis_indices[new_x_axis_idx]
    T_old_after[:3, 0] = T_old_before[:3, new_x_axis_idx] * x_axis_sign
    parameter_after[0] = parameter_before[new_x_axis_idx]
    if T_new_before is not None:
        T_new_after[:3, 0] = T_new_before[:3, new_x_axis_idx] * x_axis_sign
    if size_before is not None:
        size_after[0] = size_before[new_x_axis_idx]

    # remove candidate
    axis_indices.remove(new_x_axis_idx)

    # calculate y axis
    T_old_after[:3, 1] = np.cross(T_old_after[:3, 2], T_old_after[:3, 0])
    parameter_after[1] = parameter_before[axis_indices[0]]
    if T_new_before is not None:
        T_new_after[:3, 1] = np.cross(T_new_after[:3, 2], T_new_after[:3, 0])
    if size_before is not None:
        size_after[1] = size_before[axis_indices[0]]

    if T_new_before is not None:
        if size_before is not None:
            return parameter_after, T_old_after, T_new_after, size_after
        else:
            return parameter_after, T_old_after, T_new_after
    else:
        if size_before is not None:
            return parameter_after, T_old_after, size_after
        else:
            return parameter_after, T_old_after

def SE2_from_SE3(T):
    C = np.eye(4, 4)

    # rotation matrix
    z_axis = T[:3, 2]
    cross_product = np.cross(z_axis, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(cross_product)
    cos_theta = np.squeeze(np.array([[0, 0, 1]]).dot(z_axis))
    if sin_theta < 1e-12:
        if cos_theta > 0:
            C[:3, :3] = T[:3, :3]
        else:
            C[:3, :3] = -T[:3, [1, 0, 2]]
    else:
        screw = cross_product / sin_theta
        theta = np.arctan2(sin_theta, cos_theta)
        C_temp = T[:3, :3].dot(np.linalg.inv(exp_so3(theta * screw)))
        C[:3, :3] = T[:3, :3].dot(exp_so3(theta * np.linalg.inv(T[:3, :3]).dot(screw)))

        # print(C_temp - C[:3, :3])



    # x,y positions
    C[:2, 3] = T[:2, 3]
    # print(C[2, 2])

    return C