import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
import h5py

from functions.point_clouds import noise_augmentation, upsample_pointcloud


class SurroundRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg_data):
        self.noise_augment = cfg_data.get("noise_augment", True)
        self.noise_std = cfg_data.get("noise_std", 0)

        self.pc_list, self.pc_object_list, self.gt_list, self.object_info_list, self.positions_list, self.orientations_list = self.load_data(split, cfg_data)

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx):
        pc = self.pc_list[idx]
        pc_object = self.pc_object_list[idx]
        gt = self.gt_list[idx]

        if self.noise_augment:
            pc = noise_augmentation(pc, self.noise_std)

        pc, gt, mean_xyz, diagonal_len = normalize_pointcloud(pc, pc_object, gt)

        pc = torch.Tensor(pc)
        gt = torch.Tensor(gt)

        object_info = self.object_info_list[idx]
        position = self.positions_list[idx]
        orientation = self.orientations_list[idx]

        return pc, gt, object_info, position, orientation, mean_xyz, diagonal_len

    def load_data(self, split, cfg_data):
        data_path_list = cfg_data['paths']

        file_list_dict = {}
        for data_path in data_path_list:
            file_list_dict[data_path] = os.listdir(osp.join(data_path, split))

        dataset_size = sum(len(file_list)*int(data_path[-1]) for data_path, file_list in file_list_dict.items())
        pc_list = [None] * dataset_size
        gt_list = [None] * dataset_size
        object_info_list = [None] * dataset_size
        positions_list = [None] * dataset_size
        orientations_list = [None] * dataset_size
        pc_object_list = [None] * dataset_size

        data_idx = 0
        for data_path, file_list in file_list_dict.items():
            for file_idx in tqdm(file_list, desc=f"Loading {str(data_path)}/{split} ... ", leave=False):
                file = h5py.File(osp.join(osp.join(data_path, split), file_idx), 'r')

                # segmentation label
                label = file['labels_down_wo_plane'][()]

                for object_id in np.unique(label):
                    # point cloud data
                    pc_object = file['pc_down_wo_plane'][()][label==object_id].transpose()
                    if pc_object.shape[1] < 100:
                        continue
                    pc_object = np.concatenate((pc_object, np.ones((1, pc_object.shape[1]))), axis=0)
                    pc_surround = file['pc_down_wo_plane'][()][label!=object_id].transpose()
                    pc_surround = np.concatenate((pc_surround, np.zeros((1, pc_surround.shape[1]))), axis=0)

                    pc_object_list[data_idx] = pc_object
                    pc_list[data_idx] = np.concatenate((pc_object, pc_surround), axis=1)

                    # gt point cloud
                    gt = file['pc_gt'][()][object_id-1].transpose()
                    gt_list[data_idx] = gt

                    # gt object info
                    object_info = eval(file['object_info'][()])[object_id-1]
                    object_info_list[data_idx] = str(object_info)

                    # gt postion and orienation
                    position = file['positions_old'][()][object_id-1]
                    orientation = file['orientations_old'][()][object_id-1]

                    positions_list[data_idx] = position
                    orientations_list[data_idx] = orientation

                    data_idx += 1

        # delete None values
        pc_list = pc_list[:data_idx]
        pc_object_list = pc_object_list[:data_idx]
        gt_list = gt_list[:data_idx]
        object_info_list = object_info_list[:data_idx]
        positions_list = positions_list[:data_idx]
        orientations_list = orientations_list[:data_idx]

        return pc_list, pc_object_list, gt_list, object_info_list, positions_list, orientations_list


def normalize_pointcloud(pc, pc_object, gt=None):
    mean_xyz = np.expand_dims(np.mean(pc_object[:3, :], axis=1), 1)
    max_xyz = np.max(pc_object[:3, :], axis=1)
    min_xyz = np.min(pc_object[:3, :], axis=1)
    diagonal_len = np.linalg.norm(max_xyz-min_xyz)

    pc[:3, :] -= mean_xyz
    if diagonal_len != 0:
        pc[:3, :] /= diagonal_len
    else:
        diagonal_len = np.array(1)

    if gt is not None:
        gt[:3] -= mean_xyz
        gt[:3] /= diagonal_len

        return pc, gt, mean_xyz, diagonal_len
    else:
        return pc, mean_xyz, diagonal_len
