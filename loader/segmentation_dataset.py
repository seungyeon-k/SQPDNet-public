import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
import h5py

from functions.point_clouds import noise_augmentation


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg_data):
        self.num_classes = cfg_data["num_classes"]
        self.num_points = cfg_data["num_points"]
        self.noise_augment = cfg_data.get("noise_augment", True)
        self.noise_std = cfg_data.get("noise_std", 0)

        self.pc_list, self.labels_list = self.load_data(split, cfg_data)

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx):
        pc = self.pc_list[idx]

        if self.noise_augment:
            pc = noise_augmentation(pc, self.noise_std)

        pc, _, _ = normalize_pointcloud(pc)
        
        pc = torch.Tensor(pc)
        labels = self.labels_list[idx]

        return pc, labels

    def load_data(self, split, cfg_data):
        data_path_list = cfg_data['paths']

        file_list_dict = {}
        for data_path in data_path_list:
            file_list_dict[data_path] = os.listdir(osp.join(data_path, split))

        dataset_size = sum(len(file_list) for file_list in file_list_dict.values())
        pc_list = [None] * dataset_size
        labels_list = [None] * dataset_size

        data_idx = 0
        for data_path, file_list in file_list_dict.items():
            for file_idx in tqdm(file_list, desc=f"Loading {str(data_path)}/{split} ... ", leave=False):
                file = h5py.File(osp.join(osp.join(data_path, split), file_idx), 'r')

                # check the number of point
                assert file['pc_down_wo_plane'].shape[0] == self.num_points, \
                    f"Number of points in point cloud data should be {self.num_points}, but is {file['pc_down_wo_plane'].shape[0]}"
                assert file['labels_down_wo_plane'].shape[0] == self.num_points, \
                    f"Number of points in label data should be {self.num_points}, but is {file['labels_down_wo_plane'].shape[0]}"

                # point cloud data
                pc = file['pc_down_wo_plane'][()].transpose()
                pc_list[data_idx] = pc

                # segmentation label
                labels = file['labels_down_wo_plane'][()]

                assert max(labels) <= self.num_classes, f"Number of classes must be {max(labels)}, but is {self.num_classes}"

                # one-hode encode segmentation label
                labels = np.eye(self.num_classes)[labels]

                labels_list[data_idx] = labels

                data_idx += 1

        # delete None values
        pc_list = pc_list[:data_idx]
        labels_list = labels_list[:data_idx]

        return pc_list, labels_list


def normalize_pointcloud(pc):
    mean_xyz = np.expand_dims(np.mean(pc, axis=1), 1)
    max_xyz = np.max(pc, axis=1)
    min_xyz = np.min(pc, axis=1)
    diagonal_len = np.linalg.norm(max_xyz-min_xyz)

    pc -= mean_xyz
    pc /= diagonal_len

    return pc, mean_xyz, diagonal_len
