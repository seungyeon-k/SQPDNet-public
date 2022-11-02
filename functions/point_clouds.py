import numpy as np
from sklearn.preprocessing import normalize
import open3d as o3d
import random


def noise_augmentation(pc, noise_std=0.001):
    """
    input: point cloud (3, n)
    output: noisy point cloud (3, n)
    """
    pc_wo_label = pc[:3, :]
    noise = np.random.uniform(-1, 1, size=pc_wo_label.shape)
    noise = normalize(noise, axis=0, norm="l2")

    scale = np.random.normal(loc=0, scale=noise_std, size=(1, pc_wo_label.shape[1]))
    scale = scale.repeat(pc_wo_label.shape[0], axis=0)

    pc[:3, :] = pc[:3, :] + noise * scale

    return pc


def upsample_pointcloud(num_upsample_pc, pc, rgb_pc=None, labels=None):
    while pc.shape[0] < num_upsample_pc:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        partial_pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        idx = random.choice(range(pc.shape[0]))
        idx_nearest_points = list(partial_pcd_tree.search_knn_vector_3d(pcd.points[idx], 2)[1])
        pc_append = np.mean([pc[idx_nearest_points]], axis=1)
        pc = np.append(pc, pc_append, axis=0)
        
        if rgb_pc is not None:
            rgb_pc_append = np.mean([rgb_pc[idx_nearest_points]], axis=1)
            rgb_pc = np.append(rgb_pc, rgb_pc_append, axis=0)
        
        if labels is not None:
            idx_nearest_points.remove(idx)

            label_append = labels[idx_nearest_points]
            labels = np.append(labels, label_append)

    if rgb_pc is not None:
        if labels is not None:
            return pc, rgb_pc, labels
        else:
            return pc, rgb_pc
    else:
        if labels is not None:
            return pc, labels
        else:
            return pc
