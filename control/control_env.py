import pybullet as p
import numpy as np
import time
import random

from data_generation.sim import PybulletSim
from data_generation.utils import project_pts_to_3d, euler2rotm


class ControlSimulationEnv:
    def __init__(self, enable_gui=True):
        self.enable_gui = enable_gui
        self.sim = PybulletSim(enable_gui=enable_gui)
        # self.workspace_bounds = self.sim.workspace_bounds
        self.workspace_bounds = self.sim.workspace_bounds
        self.num_pts_down_wo_plane = 2048
        self.object_ids = []
        self.voxel_size = 0.004

    def reset(self, cfg_objects):
        while True:
            # remove objects
            for obj_id in self.object_ids:
                p.removeBody(obj_id)
            self.object_ids = []

            # load objects
            self._drop_objects(cfg_objects)
            time.sleep(1)
            
            # wait until objects stop moving
            flag = False
            old_po = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
            for _ in range(10):
                time.sleep(1)
                new_ps = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
                if np.sum((new_ps - old_po) ** 2) < 1e-6:
                    flag = True
                    break
                old_po = new_ps
            if not flag:
                continue

            return

    def _drop_objects(self,  cfg_objects):
        for obj_dict in cfg_objects.values():
            position_xy = obj_dict.position_xy
            orientation = obj_dict.orientation
            
            if obj_dict.type == 'box':
                dim = np.array(obj_dict.XYZ)

                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim/2)
                position = position_xy + [float(self.workspace_bounds[2, 0] + obj_dict.XYZ[2] / 2 + 0.01)]
                body_id = p.createMultiBody(0.05, collision_id, -1, position, orientation)

                p.changeDynamics(body_id, -1, spinningFriction=0.002, lateralFriction=0.4, mass=0.737*dim[0]*dim[1]*dim[2]/250)
                p.changeVisualShape(body_id, -1, rgbaColor=np.concatenate([1*np.random.rand(3), [1]]))

                self.object_ids.append(body_id)

                time.sleep(0.2)
            
            elif obj_dict.type == 'cylinder':
                collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=obj_dict.RH[0], height=obj_dict.RH[1])
                position = position_xy + [float(self.workspace_bounds[2, 0] + obj_dict.RH[1] / 2 + 0.01)]
                body_id = p.createMultiBody(0.05, collision_id, -1, position, orientation)
    
                p.changeDynamics(body_id, -1, spinningFriction=0.002, lateralFriction=0.4, mass=0.585*np.pi*obj_dict.RH[0]**2*obj_dict.RH[1]/250)
                p.changeVisualShape(body_id, -1, rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))

                self.object_ids.append(body_id)

                time.sleep(0.2)

    def _get_pc(self):
        color_image, depth_image, _ = self.sim.get_camera_data(self.sim.camera_params[0])

        camera_pose = self.sim.camera_params[0]['camera_pose']
        camera_intr = self.sim.camera_params[0]['camera_intr']

        organized_pc, _ = project_pts_to_3d(color_image, depth_image, camera_intr, camera_pose)

        # pc = self._get_workspace_pc(organized_pc)
        pc = organized_pc.reshape(-1, organized_pc.shape[2])

        wo_plane_idxs = (pc[:, 2] >= self.workspace_bounds[2, 0] + 0.002) * \
                        (pc[:, 2] <= self.workspace_bounds[2, 1]) * \
                        (pc[:, 0] >= self.workspace_bounds[0, 0])
        wo_plane_idxs = np.where(wo_plane_idxs==1)[0].tolist()

        down_wo_plane_idxs = random.sample(wo_plane_idxs, self.num_pts_down_wo_plane) if len(wo_plane_idxs) > self.num_pts_down_wo_plane else wo_plane_idxs
        pc = pc[down_wo_plane_idxs]

        return pc

    def _get_workspace_pc(self, organized_pc):
        pc = organized_pc.reshape(-1, organized_pc.shape[2])
                
        valid_idxs = np.logical_and(
            np.logical_and(
                np.logical_and(pc[:, 0]>=self.workspace_bounds[0, 0], pc[:, 0]<=self.workspace_bounds[0, 1]),
                np.logical_and(pc[:, 1]>=self.workspace_bounds[1, 0], pc[:, 1]<=self.workspace_bounds[1, 1])
            ),
            np.logical_and(pc[:, 2]>=self.workspace_bounds[2, 0]-0.001, pc[:, 2]<=self.workspace_bounds[2, 1])
        )

        pc = pc[valid_idxs]

        return pc

    def _get_image(self):
        self.color_image_large, self.depth_image_large, mask_image_large = self.sim.get_camera_data(self.sim.camera_params[0])
        self.color_image_small, self.depth_image_small, mask_image_small = self.sim.get_camera_data(self.sim.camera_params[1])
        
        self.mask_image_large = np.zeros_like(mask_image_large)
        for i, object_id in enumerate(self.object_ids):
            self.mask_image_large += (mask_image_large == object_id) * (i + 1)

        self.mask_image_small = np.zeros_like(mask_image_small)
        for i, object_id in enumerate(self.object_ids):
            self.mask_image_small += (mask_image_small == object_id) * (i + 1)

    def _get_mask_3d(self, old_po_ors):
        vol_bnds = self.workspace_bounds
        mask = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds], dtype=np.int)

        cur_cnt = 0
        for obj_id, old_po_or in zip(self.object_ids, old_po_ors):
            cur_cnt += 1

            position, orientation = old_po_or
            old_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

            valid_idx = np.logical_and(
                np.logical_and(old_coord[:, 1] >= 0, old_coord[:, 1] < mask.shape[0]),
                np.logical_and(
                    np.logical_and(old_coord[:, 0] >= 0, old_coord[:, 0] < mask.shape[1]),
                    np.logical_and(old_coord[:, 2] >= 0, old_coord[:, 2] < mask.shape[2])
                )
            )
            x = old_coord[valid_idx, 1]
            y = old_coord[valid_idx, 0]
            z = old_coord[valid_idx, 2]

            mask[x, y, z] = cur_cnt

        return mask

    def _get_coord(self, obj_id, position, orientation, vol_bnds=None, voxel_size=None):
        # if vol_bnds is not None, return coord in voxel, else, return world coord
        coord = self.voxel_coord[obj_id]
        mat = euler2rotm(p.getEulerFromQuaternion(orientation))
        coord = (mat @ (coord.T)).T + np.asarray(position)
        if vol_bnds is not None:
            coord = np.round((coord - vol_bnds[:, 0]) / voxel_size).astype(np.int)
        return coord
        
    def _reset_objects(self, old_pos, old_ors):
        for idx, object_id in enumerate(self.object_ids):
            p.resetBasePositionAndOrientation(object_id, old_pos[idx], old_ors[idx])