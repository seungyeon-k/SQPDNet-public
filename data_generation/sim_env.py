import time
import numpy as np
import pybullet as p
import random
import open3d as o3d
import math
from copy import deepcopy

from .sim import PybulletSim
from functions.point_clouds import upsample_pointcloud
from .utils import euler2rotm, project_pts_to_2d, project_pts_to_3d, get_volume
from functions.utils import get_SE3s


class SimulationEnv():
    def __init__(self, enable_gui):
        self.enable_gui = enable_gui
        self.sim = PybulletSim(enable_gui=enable_gui)
        self.workspace_bounds = self.sim.workspace_bounds
        self.num_pts_down = 4096
        self.num_pts_down_wo_plane = 2048
        self.num_pts_gt = 512
        self.num_directions = 8
        self.num_z = 5
        self.voxel_size = 0.004
        self.object_ids = []

        # box size list (cm)
        self.box_size_list_dict = {}
        self.box_size_list_dict['known'] = np.array([[3, 7.5, 8.5],
                                                     [5.5, 5.5, 5.5],
                                                     [10.5, 15, 5.5],
                                                     [9.5, 5.5, 16],
                                                     [2.5, 15, 19],
                                                     [12, 2.5, 6],
                                                     [5.8, 6.3, 6],
                                                     [6.4, 6, 2.8],
                                                     [6, 6.2, 5.8]]) * 5
        self.box_size_list_dict['unknown'] = np.array([[4, 10, 11.2],
                                                       [7, 7, 7],
                                                       [7.5, 11, 4],
                                                       [7, 4, 11.5],
                                                       [2, 10, 13],
                                                       [9.5, 2, 5],
                                                       [7.7, 8.4, 8],
                                                       [7.5, 7, 3.3],
                                                       [4, 4.1, 3.9]]) * 5

        # cylinder size list (cm)
        self.cylinder_size_list_dict = {}
        self.cylinder_size_list_dict['known'] = np.array([[3, 6],
                                                          [3, 10],
                                                          [2.5, 13],
                                                          [3.5, 20],
                                                          [4, 8],
                                                          [4, 4],
                                                          [4, 12],
                                                          [2, 3],
                                                          [4, 15]]) * 5
        self.cylinder_size_list_dict['unknown'] = np.array([[4.5, 9],
                                                            [4, 13],
                                                            [2, 10],
                                                            [2.5, 14],
                                                            [2, 4],
                                                            [3, 3],
                                                            [4.5, 13.5],
                                                            [3, 4.5],
                                                            [3, 12],]) * 5

    def reset(self, object_types, knowledge, num_objects, enable_stacking):
        # old position orientation for transform
        self.old_po_ors_for_transform = None

        while True:
            # remove objects
            for obj_id in self.object_ids:
                p.removeBody(obj_id)
            self.object_ids = []
            self.voxel_coord = {}
            self.meshes = {}
            self.object_info = []

            # load objects
            self._random_drop(object_types, knowledge, num_objects, enable_stacking)
            time.sleep(1)

            # wait until objets stop moving
            flag = False
            old_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
            for _ in range(10):
                time.sleep(1)
                new_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
                if np.sum((new_pos - old_pos) ** 2) < 1e-6:
                    flag = True
                    break
                old_pos = new_pos
            if not flag:
                continue

            # check if all objects are in workspace
            if not self._check_workspace():
                continue

            # check stacked if enable_stacking is False
            if not enable_stacking and self._check_stacked():
                continue

            # check occlusion
            if self._check_occlusion():
                continue

            return

    def poke(self):
        output = {}

        # object information
        output['object_info'] = str(self.object_info)

        # log before-action scene information
        old_scene_info, old_po_ors = self._get_scene_info_before()
        output.update(old_scene_info)

        # sample action
        policy = self._action_sampler()
        if policy is not None:
            x_pixel, y_pixel, z_pixel, x_coord, y_coord, z_coord, direction_idx, direction_angle = policy
        else:
            return None

        # log action by pixel and coordinates
        output['action_pixel'] = np.array([direction_idx, y_pixel, x_pixel, z_pixel])
        output['action_coord'] = np.array([direction_angle, x_coord, y_coord, z_coord])

        # log after-action scene information
        new_scene_info = self._get_scene_info_after(old_po_ors)
        output.update(new_scene_info)

        return output

    def _random_drop(self, object_types, knowledge, num_objects, enable_stacking):
        distance_threshold = 0.07 if not enable_stacking else 0.05

        while True:
            xy_pos = np.random.rand(num_objects, 2)
            xy_pos *= 0.2 if not enable_stacking else 0.1

            if num_objects == 1:
                break

            distance_list = []
            for i in range(num_objects - 1):
                for j in range(i + 1, num_objects):
                    distance = np.sqrt(np.sum((xy_pos[i] - xy_pos[j])**2))
                    distance_list += [distance]

            # if not enable stacking, make objects far away
            if not enable_stacking and min(distance_list) > distance_threshold:
                break

            # if enable stacking, make objects close each others
            if enable_stacking and max(distance_list) < distance_threshold:
                break

        # make all objects locate around the center of workspce
        xy_pos -= np.mean(xy_pos, axis=0)
        xy_pos += np.mean(self.workspace_bounds, axis=1)[:2]

        for i in range(num_objects):
            object_type = random.choice(object_types)

            if object_type == 'box':
                box_index = np.random.randint(len(self.box_size_list_dict[knowledge]))

                size_box = self.box_size_list_dict[knowledge][box_index]
                exchange_dims = random.choice([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
                size_box = size_box[exchange_dims]
                size_x = np.int(np.round(size_box[0]))
                size_y = np.int(np.round(size_box[1]))
                size_z = np.int(np.round(size_box[2]))

                md = np.ones([size_x, size_y, size_z])
                coord = (np.asarray(np.nonzero(md)).T + 0.5 - np.array([size_x/2, size_y/2, size_z/2]))

                size_box = 500

                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([size_x/2, size_y/2, size_z/2])/size_box)
                if not enable_stacking:
                    orientation = [0, 0, np.random.rand()*2*np.pi]
                else:
                    orientation = [np.random.rand()*2*np.pi, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                position = np.append(xy_pos[i], size_z / size_box / 2 + self.workspace_bounds[2, 0] + 0.01)
                if enable_stacking:
                    position[2] += 0.1
                body_id = p.createMultiBody(0.05, collision_id, -1, position, p.getQuaternionFromEuler(orientation))

                p.changeDynamics(body_id, -1, spinningFriction=0.002, lateralFriction=0.4, mass=0.737*size_x*size_y*size_z/(125*1000))
                p.changeVisualShape(body_id, -1, rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))

                self.object_ids.append(body_id)

                self.voxel_coord[body_id] = coord / size_box

                mesh_box = o3d.geometry.TriangleMesh.create_box(width = size_x/size_box, height = size_y/size_box, depth = size_z/size_box)
                mesh_box.translate([-size_x/(2*size_box), -size_y/(2*size_box), -size_z/(2*size_box)]) # match center to the origin
                self.meshes[body_id] = mesh_box

                object_info = {'type': 'box', 'size': [size_x/size_box, size_y/size_box, size_z/size_box]}
                self.object_info.append(object_info)

                time.sleep(0.2 if not enable_stacking else 0.5)

            elif object_type == 'cylinder':
                cylinder_index = np.random.randint(len(self.cylinder_size_list_dict[knowledge]))

                size_cylinder = self.cylinder_size_list_dict[knowledge][cylinder_index]
                size_r = np.int(np.round(size_cylinder[0]))
                size_h = np.int(np.round(size_cylinder[1]))

                X = [r * np.cos(np.linspace(0, 2*np.pi, num=8*r, endpoint=False)) if r != 0 else [0] for r in range(size_r+1)]
                Y = [r * np.sin(np.linspace(0, 2*np.pi, num=8*r, endpoint=False)) if r != 0 else [0] for r in range(size_r+1)]
                X = np.expand_dims(np.concatenate(X), axis=1)
                Y = np.expand_dims(np.concatenate(Y), axis=1)
                Z = np.expand_dims(np.repeat(np.arange(size_h)+0.5-size_h/2, len(X)), 1)
                X = np.tile(X, (size_h, 1))
                Y = np.tile(Y, (size_h, 1))
                coord = np.concatenate((X, Y, Z), axis=1)

                size_cylinder = 500

                collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=size_r/size_cylinder, height=size_h/size_cylinder)
                position = np.append(xy_pos[i], self.workspace_bounds[2, 0] + size_h / size_cylinder / 2 + 0.01)
                orientation = [0, 0, np.random.rand()*2*np.pi]
                body_id = p.createMultiBody(0.05, collision_id, -1, position, p.getQuaternionFromEuler(orientation))

                p.changeDynamics(body_id, -1, spinningFriction=0.002, lateralFriction=0.4, mass=0.585*np.pi*size_r*size_r*size_h/(125*1000))
                p.changeVisualShape(body_id, -1, rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))

                self.object_ids.append(body_id)

                self.voxel_coord[body_id] = coord / size_cylinder

                mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=size_r/size_cylinder, height=size_h/size_cylinder, resolution=100, split=10)
                self.meshes[body_id] = mesh_cylinder

                object_info = {'type': 'cylinder', 'size': [size_r/size_cylinder, size_r/size_cylinder, size_h/size_cylinder]}
                self.object_info.append(object_info)

                time.sleep(0.2)

    def _check_workspace(self):
        for obj_id in self.object_ids:
            position, orientation = p.getBasePositionAndOrientation(obj_id)
            coord = self._get_coord(obj_id, position, orientation)

            valid_idxs = np.logical_and(
                np.logical_and(
                    np.logical_and(coord[:, 0]>=self.workspace_bounds[0, 0], coord[:, 0]<=self.workspace_bounds[0, 1]),
                    np.logical_and(coord[:, 1]>=self.workspace_bounds[1, 0], coord[:, 1]<=self.workspace_bounds[1, 1])
                ),
                np.logical_and(coord[:, 2]>=self.workspace_bounds[2, 0], coord[:, 2]<=self.workspace_bounds[2, 1])
            ).all()

            if not valid_idxs:
                return False

        return True

    def _check_stacked(self):
        po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]
        mask_3d = self._get_mask_scene_flow_3d(po_ors, get_scene_flow=False)

        # bottom and top 2d mask
        mask_2d_bot = mask_3d[:, :, 0]
        mask_2d_top = np.max(mask_3d, axis=2)

        # count pixels
        pixel_cnt_bot = np.zeros(len(self.object_ids) + 1)
        uniques, counts = np.unique(mask_2d_bot, return_counts=True)
        pixel_cnt_bot[uniques] = counts

        pixel_cnt_top = np.zeros(len(self.object_ids) + 1)
        uniques, counts = np.unique(mask_2d_top, return_counts=True)
        pixel_cnt_top[uniques] = counts

        # if the bottom 2d mask and the top 2d mask have different pixel counts, it means stacked
        if (np.abs(pixel_cnt_bot - pixel_cnt_top) < 10).all():
            return False
        else:
            return True

    def _check_occlusion(self):
        po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]
        mask, image_size, point_id, x, y = self._get_mask_scene_flow_2d(po_ors, get_scene_flow=False, occlusion=True)

        obj_num = len(self.object_ids)
        mask_sep = np.zeros([obj_num + 1, image_size[0], image_size[1]])
        mask_sep[point_id, x, y] = 1

        for i in range(obj_num):
            tot_pixel_num = np.sum(mask_sep[i + 1])
            vis_pixel_num = np.sum((mask == (i+1)).astype(np.float))

            if vis_pixel_num < 0.4 * tot_pixel_num:
                return True

        return False

    def _get_scene_info_before(self):

        # before-action positions and orientations
        old_po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]

        positions_old = np.array([old_po_or[0] for old_po_or in old_po_ors])
        orientations_old = np.array([old_po_or[1] for old_po_or in old_po_ors])

        self._get_image()

        self._move_meshes(old_po_ors)   # meshes are used to augment mask and generate gt point cloud

        # point cloud
        pc_data = self._get_pc(self.color_image_large, self.depth_image_large, self.mask_image_large)

        # ground truth point cloud
        pc_gt = self._get_gt_pc()

        # tsdf
        tsdf = get_volume(self.color_image_large, self.depth_image_large, \
            self.sim.camera_params[0]['camera_intr'], self.sim.camera_params[0]['camera_pose'], deepcopy(self.workspace_bounds), self.voxel_size)

        # 3d masks
        mask_3d = self._get_mask_scene_flow_3d(old_po_ors, get_scene_flow=False)

        # organized point cloud
        organized_pc, _ = project_pts_to_3d(self.color_image_small, self.depth_image_small, \
            self.sim.camera_params[1]['camera_intr'], self.sim.camera_params[1]['camera_pose'])

        scene_info = {
            'positions_old': positions_old,
            'orientations_old': orientations_old,
            'pc_gt': pc_gt,
            'tsdf': tsdf,
            'mask_3d_old': mask_3d,
            'mask_2d_old': self.mask_image_small,
            'organized_pc': organized_pc,
            'depth_image_old': self.depth_image_small,
            'color_image_old': self.color_image_small,
        }
        scene_info.update(pc_data)

        return scene_info, old_po_ors

    def _get_scene_info_after(self, old_po_ors):
        
        # after-action positions and orientations
        new_po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]

        positions_new = np.array([new_po_or[0] for new_po_or in new_po_ors])
        orientations_new = np.array([new_po_or[1] for new_po_or in new_po_ors])

        # 3d, 2d scene_flows
        _, scene_flow_3d = self._get_mask_scene_flow_3d(old_po_ors)
        _, scene_flow_2d = self._get_mask_scene_flow_2d(old_po_ors)

        # 3d mask
        mask_3d = self._get_mask_scene_flow_3d(new_po_ors, get_scene_flow=False)

        self._get_image()

        scene_info = {
            'positions_new': positions_new,
            'orientations_new': orientations_new,
            'scene_flow_3d': scene_flow_3d,
            'scene_flow_2d': scene_flow_2d,
            'mask_3d_new': mask_3d,
            'mask_2d_new': self.mask_image_small,
            'depth_image_new': self.depth_image_small,
            'color_image_new': self.color_image_small,
        }

        return scene_info

    def _get_image(self):
        self.color_image_large, self.depth_image_large, mask_image_large = self.sim.get_camera_data(self.sim.camera_params[0])
        self.color_image_small, self.depth_image_small, mask_image_small = self.sim.get_camera_data(self.sim.camera_params[1])
        
        self.mask_image_large = np.zeros_like(mask_image_large)
        for i, object_id in enumerate(self.object_ids):
            self.mask_image_large += (mask_image_large == object_id) * (i + 1)

        self.mask_image_small = np.zeros_like(mask_image_small)
        for i, object_id in enumerate(self.object_ids):
            self.mask_image_small += (mask_image_small == object_id) * (i + 1)

    def _move_meshes(self, new_po_ors):
        if self.old_po_ors_for_transform is None:
            for obj_id, po_or in zip(self.object_ids, new_po_ors):
                position, orientation = po_or

                # get mesh
                mesh = self.meshes[obj_id]

                # get T matrix
                R = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3,3)
                T = get_SE3s(R, np.array(position))

                # tranform mesh
                mesh.transform(T)

        else:
            for obj_id, new_po_or, old_po_or in zip(self.object_ids, new_po_ors, self.old_po_ors_for_transform):
                position_new, orientation_new = new_po_or
                position_old, orientation_old = old_po_or

                # get mesh
                mesh = self.meshes[obj_id]

                # get T_new matrix
                R_new = np.asarray(p.getMatrixFromQuaternion(orientation_new)).reshape(3, 3)
                T_new = get_SE3s(R_new, np.array(position_new))

                # get T_old matrix
                R_old = np.asarray(p.getMatrixFromQuaternion(orientation_old)).reshape(3, 3)
                T_old = get_SE3s(R_old, np.asarray(position_old))

                # transform mesh
                mesh.transform(np.linalg.inv(T_old))
                mesh.transform(T_new)

        self.old_po_ors_for_transform = new_po_ors

    def _get_coord(self, obj_id, position, orientation, vol_bnds=None, voxel_size=None):
        # if vol_bnds is not None, return coord in voxel, else, return world coord
        coord = self.voxel_coord[obj_id]
        mat = euler2rotm(p.getEulerFromQuaternion(orientation))
        coord = (mat @ (coord.T)).T + np.asarray(position)
        if vol_bnds is not None:
            coord = np.round((coord - vol_bnds[:, 0]) / voxel_size).astype(np.int)
        return coord

    def _get_mask_scene_flow_3d(self, old_po_ors, get_scene_flow=True):
        vol_bnds = self.workspace_bounds
        mask = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds], dtype=np.int)

        if get_scene_flow:
            scene_flow = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds] + [3])

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

            if get_scene_flow:
                position, orientation = p.getBasePositionAndOrientation(obj_id)
                new_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

                motion = new_coord - old_coord

                motion = motion[valid_idx]
                motion = np.stack([motion[:, 1], motion[:, 0], motion[:, 2]], axis=1)

                scene_flow[x, y, z] = motion

        if get_scene_flow:
            return mask, scene_flow
        else:
            return mask

    def _get_mask_scene_flow_2d(self, old_po_ors, get_scene_flow=True, occlusion=False):
        old_coords = []
        point_id = []
        cur_cnt = 0

        camera_view_matrix = np.array(self.sim.camera_params[1]['camera_view_matrix']).reshape(4, 4).T
        camera_intr = self.sim.camera_params[1]['camera_intr']
        image_size = self.sim.camera_params[1]['camera_image_size']

        mask = np.zeros([image_size[0], image_size[1]])
        
        for obj_id, po_or in zip(self.object_ids, old_po_ors):
            cur_cnt += 1

            position, orientation = po_or
            old_coord = self._get_coord(obj_id, position, orientation)

            old_coords.append(old_coord)            
            point_id.append([cur_cnt for _ in range(old_coord.shape[0])])

        point_id = np.concatenate(point_id)
        old_coords_world = np.concatenate(old_coords)

        old_coords_2d = project_pts_to_2d(old_coords_world.T, camera_view_matrix, camera_intr)

        y = np.round(old_coords_2d[0]).astype(np.int)
        x = np.round(old_coords_2d[1]).astype(np.int)
        depth = old_coords_2d[2]

        valid_idx = np.logical_and(
            np.logical_and(x >= 0, x < image_size[0]),
            np.logical_and(y >= 0, y < image_size[1])
        )
        x = x[valid_idx]
        y = y[valid_idx]
        depth = depth[valid_idx]
        point_id = point_id[valid_idx]

        sort_id = np.argsort(-depth)
        x = x[sort_id]
        y = y[sort_id]
        point_id = point_id[sort_id]

        mask[x, y] = point_id

        if get_scene_flow:
            new_coords = []

            scene_flow = np.zeros([image_size[0], image_size[1], 3])

            for obj_id in self.object_ids:
                position, orientation = p.getBasePositionAndOrientation(obj_id)
                new_coord = self._get_coord(obj_id, position, orientation)
                new_coords.append(new_coord)

            new_coords_world = np.concatenate(new_coords)

            motion = (new_coords_world - old_coords_world)[valid_idx]

            motion = motion[sort_id]
            motion = np.stack([motion[:, 0], motion[:, 1], motion[:, 2]], axis=1)

            scene_flow[x, y] = motion

        if occlusion:
            return mask, image_size, point_id, x, y
        else:
            if get_scene_flow:
                return mask, scene_flow
            else:
                return mask

    def _get_pc(self, color_image, depth_image, mask_image):
        camera_pose = self.sim.camera_params[0]['camera_pose']
        camera_intr = self.sim.camera_params[0]['camera_intr']

        organized_pc, organized_rgb_pc = project_pts_to_3d(color_image, depth_image, camera_intr, camera_pose)

        pc, rgb_pc, labels = self._get_workspace_pc(organized_pc, organized_rgb_pc, mask_image)

        wo_plane_idxs = pc[:, 2] >= self.workspace_bounds[2, 0] + 0.002
        wo_plane_idxs = np.where(wo_plane_idxs==1)[0].tolist()

        if pc.shape[0] > self.num_pts_down:
            down_idxs = random.sample(range(pc.shape[0]), self.num_pts_down)
        else:
            down_idxs = list(range(pc.shape[0]))
        pc_down = pc[down_idxs]
        rgb_pc_down = rgb_pc[down_idxs]
        labels_down = labels[down_idxs]

        if len(wo_plane_idxs) > self.num_pts_down_wo_plane:
            down_wo_plane_idxs = random.sample(wo_plane_idxs, self.num_pts_down_wo_plane)
            pc_down_wo_plane = pc[down_wo_plane_idxs]
            rgb_pc_down_wo_plane = rgb_pc[down_wo_plane_idxs]
            labels_down_wo_plane = labels[down_wo_plane_idxs]
        else:
            pc_wo_plane = pc[wo_plane_idxs]
            rgb_pc_wo_plane = rgb_pc[wo_plane_idxs]
            labels_wo_plane = labels[wo_plane_idxs]
            pc_down_wo_plane, rgb_pc_down_wo_plane, labels_down_wo_plane = \
                upsample_pointcloud(self.num_pts_down_wo_plane, pc_wo_plane, rgb_pc_wo_plane, labels_wo_plane)

        pc_data = {
            'pc': pc,
            'rgb_pc': rgb_pc,
            'labels': labels,
            'pc_down': pc_down,
            'rgb_pc_down': rgb_pc_down,
            'labels_down': labels_down,
            'pc_down_wo_plane': pc_down_wo_plane,
            'rgb_pc_down_wo_plane': rgb_pc_down_wo_plane,
            'labels_down_wo_plane': labels_down_wo_plane,
        }

        return pc_data

    def _get_workspace_pc(self, organized_pc, rgb_pc, labels):
        pc = organized_pc.reshape(-1, organized_pc.shape[2])
        rgb_pc = rgb_pc.reshape(-1, rgb_pc.shape[2])
        labels = labels.reshape(-1)

        valid_idxs = np.logical_and(
            np.logical_and(
                np.logical_and(pc[:, 0]>=self.workspace_bounds[0, 0], pc[:, 0]<=self.workspace_bounds[0, 1]),
                np.logical_and(pc[:, 1]>=self.workspace_bounds[1, 0], pc[:, 1]<=self.workspace_bounds[1, 1])
            ),
            np.logical_and(pc[:, 2]>=self.workspace_bounds[2, 0]-0.001, pc[:, 2]<=self.workspace_bounds[2, 1])
        )

        pc = pc[valid_idxs]
        rgb_pc = rgb_pc[valid_idxs]
        labels = labels[valid_idxs]

        return pc, rgb_pc, labels

    def _get_gt_pc(self):
        gt_pc_list = []
        for obj_id in self.object_ids:
            # get mesh
            mesh = self.meshes[obj_id]
            mesh.compute_vertex_normals()

            # uniform sampling
            pcd = mesh.sample_points_uniformly(number_of_points=self.num_pts_gt)
            gt_pc = np.asarray(pcd.points)
            gt_normals = np.asarray(pcd.normals)
            gt_pc = np.concatenate((gt_pc, gt_normals), axis=1)

            gt_pc_list.append(gt_pc)

        return np.array(gt_pc_list)

    def _action_sampler(self):
        # parameters
        collision_epsilon = 5e-4
        falldown_epsilon = 5e-4

        for _ in range(50):
            # choose object
            obj_id = np.random.choice(self.object_ids)

            # get position
            position = p.getBasePositionAndOrientation(obj_id)[0]
            position_xy = np.asarray([position[0], position[1]])
            position_z = position[2]

            # choose direction
            direction_idx = np.random.choice(self.num_directions)
            direction_angle = direction_idx / self.num_directions * 2 * np.pi

            # choose z coordinate
            obj_height = np.clip((2 * position_z - self.workspace_bounds[2, 0]) - 0.02, self.workspace_bounds[2, 0], 1)
            z_max_height = math.floor(((obj_height - self.workspace_bounds[2, 0]) / (self.workspace_bounds[2, 1] - self.workspace_bounds[2, 0])) * (self.num_z - 1))
            z = np.random.choice(z_max_height + 1)
            z_coord = (z / (self.num_z - 1)) * (self.workspace_bounds[2, 1] - 1e-10 - self.workspace_bounds[2, 0]) + self.workspace_bounds[2, 0]

            # get diagonal length and choose initial distance
            object_info = self.object_info[self.object_ids.index(obj_id)]
            if object_info['type'] == 'box':
                min_len = np.min(object_info['size'][:2])
            elif object_info['type'] == 'cylinder':
                min_len = np.mean(object_info['size'][:2])
            init_dist = math.ceil(min_len * 100) / 100 + 0.01
            init_dist += 0.02 * random.randrange(4)

            push_initial = np.append(position_xy - np.asarray([np.cos(direction_angle), np.sin(direction_angle)]) * init_dist, z_coord)
            x_coord, y_coord = push_initial[0], push_initial[1]

            # check if pushing point is inside workspace
            if not (x_coord >= self.workspace_bounds[0, 0] and x_coord < self.workspace_bounds[0, 1] and\
                    y_coord >= self.workspace_bounds[1, 0] and y_coord < self.workspace_bounds[1, 1] and\
                    z_coord >= self.workspace_bounds[2, 0] and y_coord < self.workspace_bounds[2, 1]):
                continue

            # get previous positions and orientations
            old_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
            old_ors = np.array([p.getBasePositionAndOrientation(object_id)[1] for object_id in self.object_ids])

            # go to initial pose for checking collision
            ik_solved = self.sim.down_action(position=push_initial, rotation_angle=direction_angle)

            # reset if robot cannot make the configuration
            if not ik_solved:
                self.sim.robot_go_home()
                self._reset_objects(old_pos, old_ors)
                continue

            # reset if collision is detected
            new_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
            position_diff = np.linalg.norm(new_pos - old_pos, axis=1).max()
            if position_diff > collision_epsilon:
                self.sim.up_action(position=push_initial, rotation_angle=direction_angle)
                self._reset_objects(old_pos, old_ors)
                continue

            # take action
            ik_solved = self.sim.push_action(position=push_initial, rotation_angle=direction_angle, speed=0.05, distance=0.10)
            self.sim.robot_go_home()

            # reset if robot cannot make the configuration
            if not ik_solved:
                self._reset_objects(old_pos, old_ors)
                continue

            # reset if object falls down
            new_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
            orientation_diff = np.abs(new_pos[:, 2] - old_pos[:, 2]).max()
            if orientation_diff > falldown_epsilon:
                self._reset_objects(old_pos, old_ors)
                continue

            # reset if any object goes outside workspace
            if not self._check_workspace():
                self._reset_objects(old_pos, old_ors)
                continue

            # convert coordinate to pixel
            x_pixel, y_pixel, z_pixel = self._coord2pixel(x_coord, y_coord, z_coord)

            return x_pixel, y_pixel, z_pixel, x_coord, y_coord, z_coord, direction_idx, direction_angle

        self.sim.robot_go_home()
        
        return None

    def _reset_objects(self, old_pos, old_ors):
        for idx, object_id in enumerate(self.object_ids):
            p.resetBasePositionAndOrientation(object_id, old_pos[idx], old_ors[idx])

    def _coord2pixel(self, x_coord, y_coord, z_coord):
        x_pixel = int((x_coord - self.workspace_bounds[0, 0]) / self.voxel_size)
        y_pixel = int((y_coord - self.workspace_bounds[1, 0]) / self.voxel_size)
        z_pixel = int((z_coord - self.workspace_bounds[2, 0]) / self.voxel_size)
        return x_pixel, y_pixel, z_pixel
    
    def _random_action_sampler(self):
        rotation_angle = np.pi
        position = [0.4, 0.1, 0.243]
        distance = 0.1
        speed = 0.01

        # target position
        push_orientation = [1.0, 0.0]
        push_direction = np.asarray(
            [push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
             push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle), 0.0])
        target_x = position[0] + push_direction[0] * distance
        target_y = position[1] + push_direction[1] * distance

        position_init = np.asarray([position[0], position[1], position[2] + self.sim.gripper_height]) \
                        - np.tan(self.sim.gripper_tilt) * self.sim.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])
        position_target = np.asarray([target_x, target_y, position[2] + self.sim.gripper_height]) \
                        - np.tan(self.sim.gripper_tilt) * self.sim.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])

        # align end-effector to pushing direction
        orientation = p.getQuaternionFromEuler([0, np.pi - self.sim.gripper_tilt, rotation_angle])
        
        # move to target position
        self.sim.move_tool(position_init, orientation=orientation, blocking=True, speed=speed)
        if (np.abs(p.getLinkState(self.sim._robot_body_id, self.sim._robot_tool_tip_joint_idx)[0] - position_init) > 0.1).any():
            return False

        # Use IK to compute target joint configuration
        target_joint_state = np.array(
            p.calculateInverseKinematics(self.sim._robot_body_id, self.sim._robot_tool_tip_joint_idx, position_target, orientation,
                                         maxNumIterations=10000, residualThreshold=.0001,
                                         lowerLimits=self.sim._robot_joint_lower_limit,
                                         upperLimits=self.sim._robot_joint_upper_limit))

        print('action start')

        # Move joints
        p.setJointMotorControlArray(self.sim._robot_body_id, self.sim._robot_joint_indices, p.POSITION_CONTROL,
                                    target_joint_state, positionGains=speed * np.ones(len(self.sim._robot_joint_indices)))

        # get actual joint state
        actual_joint_state = [p.getJointState(self.sim._robot_body_id, x)[0] for x in self.sim._robot_joint_indices]
        ee_state = p.getLinkState(self.sim._robot_body_id, 7)
        ee_rot_initial = deepcopy(np.asarray(p.getMatrixFromQuaternion(ee_state[5])).reshape(3,3))
        ee_pos = np.array(ee_state[4])
        timeout_t0 = time.time()
        z_list = []
        while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self.sim._joint_epsilon for i in
                        range(6)]):  # and (time.time()-timeout_t0) < timeout:
            if time.time() - timeout_t0 > 5:
                break

            # get out camera view matrix
            ee_state = p.getLinkState(self.sim._robot_body_id, 7)
            ee_rot = np.asarray(p.getMatrixFromQuaternion(ee_state[5])).reshape(3,3)
            ee_pos = np.array(ee_state[4])
            # ee_pose = get_SE3s(ee_rot, ee_pos)
            actual_joint_state = [p.getJointState(self.sim._robot_body_id, x)[0] for x in self.sim._robot_joint_indices]
            time.sleep(0.001)
            print(np.linalg.norm(ee_rot.dot(ee_rot_initial.T), 'fro'))
            z_list.append(ee_pos)

        z_list_numpy = np.array(z_list)
        z_min = np.min(z_list_numpy)
        z_max = np.max(z_list_numpy)

        raise ValueError('finished')

    def _random_action_sampler_straight(self):
        rotation_angle = np.pi
        position = [0.4, 0.1, 0.243]
        distance = 0.1
        speed = 0.01
        
        # num of segments
        n_segment = 10

        # target position
        push_orientation = [1.0, 0.0]
        push_direction = np.asarray(
            [push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
             push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle), 0.0])
        target_x = position[0] + push_direction[0] * distance
        target_y = position[1] + push_direction[1] * distance

        position_init = np.asarray([position[0], position[1], position[2] + self.sim.gripper_height]) \
                        - np.tan(self.sim.gripper_tilt) * self.sim.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])
        position_target = np.asarray([target_x, target_y, position[2] + self.sim.gripper_height]) \
                        - np.tan(self.sim.gripper_tilt) * self.sim.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])

        # align end-effector to pushing direction
        orientation = p.getQuaternionFromEuler([0, np.pi - self.sim.gripper_tilt, rotation_angle])
        
        # move to target position
        self.sim.move_tool(position_init, orientation=orientation, blocking=True, speed=speed)
        if (np.abs(p.getLinkState(self.sim._robot_body_id, self.sim._robot_tool_tip_joint_idx)[0] - position_init) > 0.1).any():
            return False

        # # define segments of straight line motion
        # ee_state = p.getLinkState(self.sim._robot_body_id, 7)
        # position_initial = deepcopy(np.array(ee_state[4]))

        target_joint_state_list = []
        for i in range(n_segment):
            # Use IK to compute target joint configuration
            target_joint_state = np.array(
            p.calculateInverseKinematics(self.sim._robot_body_id, self.sim._robot_tool_tip_joint_idx, position_init + (position_target - position_init) * (i + 1) / n_segment, 
                                         orientation,
                                         maxNumIterations=10000, 
                                         residualThreshold=.0001,
                                         lowerLimits=self.sim._robot_joint_lower_limit,
                                         upperLimits=self.sim._robot_joint_upper_limit))

            print('action start')

            # Move joints
            p.setJointMotorControlArray(self.sim._robot_body_id, self.sim._robot_joint_indices, p.POSITION_CONTROL,
                                        target_joint_state, positionGains=speed * np.ones(len(self.sim._robot_joint_indices)))

            # get actual joint state
            actual_joint_state = [p.getJointState(self.sim._robot_body_id, x)[0] for x in self.sim._robot_joint_indices]
            ee_state = p.getLinkState(self.sim._robot_body_id, 7)
            ee_rot_initial = deepcopy(np.asarray(p.getMatrixFromQuaternion(ee_state[5])).reshape(3,3))
            ee_pos = np.array(ee_state[4])
            timeout_t0 = time.time()
            z_list = []
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self.sim._joint_epsilon for i in
                            range(6)]):  # and (time.time()-timeout_t0) < timeout:
                if time.time() - timeout_t0 > 5:
                    break

                # get out camera view matrix
                ee_state = p.getLinkState(self.sim._robot_body_id, 7)
                ee_rot = np.asarray(p.getMatrixFromQuaternion(ee_state[5])).reshape(3,3)
                ee_pos = np.array(ee_state[4])
                # ee_pose = get_SE3s(ee_rot, ee_pos)
                actual_joint_state = [p.getJointState(self.sim._robot_body_id, x)[0] for x in self.sim._robot_joint_indices]
                time.sleep(0.001)
                print(np.linalg.norm(ee_rot.dot(ee_rot_initial.T), 'fro'))
                z_list.append(ee_pos[2])

        z_list_numpy = np.array(z_list)
        z_min = np.min(z_list_numpy)
        z_max = np.max(z_list_numpy)
        print(z_min, z_max)

        raise ValueError('finished')
        


if __name__ == '__main__':
    env = SimulationEnv(enable_gui=False)
    env.reset(4, ['box', 'cylinder'])

    # if you just want to get the information of the scene, use env._get_scene_info
    output = env._get_scene_info_before()
    print(output.keys())

    # if use the pushing. env.poke() will also give you everything, together with scene flow
    output = env.poke()
    print(output.keys())
