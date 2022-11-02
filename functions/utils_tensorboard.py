import numpy as np
import open3d as o3d

from functions.utils import get_SE3s, quats_to_matrices
from functions.primitives import Superquadric, DeformableSuperquadric

rgb_colors = {
    "black": [0, 0, 0],
    "red": [255, 0, 0],
    "pink": [255, 96, 208],
    "purple": [160, 32, 255],
    "light_blue": [80, 208, 255],
    "blue": [0, 32, 255],
    "green": [0, 192, 0],
    "orange": [255, 160, 16],
    "brown": [160, 128, 96],
    "gray": [128, 128, 128],
}

################################################
################ Segmentation ##################
################################################

def color_pc_segmentation(label):
    class_idx_label = np.argmax(label, axis=2)

    pc_colors = np.zeros((label.shape[0], label.shape[1], 3))
    for batch_idx in range(pc_colors.shape[0]):
        for point_idx in range(pc_colors.shape[1]):
            pc_colors[batch_idx, point_idx, :] = list(rgb_colors.values())[class_idx_label[batch_idx, point_idx]]

    return pc_colors

################################################
################ Recognition ###################
################################################

def pc_with_coordinates(pc, color_list=None):
    # make coordinate frame
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    pc_coor_repeat = np.tile(np.asarray(coordinate.vertices), (pc.shape[0], 1, 1))
    color_coor_repeat = np.tile(np.asarray(coordinate.vertex_colors), (pc.shape[0], 1, 1))

    # concatenate
    pc_total = np.concatenate((pc[:, :, :3], pc_coor_repeat), axis=1)
    if color_list is None:
        if pc.shape[2] == 4:
            # color template
            rgb_colors = np.array([
                [0, 0, 0],
                [255, 0, 0],
                [255, 96, 208],
                [160, 32, 255],
                [80, 208, 255],
                [0, 32, 255],
                [0, 192, 0],
                [255, 160, 16],
                [160, 128, 96],
                [128, 128, 128],
            ])

            # processing
            bs = pc.shape[0]
            npoints = pc.shape[1]
            labels = pc[:, :, 3]
            color_total = rgb_colors[labels.astype(np.int32).reshape(-1)].reshape(bs, npoints, 3)
            color_total = np.concatenate((color_total, 255 * color_coor_repeat), axis=1)
        elif pc.shape[2] == 3:
            color_total = np.concatenate((128 * np.ones(np.shape(pc)), 255 * color_coor_repeat), axis=1)

        return pc_total, color_total
    else:
        color_total_list = []
        for color in color_list:
            color_total = np.concatenate((color, 255*color_coor_repeat), axis=1)
            color_total_list += [color_total]
    
        return pc_total, color_total_list

def mesh_generator_recog_pred(outputs, resolution=30, transform=True):
    shape_position = outputs[:, :3]
    shape_orientation = quats_to_matrices(outputs[:, 3:7])
    shape_parameters = outputs[:, 7:]
    
    mesh_list = []
    for idx in range(len(outputs)):
        SE3 = get_SE3s(shape_orientation[idx, :, :], shape_position[idx, :])
        parameters = dict()
        parameters['a1'] = shape_parameters[idx, 0]
        parameters['a2'] = shape_parameters[idx, 1]
        parameters['a3'] = shape_parameters[idx, 2]
        parameters['e1'] = shape_parameters[idx, 3]
        parameters['e2'] = shape_parameters[idx, 4]
        if shape_parameters.shape[1] > 5: # deformable superquadric
            parameters['k'] = shape_parameters[idx, 5]
            parameters['b'] = shape_parameters[idx, 6]
            parameters['cos_alpha'] = shape_parameters[idx, 7]
            parameters['sin_alpha'] = shape_parameters[idx, 8]
            mesh = DeformableSuperquadric(SE3, parameters, resolution=resolution, transform=transform).mesh
        else: # superquadric
            mesh = Superquadric(SE3, parameters, transform=transform).mesh

        mesh_list.append(mesh)

    return mesh_list


def mesh_generator_recog_gt(object_info, positions, orientations, means_xyz=None, diagonal_lengths=None, resolution=30, split=5):
    Rs = quats_to_matrices(orientations)

    mesh_list = []
    for obj_idx, (obj_info, position, R) in enumerate(zip(object_info, positions, Rs)):
        if type(obj_info) == str:
            obj_info = eval(obj_info)

        # Get meshes
        if obj_info['type'] == 'box':
            [width, height, depth] = obj_info['size']
            mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
            mesh.translate([-width/2, -height/2, -depth/2]) # match center to the origin

        elif obj_info['type'] == 'cylinder':
            [radius, _, height] = obj_info['size']
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution, split=split)

        # Move mesh using gt transform
        T = get_SE3s(R, position)
        mesh.transform(T)

        # Normalize mesh
        if means_xyz is not None and diagonal_lengths is not None:
            mean_xyz = means_xyz[obj_idx]
            diagonal_len = diagonal_lengths[obj_idx]

            mesh.translate(-mean_xyz)
            mesh.scale(1/diagonal_len, center=[0]*3)

        mesh_list.append(mesh)
    
    return mesh_list

def two_meshes_to_numpy(mesh1, mesh2, color1=[0, 0, 1], color2=[0, 1, 0]):
    if len(mesh1) != len(mesh2):
        raise ValueError("Mesh1 and mesh2 do not have same batch size")

    total_pointclouds = []
    total_faces = []
    total_colors = []

    max_num_pointclouds = 0
    max_num_faces = 0

    for batch in range(len(mesh1)):
        # color painting
        mesh1[batch].paint_uniform_color(color1)
        mesh2[batch].paint_uniform_color(color2)

        # combine meshes and add coordinate
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        mesh = coordinate + mesh1[batch] + mesh2[batch]

        pointclouds = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = 255 * np.asarray(mesh.vertex_colors)

        # append for batch
        total_pointclouds.append(pointclouds)
        total_faces.append(faces)
        total_colors.append(colors)

        # find the maximum dimension
        if pointclouds.shape[0] > max_num_pointclouds:
            max_num_pointclouds = pointclouds.shape[0]
        if faces.shape[0] > max_num_faces:
            max_num_faces = faces.shape[0]

    # matching dimension between batches for tensorboard
    for batch in range(len(mesh1)):
        diff_num_pointclouds = max_num_pointclouds - total_pointclouds[batch].shape[0]
        diff_num_faces = max_num_faces - total_faces[batch].shape[0]
        total_pointclouds[batch] = np.concatenate((total_pointclouds[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)
        total_faces[batch] = np.concatenate((total_faces[batch], np.zeros((diff_num_faces, 3))), axis=0)
        total_colors[batch] = np.concatenate((total_colors[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)

    return np.asarray(total_pointclouds), np.asarray(total_faces), np.asarray(total_colors)

################################################
################# Dynamics #####################
################################################

def mesh_generator_motion(scenes, 
                          actions, 
                          motions, 
                          target_color=[1, 0, 0], 
                          moved_color=[0, 1, 0], 
                          resolution=7,
                          vis_table=False,
                          vis_global_coord=False,
                          vis_object_coord=False):

    # initialize
    mesh_list = []

    for scene, action, motion in zip(scenes, actions, motions):
        # select real objects whose confidene is 1
        shape_idxs = scene[0] == 1
        
        shape_positions = scene[1:4, shape_idxs].transpose()
        shape_Rs = quats_to_matrices(scene[4:8, shape_idxs].transpose())
        shape_parameters = scene[8:, shape_idxs].transpose()

        # action mesh
        mesh_scene = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.015, cylinder_height=0.05, cone_height=0.04, resolution=resolution
        )
        mesh_scene.paint_uniform_color([1, 0, 0])
        mesh_scene.translate([0, 0, -0.05-0.04])
        arrow_SE3 = np.array([[0, -action[4], action[3], action[0]],
                              [0, action[3], action[4], action[1]],
                              [-1, 0, 0, action[2]],
                              [0, 0, 0, 1]])
        mesh_scene.transform(arrow_SE3)

        # table
        if vis_table:
            table = o3d.geometry.TriangleMesh.create_box(width = 0.44, height = 0.6, depth = 0.001)
            table.translate([0.22, -0.3, -0.012 - 0.0005])
            table.paint_uniform_color([0, 0, 0])
            mesh_scene += table

        # coordinate
        if vis_global_coord:
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            mesh_scene += coordinate

        shape_SE3s = get_SE3s(shape_Rs, shape_positions)

        # superquadric meshes
        for shape_idx, (SE3, shape_parameter) in enumerate(zip(shape_SE3s, shape_parameters)):
            
            # parameter processing
            parameters = dict()
            parameters['a1'] = shape_parameter[0]
            parameters['a2'] = shape_parameter[1]
            parameters['a3'] = shape_parameter[2]
            parameters['e1'] = shape_parameter[3]
            parameters['e2'] = shape_parameter[4]
            if shape_parameters.shape[1] > 5: # deformable superquadric
                parameters['k'] = shape_parameter[5]
                parameters['b'] = shape_parameter[6]
                parameters['cos_alpha'] = shape_parameter[7]
                parameters['sin_alpha'] = shape_parameter[8]
                mesh = DeformableSuperquadric(SE3, parameters, resolution=resolution).mesh
            else: # superquadric
                mesh = Superquadric(SE3, parameters, resolution=resolution).mesh

            if shape_idx == 0:
                mesh.paint_uniform_color(target_color)
            else:
                mesh.paint_uniform_color([0.7, 0.7, 0.7])

            mesh_scene += mesh
            if vis_object_coord:
                coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
                coordinate.transform(SE3)
                mesh_scene += coordinate
                
            # motion 
            if shape_idx == 0:
                # planar motion
                if len(motion) == 3:
                    motion_SE3 = np.array([[np.cos(motion[2]), -np.sin(motion[2]), 0, motion[0]],
                                           [np.sin(motion[2]), np.cos(motion[2]), 0, motion[1]],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
                # 3D motion
                elif len(motion) == 7:
                    motion_SE3 = get_SE3s(quats_to_matrices(motion[3:7]), motion[:3])
                else:
                    raise ValueError('check the dimension of the motion')

                mesh.transform(np.linalg.inv(SE3))
                mesh.transform(SE3.dot(motion_SE3))
                mesh.paint_uniform_color(moved_color)
                mesh_scene += mesh

        mesh_list.append(mesh_scene)

    return mesh_list


def meshes_to_numpy(meshes):
    total_pointclouds = []
    total_faces = []
    total_colors = []

    max_num_pointclouds = 0
    max_num_faces = 0

    for batch in range(len(meshes)):
        mesh = meshes[batch]

        pointclouds = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = 255 * np.asarray(mesh.vertex_colors)

        # append for batch
        total_pointclouds.append(pointclouds)
        total_faces.append(faces)
        total_colors.append(colors)

        # find the maximum dimension
        if pointclouds.shape[0] > max_num_pointclouds:
            max_num_pointclouds = pointclouds.shape[0]
        if faces.shape[0] > max_num_faces:
            max_num_faces = faces.shape[0]

    # matching dimension between batches for tensorboard
    for batch in range(len(meshes)):
        diff_num_pointclouds = max_num_pointclouds - total_pointclouds[batch].shape[0]
        diff_num_faces = max_num_faces - total_faces[batch].shape[0]
        total_pointclouds[batch] = np.concatenate((total_pointclouds[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)
        total_faces[batch] = np.concatenate((total_faces[batch], np.zeros((diff_num_faces, 3))), axis=0)
        total_colors[batch] = np.concatenate((total_colors[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)

    return np.asarray(total_pointclouds), np.asarray(total_faces), np.asarray(total_colors)
