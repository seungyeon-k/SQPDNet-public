import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from datetime import datetime

from functions.primitives import DeformableSuperquadric, Superquadric

def objects_renderer_with_pc(pc, Ts, parameters, resolution=10, image_size=[600, 960]):
    
    # list
    mesh_list = []

    # point cloud
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc.T)
    mesh_list.append(pc_o3d)

    # add geometries and lighting
    for id, (SE3, parameter) in enumerate(zip(Ts, parameters)):
        parameters_dict = dict()
        parameters_dict['a1'] = parameter[0]
        parameters_dict['a2'] = parameter[1]
        parameters_dict['a3'] = parameter[2]
        parameters_dict['e1'] = parameter[3]
        parameters_dict['e2'] = parameter[4]

        X = Superquadric(SE3, parameters_dict, resolution=resolution).mesh
        mesh_list.append(X)

    o3d.visualization.draw_geometries(mesh_list)

def objects_renderer(Ts, parameters, action=None, resolution=10, image_size=[600, 960]):

    # define ground plane
    a = 10.0
    plane = o3d.geometry.TriangleMesh.create_box(width=a, depth=0.05, height=a)
    plane.paint_uniform_color([1.0, 1.0, 1.0])
    plane.translate([-a/2, -a/2, -0.05])
    plane.compute_vertex_normals()
    mat_plane = rendering.MaterialRecord()
    mat_plane.shader = 'defaultLit'
    mat_plane.base_color = [1.0, 1.0, 1.0, 4.0]

    # object material
    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLitTransparency'
    mat.base_color = [1.0, 0.0, 1.0, 0.9]

    # # for transparent object (practice, example)
    # mat_box = vis.rendering.MaterialRecord()
    # # mat_box.shader = 'defaultLitTransparency'
    # mat_box.shader = 'defaultLitSSR'
    # mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
    # mat_box.base_roughness = 0.0
    # mat_box.base_reflectance = 0.0
    # mat_box.base_clearcoat = 1.0
    # mat_box.thickness = 1.0
    # mat_box.transmission = 1.0
    # mat_box.absorption_distance = 10
    # mat_box.absorption_color = [0.5, 0.5, 0.5]

    # set window
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1])
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget) 

    # coordinate
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # action
    if action is not None:
        mesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.015, cylinder_height=0.05, cone_height=0.04, resolution=resolution
        )
        mesh.paint_uniform_color([1, 0, 0])
        mesh.translate([0, 0, -0.05-0.04])
        arrow_SE3 = np.array([[0, -np.sin(action[0]), np.cos(action[0]), action[1]],
                            [0, np.cos(action[0]), np.sin(action[0]), action[2]],
                            [-1, 0, 0, action[3]],
                            [0, 0, 0, 1]])
        mesh.transform(arrow_SE3)
        widget.scene.add_geometry(f'mesh_action', mesh, mat)

    # add geometries and lighting
    for id, (SE3, parameter) in enumerate(zip(Ts, parameters)):
        parameters_dict = dict()
        parameters_dict['a1'] = parameter[0]
        parameters_dict['a2'] = parameter[1]
        parameters_dict['a3'] = parameter[2]
        parameters_dict['e1'] = parameter[3]
        parameters_dict['e2'] = parameter[4]

        X = Superquadric(SE3, parameters_dict, resolution=resolution).mesh
        widget.scene.add_geometry(f'mesh{id}', X, mat)
    widget.scene.add_geometry('frame', frame, mat)
    widget.scene.add_geometry('plane', plane, mat_plane)
    widget.scene.set_lighting(widget.scene.LightingProfile.DARK_SHADOWS, (0.3, -0.3, -0.9))
    widget.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

    gui.Application.instance.run()


def objects_renderer_baf(Ts, delta_Ts, parameters, action, resolution=10, image_size=[600, 960]):
    
    # define ground plane
    a = 10.0
    plane = o3d.geometry.TriangleMesh.create_box(width=a, depth=0.05, height=a)
    plane.paint_uniform_color([1.0, 1.0, 1.0])
    plane.translate([-a/2, -a/2, -0.05])
    plane.compute_vertex_normals()
    mat_plane = rendering.MaterialRecord()
    mat_plane.shader = 'defaultLit'
    mat_plane.base_color = [1.0, 1.0, 1.0, 4.0]

    # object material (before)
    mat_b = rendering.MaterialRecord()
    mat_b.shader = 'defaultLitTransparency'
    mat_b.base_color = [1.0, 0.0, 1.0, 0.9]

    # object material (after)
    mat_a = rendering.MaterialRecord()
    mat_a.shader = 'defaultLitTransparency'
    mat_a.base_color = [0.0, 1.0, 1.0, 0.9]

    # # for transparent object (practice, example)
    # mat_box = vis.rendering.MaterialRecord()
    # # mat_box.shader = 'defaultLitTransparency'
    # mat_box.shader = 'defaultLitSSR'
    # mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
    # mat_box.base_roughness = 0.0
    # mat_box.base_reflectance = 0.0
    # mat_box.base_clearcoat = 1.0
    # mat_box.thickness = 1.0
    # mat_box.transmission = 1.0
    # mat_box.absorption_distance = 10
    # mat_box.absorption_color = [0.5, 0.5, 0.5]

    # set window
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1])
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget) 

    # coordinate
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # action
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01, cone_radius=0.015, cylinder_height=0.05, cone_height=0.04, resolution=resolution
    )
    mesh.paint_uniform_color([1, 0, 0])
    mesh.translate([0, 0, -0.05-0.04])
    arrow_SE3 = np.array([[0, -np.sin(action[0]), np.cos(action[0]), action[1]],
                        [0, np.cos(action[0]), np.sin(action[0]), action[2]],
                        [-1, 0, 0, action[3]],
                        [0, 0, 0, 1]])
    mesh.transform(arrow_SE3)
    widget.scene.add_geometry(f'mesh_action', mesh, mat_b)

    # add geometries and lighting
    for id, (SE3, delta_SE3, parameter) in enumerate(zip(Ts, delta_Ts, parameters)):
        parameters_dict = dict()
        parameters_dict['a1'] = parameter[0]
        parameters_dict['a2'] = parameter[1]
        parameters_dict['a3'] = parameter[2]
        parameters_dict['e1'] = parameter[3]
        parameters_dict['e2'] = parameter[4]

        X = Superquadric(SE3, parameters_dict, resolution=resolution).mesh
        Y = Superquadric(SE3 @ delta_SE3, parameters_dict, resolution=resolution).mesh
        widget.scene.add_geometry(f'mesh_b_{id}', X, mat_b)
        widget.scene.add_geometry(f'mesh_a_{id}', Y, mat_a)
    widget.scene.add_geometry('frame', frame, mat_b)
    widget.scene.add_geometry('plane', plane, mat_plane)
    widget.scene.set_lighting(widget.scene.LightingProfile.DARK_SHADOWS, (0.3, -0.3, -0.9))
    widget.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

    gui.Application.instance.run()

def objects_renderer_three(T1, T2, parameters, resolution=10, image_size=[600, 960]):
    
    # define ground plane
    a = 10.0
    plane = o3d.geometry.TriangleMesh.create_box(width=a, depth=0.05, height=a)
    plane.paint_uniform_color([1.0, 1.0, 1.0])
    plane.translate([-a/2, -a/2, -0.05])
    plane.compute_vertex_normals()
    mat_plane = rendering.MaterialRecord()
    mat_plane.shader = 'defaultLit'
    mat_plane.base_color = [1.0, 1.0, 1.0, 4.0]

    # object material (before)
    mat_b = rendering.MaterialRecord()
    mat_b.shader = 'defaultLitTransparency'
    mat_b.base_color = [1.0, 0.0, 1.0, 0.9]

    # object material (after)
    mat_a = rendering.MaterialRecord()
    mat_a.shader = 'defaultLitTransparency'
    mat_a.base_color = [0.0, 1.0, 1.0, 0.9]

    # set window
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1])
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget) 

    # coordinate
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # add geometries and lighting
    for id, (SE3_1, SE3_2, parameter) in enumerate(zip(T1, T2, parameters)):
        parameters_dict = dict()
        parameters_dict['a1'] = parameter[0]
        parameters_dict['a2'] = parameter[1]
        parameters_dict['a3'] = parameter[2]
        parameters_dict['e1'] = parameter[3]
        parameters_dict['e2'] = parameter[4]

        X = Superquadric(SE3_1, parameters_dict, resolution=resolution).mesh
        Y = Superquadric(SE3_2, parameters_dict, resolution=resolution).mesh
        widget.scene.add_geometry(f'mesh_b_{id}', X, mat_b)
        widget.scene.add_geometry(f'mesh_a_{id}', Y, mat_a)
    widget.scene.add_geometry('frame', frame, mat_b)
    widget.scene.add_geometry('plane', plane, mat_plane)
    widget.scene.set_lighting(widget.scene.LightingProfile.DARK_SHADOWS, (0.3, -0.3, -0.9))
    widget.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

    gui.Application.instance.run()


def action_renderer(Ts, parameters, all_actions, actions_valid, scores, scenes, best_idx, resolution=30):
    # initialize
    mesh_actions_all = o3d.geometry.TriangleMesh()
    mesh_action_valid = o3d.geometry.TriangleMesh()
    mesh_motion = o3d.geometry.TriangleMesh()

    # make scene meshes
    mesh_scene = o3d.geometry.TriangleMesh()
    for parameter, T in zip(parameters, Ts):
        # original primitives
        # parameter processing
        parameter_dict = dict()
        parameter_dict['a1'] = parameter[0]
        parameter_dict['a2'] = parameter[1]
        parameter_dict['a3'] = parameter[2]
        parameter_dict['e1'] = parameter[3]
        parameter_dict['e2'] = parameter[4]
        mesh = Superquadric(T, parameter_dict, resolution=resolution).mesh

        mesh.paint_uniform_color([0.7, 0.7, 0.7])

        mesh_scene += mesh

    mesh_actions_all += mesh_scene
    mesh_action_valid += mesh_scene
    mesh_motion += mesh_scene

    # make all action arrows
    for action in all_actions:
        action = action[0]

        mesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.015, cylinder_height=0.05, cone_height=0.04, resolution=resolution
        )
        mesh.paint_uniform_color([1, 0, 0])
        mesh.translate([0, 0, -0.05-0.04])
        arrow_SE3 = np.array([[0, -np.sin(action[0]), np.cos(action[0]), action[1]],
                            [0, np.cos(action[0]), np.sin(action[0]), action[2]],
                            [-1, 0, 0, action[3]],
                            [0, 0, 0, 1]])
        mesh.transform(arrow_SE3)

        mesh_actions_all += mesh

    # make action arrows
    for action, score in zip(actions_valid, scores):
        action = action[0]

        mesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.015, cylinder_height=0.05, cone_height=0.04, resolution=resolution
        )
        point = ((score - min(scores)) / (max(scores) - min(scores)))
        mesh.paint_uniform_color([1, 0.9 * (1 - point), 0.9 * (1 - point)])
        mesh.translate([0, 0, -0.05-0.04])
        arrow_SE3 = np.array([[0, -np.sin(action[0]), np.cos(action[0]), action[1]],
                            [0, np.cos(action[0]), np.sin(action[0]), action[2]],
                            [-1, 0, 0, action[3]],
                            [0, 0, 0, 1]])
        mesh.transform(arrow_SE3)

        mesh_action_valid += mesh

    # get best action and scene
    best_action = actions_valid[best_idx][0]
    best_scene = scenes[best_idx][0]
    
    # make best action arrow
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01, cone_radius=0.015, cylinder_height=0.05, cone_height=0.04, resolution=resolution
    )
    mesh.paint_uniform_color([1, 0, 0])
    mesh.translate([0, 0, -0.05-0.04])
    arrow_SE3 = np.array([[0, -np.sin(best_action[0]), np.cos(best_action[0]), best_action[1]],
                        [0, np.cos(best_action[0]), np.sin(best_action[0]), best_action[2]],
                        [-1, 0, 0, best_action[3]],
                        [0, 0, 0, 1]])
    mesh.transform(arrow_SE3)

    mesh_motion += mesh

    for parameter, T in zip(parameters, best_scene):
        # original primitives
        # parameter processing
        parameter_dict = dict()
        parameter_dict['a1'] = parameter[0]
        parameter_dict['a2'] = parameter[1]
        parameter_dict['a3'] = parameter[2]
        parameter_dict['e1'] = parameter[3]
        parameter_dict['e2'] = parameter[4]
        if parameters.shape[1] > 5: # deformable superquadric
            parameter_dict['k'] = parameter[5]
            parameter_dict['b'] = parameter[6]
            parameter_dict['cos_alpha'] = parameter[7]
            parameter_dict['sin_alpha'] = parameter[8]
            mesh = DeformableSuperquadric(T, parameter_dict, resolution=resolution).mesh
        else: # superquadric
            mesh = Superquadric(T, parameter_dict, resolution=resolution).mesh

        mesh.paint_uniform_color([0, 0, 1])

        mesh_motion += mesh
        
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="All Actions", width=960-2, height=1040-32, left=960*0+1, top=31)
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Action Scores", width=960-2, height=1040-32, left=960*1+1, top=31)
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name="Best Action's Motion Prediction", width=960-2, height=1040-32, left=960*2+1, top=31)

    vis1.add_geometry(mesh_actions_all)
    vis2.add_geometry(mesh_action_valid)
    vis3.add_geometry(mesh_motion)

    while True:
        vis1.update_geometry(mesh_actions_all)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(mesh_action_valid)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
        
        vis3.update_geometry(mesh_motion)
        if not vis3.poll_events():
            break
        vis3.update_renderer()
