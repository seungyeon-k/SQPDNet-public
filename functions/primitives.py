import numpy as np
import open3d as o3d


class BasePrimitives:
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], resolution=30):
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.resolution = resolution

    def process_mesh(self):
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color(self.color)
        
    def transform_mesh(self):
        self.mesh.transform(self.SE3)


class Superquadric(BasePrimitives):
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], resolution=30, transform=True):
        super(Superquadric, self).__init__(SE3, parameters, color, resolution)
        self.type = 'superquadric'
        self.mesh = mesh_superquadric(self.parameters, self.SE3, self.resolution)
        self.process_mesh()
        if transform:
            self.transform_mesh()


class DeformableSuperquadric(BasePrimitives):
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], resolution=30, transform=True):
        super(Superquadric, self).__init__(SE3, parameters, color, resolution)
        self.type = 'deformable_superquadric'
        self.mesh = mesh_deformable_superquadric(self.parameters, self.SE3, self.resolution)
        self.process_mesh()
        if transform:
            self.transform_mesh()


def mesh_superquadric(parameters, SE3, resolution=30):
    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    e1 = parameters['e1']
    e2 = parameters['e2']
    R = SE3[0:3, 0:3]
    t = SE3[0:3, 3:]

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=resolution)
    vertices_numpy = np.asarray(mesh.vertices)
    eta = np.arcsin(vertices_numpy[:, 2:3])
    omega = np.arctan2(vertices_numpy[:, 1:2], vertices_numpy[:, 0:1])

    # make new vertices
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)

    mesh.vertices = o3d.utility.Vector3dVector(points)

    return mesh


def mesh_deformable_superquadric(parameters, SE3, resolution=30):
    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    e1 = parameters['e1']
    e2 = parameters['e2']
    if 'k' in parameters.keys():
        k = parameters['k']
    if 'b' in parameters.keys():
        b = parameters['b'] / np.maximum(a1, a2)
        cos_alpha = parameters['cos_alpha']
        sin_alpha = parameters['sin_alpha']
        alpha = np.arctan2(sin_alpha, cos_alpha)

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=resolution)
    vertices_numpy = np.asarray(mesh.vertices)
    eta = np.arcsin(vertices_numpy[:, 2:3])
    omega = np.arctan2(vertices_numpy[:, 1:2], vertices_numpy[:, 0:1])

    # make new vertices
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    points = np.concatenate((x, y, z), axis=1)
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh = mesh.subdivide_midpoint(2)

    points = np.asarray(mesh.vertices)
    x = points[:, 0:1]
    y = points[:, 1:2]
    z = points[:, 2:3]

    # tampering
    if 'k' in parameters.keys():
        f_x = k / a3 * z + 1
        f_y = k / a3 * z + 1
        x = f_x * x
        y = f_y * y

    # bending
    if 'b' in parameters.keys():
        gamma = z * b
        r = np.cos(alpha - np.arctan2(y, x)) * np.sqrt(x ** 2 + y ** 2)
        R = 1 / b - np.cos(gamma) * (1 / b - r)
        x = x + np.cos(alpha) * (R - r)
        y = y + np.sin(alpha) * (R - r)
        z = np.sin(gamma) * (1 / b - r)

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)

    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()

    return mesh


def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)


gen_primitive = {
    "superquadric": Superquadric,
    "deformable_superquadric": DeformableSuperquadric,
}

gen_parameter = {
    "superquadric": ['a1', 'a2', 'a3', 'e1', 'e2'],
    "deformable_superquadric": ['a1', 'a2', 'a3', 'e1', 'e2', 'k', 'b', 'cos_alpha', 'sin_alpha'],
}