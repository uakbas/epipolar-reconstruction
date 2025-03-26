"""
This script is used to postprocess the blender output.
(not important in here) Blender camera coordinate system convention: (x: right, y:up, -z:forward)
Blender world coordinate system convention (when we stay at the front camera location):
    x: right | y: forward --> moving away from us | z: up --> looking up to the sky
"""
import os
import torch
import trimesh
import itertools
import pyvista as pv
import open3d as o3d
import numpy as np
from pyglet.media.drivers import driver_name
from pyvista.demos.logo import atomize
from camera import get_rot_mat_x, create_transformation_matrix, homogenize
import time


def scale_volume_occupancy(vo, scale=1):
    """Scale volume occupancy by a given scaling factor.
    TODO Fix code duplication.
    """
    volume_dims = torch.as_tensor(vo.shape, dtype=torch.int)

    assert scale == 1 or scale % 2 == 0, 'Invalid scale value.'
    assert (volume_dims % scale == 0).all(), 'Shape values must be multiple of the scale.'

    shape = torch.stack([volume_dims // scale, torch.full_like(volume_dims, scale)], dim=0).T.flatten()
    vo_scaled = vo.view(*shape).sum(dim=(1, 3, 5)) > 0  # ((scale / 3) ** 3)  # > 0
    return vo_scaled.to(torch.int32)


def visualize_volume_occupancy(vo, scale=1):
    """Visualize the volume occupancy grid using trimesh.
    TODO Fix code duplication.
    """
    vo_scaled = scale_volume_occupancy(vo, scale)

    # Fill volume edges for better visualization.
    for cord1, cord2 in list(itertools.product([0, -1], [0, -1])):
        vo_scaled[:, cord1, cord2] = 1
        vo_scaled[cord1, :, cord2] = 1
        vo_scaled[cord1, cord2, :] = 1

    # Just for visualization. Swap x and y axes for VoxelGrid.
    transform = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    vg = trimesh.voxel.VoxelGrid(vo_scaled.numpy(), transform=transform)

    scene = trimesh.Scene()
    scene.add_geometry(vg.as_boxes())
    scene.add_geometry(trimesh.creation.axis(axis_radius=2, axis_length=min(vo_scaled.shape[0], 10)))
    scene.show(resolution=(1000, 1000))


def visualize_by_pyvista(path, use_atomize=False):
    voxels = pv.voxelize(pv.read(path), density=2, check_surface=False)

    if use_atomize:
        object_voxels_atomized = atomize(voxels, scale=0.5)
        object_voxels_atomized.plot()
        return

    plotter = pv.Plotter()
    plotter.add_mesh(voxels, color="red", point_size=1, render_points_as_spheres=True)
    plotter.show()


def visualize_points_as_voxels(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])


def get_voxel_points_by_pyvista(path):
    """Load object, voxelize and return the voxel points by pyvista."""
    object_mesh = pv.read(path)
    object_voxels = pv.voxelize(object_mesh, density=2, check_surface=False)
    object_points = torch.as_tensor(object_voxels.points, dtype=torch.float32)
    return object_points


def get_voxel_points(path):
    mesh = o3d.t.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()

    bbox = mesh.get_axis_aligned_bounding_box()

    min_bound, max_bound = np.floor(bbox.min_bound.numpy()), np.ceil(bbox.max_bound.numpy())
    # print(f"Min bound: {min_bound}, Max bound: {max_bound}")

    density = 2
    x_vals = np.arange(min_bound[0], max_bound[0], density)
    y_vals = np.arange(min_bound[1], max_bound[1], density)
    z_vals = np.arange(min_bound[2], max_bound[2], density)

    # print(f"X: {len(x_vals)}, Y: {len(y_vals)}, Z: {len(z_vals)}")

    # Indexing could be 'ij'. The result will not be affected since the points created are the same.
    xv, yv, zv = np.meshgrid(x_vals, y_vals, z_vals, indexing='xy')
    grid_points = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    directions = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    rays = np.zeros((grid_points.shape[0], directions.shape[0], 6), dtype="float32")
    rays[:, :, :3] = np.expand_dims(grid_points, axis=1)
    rays[:, :, 3:] = directions

    hits = scene.cast_rays(o3d.core.Tensor(rays))
    inside_mask = np.all((hits['t_hit'].numpy() != np.inf), axis=1)
    inside_points = grid_points[inside_mask]

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(inside_points)
    # o3d.visualization.draw_geometries([point_cloud])

    return inside_points



def process():
    volume_radius = 384  # Distance from origin to each face of the volume.
    volume_scale = 12  # Scale factor for the volume occupancy grid.
    root = "/Users/uveyisakbas/Desktop/blender"
    dataset_dir_path = os.path.join(root, "dataset")
    visualize = False

    """
        Transformation matrix to convert from Blender coordinates to volume occupancy coordinates.
        Origin for volume occupancy coordinate system is at the top-left-front of the volume.
        Y goes down, x goes right, z goes back.
        This transformation matrix helps to get voxels point coordinates in the volume occupancy coordinate system.
    """
    trans_mat = create_transformation_matrix(
        get_rot_mat_x(270),
        torch.tensor([-volume_radius, -volume_radius, volume_radius], dtype=torch.float32)
    )

    for directory in sorted([directory for directory in os.listdir(dataset_dir_path) if not directory.startswith(".")]):
        print(f"Processing scene: {directory}")

        scene_dir = os.path.join(dataset_dir_path, directory)
        object_path = os.path.join(scene_dir, "scene.obj")

        time_start_voxelize = time.time()
        object_points = get_voxel_points(object_path)
        # print(f"Voxelize time: {time.time() - time_start_voxelize}")

        # Convert object points to voxel indexes.
        time_start_indexing = time.time()
        object_points = torch.as_tensor(object_points, dtype=torch.float32)
        occupied_voxel_indexes = torch.round((trans_mat @ homogenize(object_points.T)).T).to(dtype=torch.int)
        # print(f"Indexing time: {time.time() - time_start_indexing}")

        # Check if the voxel points are within the volume occupancy grid.
        if not torch.all(torch.logical_and(0 <= occupied_voxel_indexes, occupied_voxel_indexes < volume_radius * 2)):
            print(f"Invalid scene: {directory}\n")
            continue

        # Create a volume occupancy and fill the occupied voxels.
        time_start_fill_occupancy = time.time()
        volume_occupancy = torch.zeros((volume_radius * 2, volume_radius * 2, volume_radius * 2))
        volume_occupancy[occupied_voxel_indexes[:, 1], occupied_voxel_indexes[:, 0], occupied_voxel_indexes[:, 2]] = 1
        # print(f"Fill occupancy time: {time.time() - time_start_fill_occupancy}")

        # Scale and save.
        time_start_scaling = time.time()
        volume_occupancy = scale_volume_occupancy(volume_occupancy, scale=volume_scale)
        torch.save(volume_occupancy, os.path.join(scene_dir, "voxel_grid.pt"))
        # print(f"Scaling time: {time.time() - time_start_scaling}")

        if visualize:
            visualize_volume_occupancy(volume_occupancy)


def test():
    dataset_dir_path = "/Users/uveyisakbas/Desktop/blender/dataset"
    volume_occupancy_paths = sorted(
        [
            os.path.join(dataset_dir_path, dir_name, 'voxel_grid.pt') for dir_name in os.listdir(dataset_dir_path)
            if os.path.isdir(os.path.join(dataset_dir_path, dir_name)) and not dir_name.startswith('.')
        ]
    )
    volume_occupancy = torch.load(volume_occupancy_paths[8])
    visualize_volume_occupancy(volume_occupancy)


if __name__ == "__main__":
    process()
