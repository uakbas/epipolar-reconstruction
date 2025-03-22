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

from pyvista.demos.logo import atomize
from camera import get_rot_mat_x, create_transformation_matrix, homogenize


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
    scene.show()


def visualize_voxels_by_pyvista(voxels, use_atomize=False):
    if use_atomize:
        object_voxels_atomized = atomize(voxels, scale=0.5)
        object_voxels_atomized.plot()
        return

    plotter = pv.Plotter()
    plotter.add_mesh(voxels, color="red", point_size=1, render_points_as_spheres=True)
    plotter.show()


def get_object_points(object_path):
    """ Load object, voxelize and return the voxel points.
    TODO figure out the relation between density and volume_scale, adjust dynamically.

    :param object_path: Path of the object to be loaded.
    :return: Voxel points.
    """
    mesh = pv.read(object_path)
    voxels = pv.voxelize(mesh, density=2, check_surface=False)
    points = torch.as_tensor(voxels.points, dtype=torch.float32)
    return points


def process():
    volume_radius = 384  # Distance from origin to each face of the volume.
    volume_scale = 12  # Scale factor for the volume occupancy grid.
    root = "/Users/uveyisakbas/Desktop/blender"
    dataset_dir_path = os.path.join(root, "dataset")
    visualize = True

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

        scene_dir = os.path.join(dataset_dir_path, directory)

        object_path = os.path.join(scene_dir, "scene.obj")
        object_points = get_object_points(object_path)

        occupied_voxel_indexes = torch.round((trans_mat @ homogenize(object_points.T)).T).to(dtype=torch.int)

        # Check if the voxel points are within the volume occupancy grid.
        if not torch.all(torch.logical_and(0 <= occupied_voxel_indexes, occupied_voxel_indexes < volume_radius * 2)):
            print(f"Invalid scene: {directory}\n")
            continue

        # Create a volume occupancy and fill the occupied voxels.
        volume_occupancy = torch.zeros((volume_radius * 2, volume_radius * 2, volume_radius * 2))
        volume_occupancy[occupied_voxel_indexes[:, 1], occupied_voxel_indexes[:, 0], occupied_voxel_indexes[:, 2]] = 1

        # Scale and save.
        volume_occupancy = scale_volume_occupancy(volume_occupancy, scale=volume_scale)
        torch.save(volume_occupancy, os.path.join(scene_dir, "voxel_grid.pt"))

        if visualize:
            visualize_volume_occupancy(volume_occupancy)


if __name__ == "__main__":
    process()
