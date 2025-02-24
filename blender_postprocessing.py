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

from camera import get_rot_mat_x, create_transformation_matrix, homogenize


def scale_volume_occupancy(vo, scale=1):
    """ Scale volume occupancy by a given scaling factor. """
    volume_dims = torch.as_tensor(vo.shape, dtype=torch.int)

    assert scale == 1 or scale % 2 == 0, 'Invalid scale value.'
    assert (volume_dims % scale == 0).all(), 'Shape values must be multiple of the scale.'

    shape = torch.stack([volume_dims // scale, torch.full_like(volume_dims, scale)], dim=0).T.flatten()
    vo_scaled = vo.view(*shape).sum(dim=(1, 3, 5)) > 0  # (scale * scale * scale) / 2
    return vo_scaled.to(torch.int32)


def visualize_volume_occupancy(vo, scale=1):
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


VOLUME_RADIUS = 384  # Distance from origin to each face of the volume.
VOLUME_SCALE = 24  # Scale factor for the volume occupancy grid.

# Transformation matrix to convert from Blender coordinates to volume occupancy coordinates.
# Origin for volume occupancy coordinate system is at the top-left-front of the volume.
# Y goes down, x goes right, z goes back.
# This transformation matrix helps to get voxels point coordinates in the volume occupancy coordinate system.
transformation_matrix = create_transformation_matrix(get_rot_mat_x(270), torch.tensor([-VOLUME_RADIUS, -VOLUME_RADIUS, VOLUME_RADIUS], dtype=torch.float32))

DATASET_PATH = os.path.join("/Users", "uveyisakbas", "Desktop", "blender_dataset", "test_scenes")
for directory in os.listdir(DATASET_PATH):

    if directory.startswith("."):  # Skip hidden files.
        continue

    print("Processing:", directory)

    scene_dir = os.path.join(DATASET_PATH, directory)

    # Voxelize and get the voxel points.
    object_voxels = pv.voxelize(pv.read(os.path.join(scene_dir, "scene.obj")), density=2, check_surface=False)
    object_points = torch.as_tensor(object_voxels.points, dtype=torch.float32)
    # print(torch.max(object_points))

    # Transform voxel points to volume occupancy indexes.
    occupied_voxel_indexes = torch.round((transformation_matrix @ homogenize(object_points.T)).T).to(dtype=torch.int)

    # Check if the voxel points are within the volume occupancy grid.
    if torch.all(torch.logical_and(0 <= occupied_voxel_indexes, occupied_voxel_indexes < VOLUME_RADIUS * 2)):
        print("valid")
    else:
        print(f"invalid: {directory}")
        continue

    # Create a volume occupancy and fill the occupied voxels.
    volume_occupancy = torch.zeros((VOLUME_RADIUS * 2, VOLUME_RADIUS * 2, VOLUME_RADIUS * 2))
    volume_occupancy[occupied_voxel_indexes[:, 1], occupied_voxel_indexes[:, 0], occupied_voxel_indexes[:, 2]] = 1

    # Scale and save.
    volume_occupancy = scale_volume_occupancy(volume_occupancy, scale=VOLUME_SCALE)
    torch.save(volume_occupancy, os.path.join(scene_dir, "voxel_grid.pt"))

    visualize_volume_occupancy(volume_occupancy)
