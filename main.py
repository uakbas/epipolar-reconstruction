import torch
import trimesh
import cv2 as cv
from trimesh.points import PointCloud
from scene import cameras, masks, images
from depth_prediction import predict_depth
from visualization import Colors
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polynomial
from camera import homogenize_vec, homogenize
from epipolar_geometry import draw_epipolar_line


def pts3d_to_trimesh(img, pts3d, valid=None):
    import numpy as np
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate((
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
    ), axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((
        img[:-1, :-1].reshape(-1, 3),
        img[:-1, :-1].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3)
    ), axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)

    return vertices, faces, face_colors


def main():
    relative_depth = 1 / torch.as_tensor(predict_depth(images['front']), dtype=torch.float32)
    # relative_depth = torch.as_tensor(masks['front'], dtype=torch.float32) * 420
    # relative_depth = relative_depth + (1 - masks['front']) * 840
    relative_depth = relative_depth.flatten()

    depth = relative_depth

    W, H = cameras['front'].sensor_resolution
    x = torch.arange(0, W, dtype=torch.float32)
    y = torch.arange(0, H, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    ones = torch.ones_like(grid_x)
    img_points = torch.stack((grid_x, grid_y, ones))

    cam_points = depth * (torch.inverse(cameras['front'].K) @ img_points)
    cam_points = cam_points.T

    # Put into a unit sphere.
    cam_points = cam_points - torch.mean(cam_points, dim=0)
    cam_points = cam_points / torch.norm(cam_points, dim=1).max()

    point_cloud = PointCloud(cam_points)

    SHOW, EXPORT = False, False

    if SHOW:
        point_cloud.show()

    if EXPORT:
        point_cloud.export(file_obj='fish_point_cloud.obj', file_type='obj')

    cam_points = cam_points.view(H, W, 3)

    print(images['front'].shape, cam_points.shape)

    vertices, faces, face_colors = pts3d_to_trimesh(images['front'], pts3d=cam_points, valid=masks['front'])
    fish = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)

    # dimensions1 = [0.1, 0.1, 0.1]
    # box = trimesh.creation.box(extents=dimensions1)
    # scene = trimesh.Scene([fish, box])

    scene = trimesh.Scene([fish])
    scene.show()

    main()