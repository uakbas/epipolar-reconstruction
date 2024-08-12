from enum import Enum
import cv2 as cv
import torch


class Colors(Enum):
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)


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


def draw_points(img, points, **kwargs):
    color = kwargs.get('color', Colors.RED.value)
    radius = kwargs.get('radius', 5)
    thickness = kwargs.get('thickness', 5)

    for point in points:
        cv.circle(img, (int(point[0]), int(point[1])), radius, color, thickness)


def draw_polynomial(img, poly):
    y_lim, x_lim, channel = img.shape
    x_coords = torch.arange(0, x_lim)
    y_coords = poly(x_coords)
    draw_points(img, torch.stack((x_coords, y_coords)).T, radius=3, color=Colors.BLUE.value, thickness=3)
