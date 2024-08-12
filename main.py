import torch
import trimesh
import cv2 as cv
from trimesh.points import PointCloud
from scene import cameras, masks, images
from depth_prediction import predict_depth
from visualization import Colors
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polynomial
from camera import homogenize
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


def get_border(mask, border_type='middle'):
    expected_border_types = ['middle', 'max', 'min']
    if border_type not in expected_border_types:
        raise ValueError(f"Invalid value for border: {border_type}. Expected types: {expected_border_types}")

    mask = torch.as_tensor(mask, dtype=torch.float32)
    H, W = mask.shape
    indexes = torch.arange(0, H).repeat((W, 1)).T

    # Set H * 2 (which will be the highest value in the indexes_masked) for min to mark the background pixels.
    background_value = 0 if border_type in ['middle', 'max'] else H * 2 if border_type == 'min' else 0
    indexes_masked = indexes * mask + (1 - mask) * background_value

    border = torch.zeros(W)
    if border_type == 'middle':
        count_per_column = torch.as_tensor(mask).sum(dim=0) + 0.0001  # 0.0001 avoids division by 0
        sum_per_colum = indexes_masked.sum(dim=0)
        mean_index_per_column = sum_per_colum / count_per_column
        border = mean_index_per_column

    elif border_type == 'max':
        max_index_per_column = torch.max(indexes_masked, dim=0).values
        border = max_index_per_column

    elif border_type == 'min':
        min_index_per_column = torch.min(indexes_masked, dim=0).values
        border = min_index_per_column

    x = torch.arange(0, border.shape[0], dtype=torch.float32)
    border_points = torch.stack((x, border)).T

    # Eliminate borders where y=background_value. Assuming they are not real borders.
    return border_points[border_points[:, 1] != background_value, :]


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


def fit_polynomial(points):
    N, TWO = points.shape
    assert TWO == 2, 'The points shape must be (Nx2) where N is the number of points.'

    poly_coefficients = polynomial.polyfit(points[:, 0], points[:, 1], 3)
    return torch.tensor(poly_coefficients)


def get_intersections(lines, poly_coefficients):
    # TODO later remove the lines with no real intersection or no first quadrant intersection, instead of assertions.
    the_poly = Poly(poly_coefficients)  # The main polynomial for which intersections are sought.

    n_lines = lines.shape[0]
    zeros = torch.zeros(n_lines)
    new_coefficients = lines[:, 1:2] * poly_coefficients.repeat(n_lines, 1)
    new_coefficients = new_coefficients + torch.stack((lines[:, 2], lines[:, 0], zeros, zeros), dim=1)

    result = []
    for i, coefficient in enumerate(new_coefficients):
        new_poly = Poly(coefficient)
        new_poly_roots = torch.tensor([root.real for root in new_poly.roots() if root.imag == 0], dtype=torch.float32)
        assert len(new_poly_roots) > 0, f'No real root for idx:{i}'

        # Derive intersection points of a line and the polynomial.
        intersections = torch.stack((new_poly_roots, the_poly(new_poly_roots)), dim=1)

        # Use only the intersections on the first quadrant to make sure it is in the image.
        intersections = intersections[(intersections[:, 0] >= 0) & (intersections[:, 1] >= 0)]
        assert len(intersections) > 0, f'No intersection in first quadrant for idx:{i}'

        # Use the intersection with smallest y value.
        intersections = intersections[torch.argsort(intersections[:, 1])]
        result.append(intersections[0])

    return torch.stack(result)


def test():
    print('Testing')
    # Find middle and min coordinates using the masks
    cam_front, cam_top = cameras['front'], cameras['top']
    img_front, img_top = images['front'], images['top']
    mask_front, mask_top = masks['front'], masks['top']

    # TOP BORDER - MAX
    border_points_top = get_border(mask_top, border_type='max')
    draw_points(img_top, border_points_top)
    poly_coefficients = fit_polynomial(border_points_top)
    draw_polynomial(img_top, Poly(poly_coefficients))

    # FRONT BORDER - MIDDLE
    border_points_front = get_border(mask_front, border_type='middle')

    F = cam_front.fundamental_matrix_between(cam_top)

    lines_top = (F.T @ homogenize(border_points_front.T)).T
    intersections_top = get_intersections(lines_top, poly_coefficients)
    print(intersections_top.shape)
    idx = 500
    selected_point = border_points_front[idx]
    intersection_top = intersections_top[idx]
    line_top = lines_top[idx]

    cv.circle(img_front, (int(selected_point[0]), int(selected_point[1])), 5, Colors.GREEN.value, 5)
    cv.circle(img_top, (int(intersection_top[0]), int(intersection_top[1])), 5, Colors.GREEN.value, 5)
    draw_epipolar_line(img_top, line_top)
    cv.imshow('Front', img_front)
    cv.imshow('Top', img_top)
    cv.waitKey()


if __name__ == '__main__':
    test()
