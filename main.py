import torch
import trimesh
from scene import cameras, masks, images
from visualization_tools import Colors, draw_points, draw_polynomial, pts3d_to_trimesh
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polynomial
from camera import homogenize
from depth_prediction import predict_depth


def fit_polynomial(points):
    N, TWO = points.shape
    assert TWO == 2, 'The points shape must be (Nx2) where N is the number of points.'

    poly_coefficients = polynomial.polyfit(points[:, 0], points[:, 1], 3)
    return torch.tensor(poly_coefficients)


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
    mask_real_border = border_points[:, 1] != background_value
    idxs_real_border = torch.nonzero(mask_real_border).squeeze()
    border_points = border_points[mask_real_border, :]

    return border_points, idxs_real_border


def get_intersections(lines, poly_coefficients, image):
    padding_y, padding_x, _ = torch.as_tensor(image.shape, dtype=torch.float32) / 4
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

        # Use only the intersections on the first quadrant (with paddings) to make sure it is in the image.
        intersections = intersections[(intersections[:, 0] >= -padding_x) & (intersections[:, 1] >= -padding_y)]
        assert len(intersections) > 0, f'No intersection in first quadrant for idx:{i}'

        # Use the intersection with smallest y value.
        intersections = intersections[torch.argsort(intersections[:, 1])]
        result.append(intersections[0])

    return torch.stack(result)


def main():
    # Find middle and min coordinates using the masks
    cam_front, cam_top = cameras['front'], cameras['top']
    img_front, img_top = images['front'], images['top']
    mask_front, mask_top = masks['front'], masks['top']

    # TOP BORDER - MAX
    border_points_top, _ = get_border(mask_top, border_type='max')
    draw_points(img_top, border_points_top)
    poly_coefficients = fit_polynomial(border_points_top)
    draw_polynomial(img_top, Poly(poly_coefficients))

    # FRONT BORDER - MIDDLE
    border_points_front, border_idxs_front = get_border(mask_front, border_type='middle')

    F = cam_front.fundamental_matrix_between(cam_top)
    lines_top = (F.T @ homogenize(border_points_front.T)).T
    intersections_top = get_intersections(lines_top, poly_coefficients, img_top)

    points_ref, points_tar = border_points_front, intersections_top
    n_intersection = points_ref.shape[0]
    depths_abs = torch.zeros(n_intersection)
    print('Finding depths for n_intersection:', n_intersection)
    for i in range(n_intersection):
        depth = cam_front.get_depth_from_two_points(cam_top, points_ref[i], points_tar[i])
        depths_abs[i] = depth

    assert depths_abs[depths_abs > 0].shape[0] == n_intersection, 'Absolute depths are complete'

    print(points_ref.shape, points_tar.shape)
    assert points_ref.shape == points_tar.shape, 'Reference and target points do not match'

    depth_map_rel = 1 / torch.as_tensor(predict_depth(img_front), dtype=torch.float32)
    depths_rel = depth_map_rel[points_ref.to(torch.int32)[:, 1], points_ref.to(torch.int32)[:, 0]]

    scaling_ratio = 0.7

    scale_coefficients = (depths_abs / depths_rel) * scaling_ratio
    translation_coefficients = depths_abs * (1 - scaling_ratio)

    mask_w = mask_front.shape[1]
    scale_coefficients_complete = torch.zeros(mask_w)
    scale_coefficients_complete[border_idxs_front] = scale_coefficients

    translation_coefficients_complete = torch.zeros(mask_w)
    translation_coefficients_complete[border_idxs_front] = translation_coefficients

    depth_map_abs = depth_map_rel * scale_coefficients_complete + translation_coefficients_complete
    depth_map_abs = depth_map_abs.flatten()

    W, H = cameras['front'].sensor_resolution
    x = torch.arange(0, W, dtype=torch.float32)
    y = torch.arange(0, H, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    ones = torch.ones_like(grid_x)
    img_points = torch.stack((grid_x, grid_y, ones))

    cam_points = depth_map_abs * (torch.inverse(cameras['front'].K) @ img_points)
    cam_points = cam_points.T

    # Put into a unit sphere.
    cam_points = cam_points - torch.mean(cam_points, dim=0)
    cam_points = cam_points / torch.norm(cam_points, dim=1).max()
    cam_points = cam_points.view(H, W, 3)

    vertices, faces, face_colors = pts3d_to_trimesh(images['front'], pts3d=cam_points, valid=masks['front'])
    fish = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)
    scene = trimesh.Scene([fish])
    scene.export(file_obj='fish_mesh', file_type='obj')
    scene.show()

    # print(depths_rel[0])
    # print(scale_coefficients[0], translation_coefficients[0])
    # print(depths_rel[0] * scale_coefficients[0] + translation_coefficients[0])
    # print(depths_abs[0])
    print('stop')

    """
    for i in [0, 300, 600, 900, 1000]:
        depth = cam_front.get_depth_from_two_points(cam_top, points_ref[i], points_tar[i])
        cv.circle(img_front, (int(points_ref[i][0]), int(points_ref[i][1])), 5, Colors.GREEN.value, 5)
        cv.circle(img_top, (int(points_tar[i][0]), int(points_tar[i][1])), 5, Colors.GREEN.value, 5)
        draw_epipolar_line(img_top, lines_top[i])
        print(depth)

    cv.imshow('Front', img_front)
    cv.imshow('Top', img_top)
    cv.waitKey()
    """


if __name__ == '__main__':
    main()
