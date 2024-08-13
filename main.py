import torch
import trimesh
from scene import cameras, masks, images
from visualization_tools import draw_points, draw_polynomial, pts3d_to_trimesh
from numpy.polynomial import Polynomial as Poly
from camera import homogenize
from depth_prediction import predict_depth
from geometry import get_horizontal_borders, get_poly_intersections_for_lines, fit_polynomial


def main():
    # Find middle and min coordinates using the masks
    cam_front, cam_top = cameras['front'], cameras['top']
    img_front, img_top = images['front'], images['top']
    mask_front, mask_top = masks['front'], masks['top']

    # TOP BORDER - MAX
    border_points_top, _ = get_horizontal_borders(mask_top, border_type='max')
    draw_points(img_top, border_points_top)
    poly_coefficients = fit_polynomial(border_points_top)
    draw_polynomial(img_top, Poly(poly_coefficients))

    # FRONT BORDER - MIDDLE
    border_points_front, border_idxs_front = get_horizontal_borders(mask_front, border_type='middle')

    F = cam_front.fundamental_matrix_between(cam_top)
    lines_top = (F.T @ homogenize(border_points_front.T)).T
    intersections_top = get_poly_intersections_for_lines(lines_top, poly_coefficients, img_top)

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

    # TODO this parameter could be learned.
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
