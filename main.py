import torch
import cv2 as cv
from scene import cameras, masks, images
from visualization_tools import Colors, draw_points, draw_polynomial
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polynomial
from camera import homogenize
from epipolar_geometry import draw_epipolar_line


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


def fit_polynomial(points):
    N, TWO = points.shape
    assert TWO == 2, 'The points shape must be (Nx2) where N is the number of points.'

    poly_coefficients = polynomial.polyfit(points[:, 0], points[:, 1], 3)
    return torch.tensor(poly_coefficients)


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
    border_points_top = get_border(mask_top, border_type='max')
    draw_points(img_top, border_points_top)
    poly_coefficients = fit_polynomial(border_points_top)
    draw_polynomial(img_top, Poly(poly_coefficients))

    # FRONT BORDER - MIDDLE
    border_points_front = get_border(mask_front, border_type='middle')

    F = cam_front.fundamental_matrix_between(cam_top)

    lines_top = (F.T @ homogenize(border_points_front.T)).T
    intersections_top = get_intersections(lines_top, poly_coefficients, img_top)
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
    main()
