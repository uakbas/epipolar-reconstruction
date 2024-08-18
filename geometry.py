import torch
from numpy.polynomial import polynomial
from numpy.polynomial import Polynomial as Poly


def get_horizontal_borders(mask, border_type='middle'):
    """
    Finds horizontal borders of a mask using the strategy defined by 'border_type' parameter.
    Eliminates pseudo borders which belongs to background.

    :param mask: The mask which is used to find the borders.
    :param border_type: The definition of which border to find.
    :return: border_points [(NX2) where N is the number of border points.],
             idxs_real_border [indexes of border points which are on the mask]
    """
    expected_border_types = ['middle', 'max', 'min']
    if border_type not in expected_border_types:
        raise ValueError(f"Invalid value for border: {border_type}. Expected types: {expected_border_types}")

    mask = torch.as_tensor(mask, dtype=torch.float32)
    H, W = mask.shape
    indexes = torch.arange(0, H).repeat((W, 1)).T

    # Set H * 2 (which will be the highest value in the indexes_masked) for min to mark the background pixels.
    background_value = 0 if border_type in ['middle', 'max'] else H * 2 if border_type == 'min' else 0
    indexes_masked = indexes * mask + (1 - mask) * background_value

    border_y = torch.zeros(W)
    if border_type == 'middle':
        count_per_column = torch.as_tensor(mask).sum(dim=0) + 0.0001  # 0.0001 avoids division by 0
        sum_per_colum = indexes_masked.sum(dim=0)
        mean_index_per_column = sum_per_colum / count_per_column
        border_y = mean_index_per_column

    elif border_type == 'max':
        max_index_per_column = torch.max(indexes_masked, dim=0).values
        border_y = max_index_per_column

    elif border_type == 'min':
        min_index_per_column = torch.min(indexes_masked, dim=0).values
        border_y = min_index_per_column

    border_x = torch.arange(0, border_y.shape[0], dtype=torch.float32)
    border_points = torch.stack((border_x, border_y)).T

    # Eliminate borders where y=background_value. Assuming they are pseudo borders.
    mask_real_border = border_points[:, 1] != background_value
    idxs_real_border = torch.nonzero(mask_real_border).squeeze()
    border_points = border_points[mask_real_border, :]

    return border_points, idxs_real_border


def fit_polynomial(points):
    """ This is a wrapper around polynomial.polyfit method. """
    N, TWO = points.shape
    assert TWO == 2, 'The points shape must be (Nx2) where N is the number of points.'

    poly_coefficients = polynomial.polyfit(points[:, 0], points[:, 1], 3)
    return torch.tensor(poly_coefficients)


def get_poly_intersections_for_lines(lines, poly_coefficients, image):
    """
    Finds intersections between given epipolar lines and a polynomial. Polynomial is defined with coefficients.

    :param lines: The lines which is assumed to intersect the given polynomial.
    :param poly_coefficients: The polynomial where the intersection points are searched on.
    :param image: Used to restrict interval of intersection points since there may be multiple intersection points.
    :return: The intersection points for each line. (Nx2) where N is the number of intersection points.
    """

    # The main polynomial for which intersections are sought.
    the_poly = Poly(poly_coefficients)

    # let polynomial y = a + b.x + c.x**2 + d.x**3
    # let an epipolar line 0 = kx + ly + m
    # Putting together, 0 = (l.a + m) + (l.b + k)x + l.c.x**2 + l.d.x**3
    # Find the roots of this equation gives intersection between the line and polynomial
    # new_coefficients define a polynomial with coefficients: [(l.a + m), (l.b + k), l.c, l.d]
    # Finding the roots of polynomial with new_coefficients means finding intersection points.
    n_lines = lines.shape[0]
    zeros = torch.zeros(n_lines)
    new_coefficients = lines[:, 1:2] * poly_coefficients.repeat(n_lines, 1)
    new_coefficients = new_coefficients + torch.stack((lines[:, 2], lines[:, 0], zeros, zeros), dim=1)

    # Use for interval constraints of intersection points.
    padding_y, padding_x, _ = torch.as_tensor(image.shape, dtype=torch.float32) / 2

    # Finding and eliminating roots. Since a line and a polynomial may have multiple intersections or
    # have intersections with imaginary parts, it is necessary to eliminate intersections and choose one of them.
    result = []
    for i, coefficient in enumerate(new_coefficients):
        new_poly = Poly(coefficient)

        # Only the real roots
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


def align_points_by_polynomial(points: torch.Tensor, mask):
    """
    Fit a polynomial on the given points. Sample the polynomial points and filter in order to get the polynomial points
    which are inside the mask.
    :param points: Points to be fit by a polynomial.
    :param mask: Binary mask which filters the polynomial points.
    :return: (Polynomial points, X coordinates of polynomial points)
    """
    poly = Poly(fit_polynomial(points))

    # Create polynomial points by sampling the x coordinate with step=1
    mask = torch.as_tensor(mask)
    H, W = mask.shape
    x_coords = torch.arange(0, W, dtype=torch.int)
    y_coords = torch.as_tensor(poly(x_coords), dtype=torch.int)
    poly_points = torch.stack((x_coords, y_coords), dim=0).T

    # Use only the polynomial points which are inside the borders of the mask
    poly_points = poly_points[(poly_points[:, 1] >= 0) & (poly_points[:, 1] <= H - 1)]

    # Binary polynomial mask where only the pixels that the polynomial passes through are set to 1.
    poly_mask = torch.zeros_like(mask, dtype=torch.int)
    poly_mask[poly_points[:, 1], poly_points[:, 0]] = 1

    # Filter the polynomial points where the mask is 0
    poly_mask = poly_mask * mask

    y_coords_valid, x_coords_valid = torch.nonzero(poly_mask, as_tuple=True)
    poly_points_valid = torch.stack((x_coords_valid, y_coords_valid), dim=0).T

    # Sort by x values, asc
    poly_points_valid = poly_points_valid[torch.argsort(poly_points_valid[:, 0])]

    return poly_points_valid.to(torch.float32), poly_points_valid[:, 0]
