import cv2 as cv
import numpy as np


def generate_points_from(epipolar_line: list, y_limit, x_limit):
    assert len(epipolar_line) == 3, 'Line must have 3 elements.'

    a, b, c = epipolar_line

    if b == 0:
        # a.x + c = 0 ==> Line is vertical.
        assert a != 0, 'Both coefficients cannot be zero at the same time.'

        return (-c / a, 0), (-c / a, y_limit)

    else:
        # ax + by + c = 0 ==> y = (-c - ax) / b
        def line_func(x1):
            return (-c - a * x1) / b

        return (0, line_func(0)), (x_limit, line_func(x_limit))


def draw_epipolar_line(image: np.ndarray, epipolar_line, color=(0, 255, 0), thickness=2):
    H, W, _ = image.shape
    start_point, end_point = generate_points_from(epipolar_line, H, W)
    start_point, end_point = [(int(point[0]), int(point[1])) for point in [start_point, end_point]]
    print(f'Drawing line between {start_point} and {end_point} on an image with size (H:{H} W:{W})')

    cv.line(image, start_point, end_point, color, thickness)
    cv.imshow('Image with Line', image)
    cv.waitKey(0)
