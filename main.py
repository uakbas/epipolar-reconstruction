import cv2 as cv
from epipolar_geometry import draw_epipolar_line
from epipolar_geometry import skew_symmetric
import math
import torch


def test_epipolar_line_drawing(line):
    image_top = cv.imread('images/top.png')
    # line = [-1, 2, 0]  # -2x + y = 0 ==> y = 2x
    # line = [525, 105, -105]  # -2x + 100 = 0 ==> x = 50
    draw_epipolar_line(image_top, line)


def getRotMatX(deg):
    rad = deg / 180 * math.pi
    return torch.tensor([
        [1, 0, 0],
        [0, math.cos(rad), -math.sin(rad)],
        [0, math.sin(rad), math.cos(rad)]
    ], dtype=torch.float32)


radius = 420  # Distance to the center of the unit.
R_1_2 = getRotMatX(270)
T_1_2 = torch.tensor([0, -radius, radius], dtype=torch.float32)
T_1_2_X = skew_symmetric(T_1_2)
E = T_1_2_X @ R_1_2  # p_1^T @ E @ p_2 = 0 where E is the essential matrix.

print('R_1_2: ', R_1_2)
print('T_1_2: ', T_1_2)
print('essential_matrix: ', E)

cam_dist = 420
focal_length = 8
sensor_size = (11.33, 7.13)
sensor_resolution = (1280, 800)

# TODO DOUBLE CHECK THE WIDTH AND HEIGHT ORDER.
img_w, img_h = sensor_resolution
fy = focal_length * img_h / sensor_size[1]
fx = fy

# TODO DOUBLE CHECK CALIBRATION MATRIX.
K = torch.tensor([
    [fx, 0, img_w / 2],
    [0, fy, img_h / 2],
    [0, 0, 1]
], dtype=torch.float32)

K1, K2 = K, K
F = torch.t(torch.inverse(K1)) @ E @ torch.inverse(K2)

# Testing...
# E^T @ p1 = l2
p1 = torch.tensor([img_w / 2 - 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
# l2 = torch.transpose(E, 0, 1) @ p1
# print(l2)
# test_epipolar_line_drawing(l2)

# F^T @ p1 = l2
l2 = torch.t(F) @ p1
print(l2)
test_epipolar_line_drawing(l2)

p2 = torch.tensor([img_w / 2 - 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
l1 = F @ p2
image_front = cv.imread('images/front.png')
draw_epipolar_line(image_front, l1)

