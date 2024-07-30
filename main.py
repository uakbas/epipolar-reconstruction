import cv2 as cv
import torch

from epipolar_geometry import draw_epipolar_line
from epipolar_geometry import fundamental_matrix

from camera import K, img_w, img_h, R_1_2, T_1_2

K1, K2 = K, K
F = fundamental_matrix(R_1_2, T_1_2, K1, K2)

# Testing: F^T @ p1 = l2
# FRONT TO TOP
image_top = cv.imread('images/top.png')
p1 = torch.tensor([img_w / 2 + 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
l2 = torch.t(F) @ p1
draw_epipolar_line(image_top, l2)

# TOP TO FRONT
image_front = cv.imread('images/front.png')
p2 = torch.tensor([img_w / 2 - 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
l1 = F @ p2
draw_epipolar_line(image_front, l1)
