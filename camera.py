import torch
import math


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
