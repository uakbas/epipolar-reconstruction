import cv2 as cv
import torch

from epipolar_geometry import draw_epipolar_line
from epipolar_geometry import fundamental_matrix
from camera import Camera, getRotMatX

radius = 420  # Distance to the center of the unit.

cameras = {
    'front': Camera(getRotMatX(0), torch.tensor([0, 0, -radius], dtype=torch.float32)),
    'top': Camera(getRotMatX(270), torch.tensor([0, -radius, 0], dtype=torch.float32)),
    'back': Camera(getRotMatX(180), torch.tensor([0, 0, radius], dtype=torch.float32)),
    'bottom': Camera(getRotMatX(90), torch.tensor([0, radius, 0], dtype=torch.float32)),
}


def fundamental_matrix_between(camera1: Camera, camera2: Camera):
    R, t = camera1.transformation_between(camera2)
    return fundamental_matrix(R, t, camera1.K, camera2.K)


def test_fundamental_matrix():
    image_top = cv.imread('images/top.png')
    image_front = cv.imread('images/front.png')
    image_back = cv.imread('images/back.png')
    image_bottom = cv.imread('images/bottom.png')

    img_w, img_h = cameras['front'].sensor.resolution

    F_front_top = fundamental_matrix_between(cameras['front'], cameras['top'])
    # FRONT TO TOP
    p_front = torch.tensor([img_w / 2 + 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
    line_top = F_front_top.T @ p_front
    draw_epipolar_line(image_top, line_top)

    # TOP TO FRONT
    p_top = torch.tensor([img_w / 2 - 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
    line_front = F_front_top @ p_top
    draw_epipolar_line(image_front, line_front)

    F_top_back = fundamental_matrix_between(cameras['top'], cameras['back'])
    # TOP TO BACK
    p_top = torch.tensor([img_w / 2 + 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
    line_back = F_top_back.T @ p_top
    draw_epipolar_line(image_back, line_back)

    # BOTTOM TO FRONT
    F_bottom_front = fundamental_matrix_between(cameras['bottom'], cameras['front'])
    p_bottom = torch.tensor([img_w / 2 - 300, img_h / 2, 1], dtype=torch.float32).unsqueeze(1)
    line_front = F_bottom_front.T @ p_bottom
    # draw_epipolar_line(image_front, line_front)


test_fundamental_matrix()
