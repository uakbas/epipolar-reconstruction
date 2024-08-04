import cv2 as cv
import numpy as np
import torch
from enum import Enum
from epipolar_geometry import draw_epipolar_line
from epipolar_geometry import fundamental_matrix
from camera import Camera, homogenize_vec, create_projection_matrix
from scene import cameras, images, masks, radius
from depth_prediction import predict_depth, show_depth


def fundamental_matrix_between(camera1: Camera, camera2: Camera):
    R, t = camera1.transformation_between(camera2)
    return fundamental_matrix(R, t, camera1.K, camera2.K)


class Colors(Enum):
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)


def draw_epipolar_geometry_between(position_1, position_2):
    img1, img2 = images[position_1], images[position_2]
    cam1 = cameras[position_1]
    cam2 = cameras[position_2]

    F = fundamental_matrix_between(cam1, cam2)

    def draw_epipolar_geometry(point, color):
        if not torch.is_tensor(point):
            point = torch.tensor(point, dtype=torch.float32)

        line = F.T @ homogenize_vec(point)
        draw_epipolar_line(img2, line, color)
        cv.circle(img1, (int(point[0]), int(point[1])), radius=5, color=color, thickness=4)
        return img1, img2

    return draw_epipolar_geometry


def test_circle():
    img_w, img_h = cameras['front'].sensor.resolution

    draw = draw_epipolar_geometry_between('front', 'top')

    # Front Body
    draw([img_w / 2 + 100, img_h / 2 - 200], Colors.RED.value)
    draw([img_w / 2 + 100, img_h / 2 - 100], Colors.GREEN.value)
    draw([img_w / 2 + 100, img_h / 2 + 30], Colors.BLUE.value)

    # Fish eye
    draw([img_w / 2 + 375, img_h / 2 - 75], Colors.RED.value)

    # Fin top
    draw([img_w / 2 - 35, img_h / 2 - 280], Colors.GREEN.value)

    # Tail
    draw([150, img_h / 2 - 180], Colors.RED.value)
    draw([150, img_h / 2 - 110], Colors.GREEN.value)
    img1, img2 = draw([150, img_h / 2 - 60], Colors.BLUE.value)

    cv.imshow('Front', img1)
    cv.imshow('Top', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def map_to_absolute_depth(depth, mask):
    print(f'Max Depth: {np.max(depth)} | Min Depth: {np.min(depth)}')
    avg_fish_depth = np.ma.average(np.ma.array(depth, mask=~mask.astype(bool)))
    scale = radius / avg_fish_depth
    depth_masked = depth * mask
    depth_scaled = depth_masked * scale
    return depth_scaled


def test():
    img_front = images['front']
    img_top = images['top']
    cam_front = cameras['front']
    cam_top = cameras['top']
    img_w, img_h = cameras['front'].sensor.resolution

    point = torch.tensor([img_w / 2, img_h / 2])
    depth = 240
    point_top = cam_front.map_image_to_image(point, depth, cam_top)

    cv.circle(img_top, (int(point_top[0]), int(point_top[1])), radius=5, color=Colors.GREEN.value, thickness=4)

    cv.imshow('test', img_top)
    cv.waitKey()


test()
