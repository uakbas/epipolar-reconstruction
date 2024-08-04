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


def map_img_coord_into_3D(p_img, K, depth):
    # USE p_img as 2D for now.
    p_img_hom = homogenize_vec(p_img)
    p_3d = (torch.inverse(K) @ p_img_hom) * depth
    return p_3d


def test():
    img_front = images['front']
    img_top = images['top']
    cam_front = cameras['front']
    cam_top = cameras['top']
    img_w, img_h = cameras['front'].sensor.resolution

    # From a point in the front image plane to a point in the cam coordinate system of the front camera.
    point = torch.tensor([img_w / 2, img_h / 2])
    point_3D = cam_front.map_image_plane_to_3d(point, 420)

    # Project point in the front camera coordinates in to the top image plane.
    project_mat = cam_front.projection_between(cam_top)
    point_3D_hom = homogenize_vec(point_3D)
    point_top = project_mat @ point_3D_hom
    point_top = point_top / point_top[2]  # Normalize dividing by z.

    cv.circle(img_top, (int(point_top[0]), int(point_top[1])), radius=5, color=Colors.GREEN.value, thickness=4)

    cv.imshow('test', img_top)
    cv.waitKey()


test()
