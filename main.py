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


def test_lines():
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
    scale = (radius * 0.9) / avg_fish_depth
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


def test_depth():
    image = images['front']
    depths = predict_depth(image[:, :, ::-1])  # BGR --> RGB

    show_depth(depths)


def get_absolute_depths(image, mask):
    depths = predict_depth(image[:, :, ::-1])  # BGR --> RGB
    depths = 1 / depths  # invert the depth map
    avg_fish_depth = np.ma.average(np.ma.array(depths, mask=~mask.astype(bool)))
    scale = radius / avg_fish_depth  # assuming the fish is at the center of the unit
    depths = depths * mask * scale
    depths = depths + (1 - mask) * radius * 2  # Set the surrounding of the fish as the end of unit: radius * 2
    return depths


def test_depth_to_line_segments():
    img_front = images['front']
    img_top = images['top']
    img_w, img_h = cameras['front'].sensor_resolution

    depths = get_absolute_depths(images['front'], masks['front'])

    point = torch.tensor([img_w / 2 + 375, img_h / 2 - 75])  # eye
    point = torch.tensor([img_w / 2 + 100, img_h / 2])  # head
    # point = torch.tensor([img_w / 2, img_h / 2])  # middle
    point = torch.tensor([img_w / 2 - 100, img_h / 2])  # tail
    cv.circle(img_front, (int(point[0]), int(point[1])), radius=10, color=Colors.RED.value, thickness=10)

    point_depth = depths[int(point[1]), int(point[0])]
    print('Point depth: ', point_depth)
    # point_depth = 420

    offset = 30
    point_depth_max = point_depth + offset
    point_top_max = cameras['front'].map_image_to_image(point, point_depth_max, cameras['top'])
    point_depth_min = point_depth - offset
    point_top_min = cameras['front'].map_image_to_image(point, point_depth_min, cameras['top'])
    cv.line(img_top, (int(point_top_min[0]), int(point_top_min[1])), (int(point_top_max[0]), int(point_top_max[1])),
            color=Colors.RED.value, thickness=4)

    cv.imshow('test a', img_top)
    cv.imshow('test b', img_front)
    cv.waitKey()


def test_image_to_image():
    img_front, img_top = images['front'], images['top']
    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cam_front.sensor_resolution

    p1 = torch.tensor([img_w / 2 + 1, img_h / 2])
    cv.circle(img_front, (int(p1[0]), int(p1[1])), radius=4, color=Colors.RED.value, thickness=4)

    depth = 450
    p2 = cam_front.map_image_to_image(p1, depth, cam_top)
    cv.circle(img_top, (int(p2[0]), int(p2[1])), radius=4, color=Colors.RED.value, thickness=4)

    print(f'P1:{p1} | P2:{p2} | img:{(img_w, img_h)} | depth:{depth}')

    '''
    cv.imshow('front', img_front)
    cv.imshow('top', img_top)
    cv.waitKey()
    '''


def test_depth_from_two_points():
    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cam_front.sensor_resolution

    X1 = torch.tensor([img_w / 2 + 1, img_h / 2])
    X2 = torch.tensor([641.0714, 335.8847])

    F = fundamental_matrix_between(cam_front, cam_top)

    print('Epipolar Constraint Check: ', homogenize_vec(X1).view(-1, 1).T @ F @ homogenize_vec(X2))
    print(cam_front.get_depth_from_two_points(cam_top, X1, X2))


# test_lines()
# test_depth()
# test_depth_to_line_segments()
test_image_to_image()
test_depth_from_two_points()
