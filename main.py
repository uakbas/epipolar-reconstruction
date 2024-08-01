import cv2 as cv
import torch
from enum import Enum
from epipolar_geometry import draw_epipolar_line
from epipolar_geometry import fundamental_matrix
from camera import Camera
from scene import cameras, images
import numpy as np
import matplotlib
from depth.depth_anything_v2.dpt import DepthAnythingV2


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
        line = F.T @ torch.tensor([point[0], point[1], 1], dtype=torch.float32).unsqueeze(1)
        draw_epipolar_line(img2, line, color)
        cv.circle(img1, (int(point[0]), int(point[1])), radius=5, color=color, thickness=4)
        return img1, img2

    return draw_epipolar_geometry


def test_circle():
    img_w, img_h = cameras['front'].sensor.resolution

    draw = draw_epipolar_geometry_between('front', 'top')

    # Front Body
    draw((img_w / 2 + 100, img_h / 2 - 200), Colors.RED.value)
    draw((img_w / 2 + 100, img_h / 2 - 100), Colors.GREEN.value)
    draw((img_w / 2 + 100, img_h / 2 + 30), Colors.BLUE.value)

    # Fish eye
    draw((img_w / 2 + 375, img_h / 2 - 75), Colors.RED.value)

    # Fin top
    draw((img_w / 2 - 35, img_h / 2 - 280), Colors.GREEN.value)

    # Tail
    draw((150, img_h / 2 - 180), Colors.RED.value)
    draw((150, img_h / 2 - 110), Colors.GREEN.value)
    img1, img2 = draw((150, img_h / 2 - 60), Colors.BLUE.value)

    cv.imshow('Front', img1)
    cv.imshow('Top', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


# test_circle()

def test_depth():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
    encoder = 'vitl'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    depth = model.infer_image(images['front'])
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth = depth.astype(np.uint8)
    print(np.min(depth), np.max(depth))

    depth = model.infer_image(images['top'])
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth = depth.astype(np.uint8)
    print(np.min(depth), np.max(depth))


test_depth()
