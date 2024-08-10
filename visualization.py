import torch
import cv2 as cv
import numpy as np
from enum import Enum
from scene import cameras, images, masks, radius
from camera import Camera, homogenize_vec
from epipolar_geometry import draw_epipolar_line
from depth_prediction import predict_depth


class Colors(Enum):
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)


def draw_epipolar_geometry_between(cam_position_1, cam_position_2):
    cam1: Camera = cameras[cam_position_1]
    cam2: Camera = cameras[cam_position_2]
    img1, img2 = images[cam_position_1], images[cam_position_2]

    F = cam1.fundamental_matrix_between(cam2)

    def draw_epipolar_geometry(point, color):
        point = torch.as_tensor(point, dtype=torch.float32)
        line = F.T @ homogenize_vec(point)
        cv.circle(img1, (int(point[0]), int(point[1])), radius=5, color=color, thickness=5)
        draw_epipolar_line(img2, line, color)
        return img1, img2

    return draw_epipolar_geometry


def test_epipolar_lines():
    """ Test finding epipolar lines given the points on the reference image. """
    cam = cameras['front']
    img_w, img_h = cam.sensor_resolution

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


def test_image_point_to_image_point():
    """ Test finding the target image point given the reference image point and its depth value. """

    img_front, img_top = images['front'], images['top']
    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cam_front.sensor_resolution

    p1 = torch.tensor([img_w / 2 + 1, img_h / 2])
    cv.circle(img_front, (int(p1[0]), int(p1[1])), radius=5, color=Colors.RED.value, thickness=5)

    depth = 450
    p2 = cam_front.map_image_to_image(p1, depth, cam_top)
    cv.circle(img_top, (int(p2[0]), int(p2[1])), radius=5, color=Colors.RED.value, thickness=5)

    print(f'P1:{p1} | P2:{p2} | img:{(img_w, img_h)} | depth:{depth}')

    cv.imshow('top', img_top)
    cv.imshow('front', img_front)
    cv.waitKey()
    cv.destroyAllWindows()


def test_depth_from_two_points():
    """ Test finding depth from two points which correspond to same point in 3D. """

    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cam_front.sensor_resolution

    X1 = torch.tensor([img_w / 2 + 1, img_h / 2])
    X2 = torch.tensor([641.0714, 335.8847])

    F = cam_front.fundamental_matrix_between(cam_top)

    # The value should be close to zero.
    print('Epipolar Constraint Check: ', homogenize_vec(X1).view(-1, 1).T @ F @ homogenize_vec(X2))
    print('Depth between 2 points is: ', cam_front.get_depth_from_two_points(cam_top, X1, X2))


def test_depth_to_line_segments():
    """ Test finding an epipolar line segment given a point and an estimated depth value. """

    img_front, img_top = images['front'], images['top']
    img_w, img_h = cameras['front'].sensor_resolution
    mask_front = masks['front']

    # Calculate absolute depth value roughly, assuming the fish at the center.
    depths = 1 / predict_depth(img_front[:, :, ::-1])  # BGR --> RGB
    avg_fish_depth = np.ma.average(np.ma.array(depths, mask=~mask_front.astype(bool)))
    depths = depths * mask_front * (radius / avg_fish_depth)  # assuming the fish is at the center of the unit
    depths = depths + (1 - mask_front) * radius * 2  # Set the surrounding of the fish as the end of unit: radius * 2

    selected_points = {
        'eye': torch.tensor([img_w / 2 + 375, img_h / 2 - 75]),
        'head': torch.tensor([img_w / 2 + 100, img_h / 2]),
        'middle': torch.tensor([img_w / 2, img_h / 2]),
        'tail': torch.tensor([img_w / 2 - 100, img_h / 2])
    }
    point = selected_points['tail']

    point_ = tuple(map(int, point))
    point_depth, offset = depths[point_[1], point_[0]], 30
    point_depth_max, point_depth_min = point_depth + offset, point_depth - offset

    point_top_max = cameras['front'].map_image_to_image(point, point_depth_max, cameras['top'])
    point_top_min = cameras['front'].map_image_to_image(point, point_depth_min, cameras['top'])
    point_top_max_, point_top_min_ = tuple(map(int, point_top_max)), tuple(map(int, point_top_min))

    cv.circle(img_front, point_, radius=10, color=Colors.RED.value, thickness=10)
    cv.line(img_top, point_top_max_, point_top_min_, color=Colors.RED.value, thickness=5)

    print(f'Point Depth: {point_depth} | Max: {point_depth_max} | Min: {point_depth_min}')
    cv.imshow('Top', img_top)
    cv.imshow('Front', img_front)
    cv.waitKey()


def test_transforming_relative_depth():
    """ Test linear transformation of relative depth by using absolute depth values of selected points. """

    img_front, img_top = images['front'], images['top']
    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cameras['front'].sensor_resolution

    def helper_func(p_, d_, color=Colors.RED.value, log=False):
        cv.circle(img_front, (int(p_[0]), int(p_[1])), radius=5, color=color, thickness=5)
        p_t = cam_front.map_image_to_image(p_, d_, cam_top)
        cv.circle(img_top, (int(p_t[0]), int(p_t[1])), radius=5, color=color, thickness=5)
        if log:
            print(f'front: {p_} | top: {p_t} | D: {d_}')
        return p_, p_t, d_

    # Draw selected reference points.
    helper_func(torch.tensor([img_w / 2 + 375, img_h / 2 - 75]), 445)  # 1: EYE
    helper_func(torch.tensor([img_w / 2 + 150, img_h / 2 - 75]), 435, color=Colors.GREEN.value)  # 2: RIGHT BODY
    helper_func(torch.tensor([img_w / 2 + 50, img_h / 2 - 75]), 430, color=Colors.BLUE.value)  # 3: MIDDLE BODY
    helper_func(torch.tensor([img_w / 2 - 150, img_h / 2 - 75]), 410)  # 4: LEFT BODY
    helper_func(torch.tensor([img_w / 2 - 300, img_h / 2 - 75]), 390, color=Colors.GREEN.value)  # 5: TAIL 1
    helper_func(torch.tensor([img_w / 2 - 450, img_h / 2 - 75]), 365, color=Colors.BLUE.value)  # 6: TAIL 2
    helper_func(torch.tensor([img_w / 2 - 40, img_h / 2 - 270]), 450)  # 7: TOP MIDDLE FIN
    helper_func(torch.tensor([200, img_h / 2 - 200]), 375, color=Colors.GREEN.value)  # 8: BACK MIDDLE FIN

    # Selected points
    points = torch.tensor([
        [1015., 325.],
        [790., 325.],
        [690., 325.],
        [490., 325.],
        [340., 325.],
        [190., 325.],
        [600., 130.],
        [200., 200.],
    ], dtype=torch.float32)
    points = points[:, [1, 0]].int()  # (x,y) -> (y,x)

    # Relative Depths
    relative_depth_map = torch.tensor(predict_depth(img_front[:, :, ::-1]), dtype=torch.float32)  # BGR --> RGB
    relative_depth_map = 1 / relative_depth_map
    relative_depths = relative_depth_map[points[:, 0], points[:, 1]]

    A = torch.stack((relative_depths, torch.ones(relative_depths.shape))).T
    print('Relative Depths: ', relative_depths)

    # Absolute depths. Manually derived.
    b = torch.tensor([445, 435, 430, 410, 390, 365, 450, 375], dtype=torch.float32).view(-1, 1)
    print('Absolute Depths: ', b.squeeze())

    # Solve the minimum least square problem.
    coefficients = torch.inverse((A.T @ A)) @ A.T @ b
    print('Coefficients: ', coefficients.squeeze())

    # Estimated absolute depths
    p = A @ coefficients
    print('Estimated Absolute Depths: ', p.squeeze())
    print('Difference Between Depths: ', (b - p).squeeze())
    print('SSE: ', torch.sum(torch.square(b - p)))

    cv.imshow('Top', img_top)
    cv.imshow('Front', img_front)
    cv.waitKey()

    '''
        Relative Depths:            tensor([0.1440, 0.1376, 0.1372, 0.1399, 0.1415, 0.1473, 0.1487, 0.1459])
        Absolute Depths:            tensor([445., 435., 430., 410., 390., 365., 450., 375.])
        Coefficients:               tensor([-1847.0676,   676.2079])
        Estimated Absolute Depths:  tensor([410.2569, 422.0400, 422.8638, 417.7894, 414.8478, 404.2094, 401.6009, 406.7561])
        Difference Between Depths:  tensor([ 34.7431,  12.9600,   7.1362,  -7.7894, -24.8478, -39.2094,  48.3991, -31.7561])
        SSE:                        tensor(6992.3652)
    '''

# test_epipolar_lines()
# test_image_point_to_image_point()
# test_depth_from_two_points()
# test_depth_to_line_segments()
# test_transforming_relative_depth()
