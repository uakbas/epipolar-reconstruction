import torch
import trimesh
import cv2 as cv
import numpy as np
import json
import os

from camera import Camera, homogenize_vec, Sensor, get_rot_mat_x
from epipolar_geometry import draw_epipolar_line
from visualization_tools import Colors, points_3d_to_trimesh
from trimesh.points import PointCloud
from depth_prediction import predict_depth
from scene import Scene


def load_blender_dataset(model_dir, cam_position, normalize=False):

    meta_path = os.path.join(model_dir, 'meta.json')
    image_dir = os.path.join(model_dir, 'image')
    depth_dir = os.path.join(model_dir, 'depth')

    with open(meta_path) as meta:
        meta = json.load(meta)

    # The radius of the sphere in which the object is located.
    volume_radius = meta['volume_radius']
    scale_factor = 1.0 / volume_radius if normalize else 1.0

    # load camera
    cam_meta = meta['cameras'][cam_position]
    R = torch.tensor(cam_meta['rotation'], dtype=torch.float32)
    t = torch.tensor(cam_meta['translation'], dtype=torch.float32)
    t = t * scale_factor  # apply normalization

    sensor = cam_meta['sensor']
    sensor = Sensor(focal_length=sensor['focal'], size=tuple(sensor['size']), resolution=tuple(sensor['resolution']))

    # Blender camera: X: right, Y: up, -Z: forward | OpenCV camera: X: right, Y: down, Z: forward
    # Convertion is needed from Blender to OpenCV.
    # R, t from blender
    # World Coordinate --> R1, t --> Camera Coordinate (Blender) --> R2 --> Camera Coordinate (OpenCV)
    # R^-1 @ (Xw - t) = Xc | R1^-1 @ R2^-1 @ (Xw -t) = Xc
    # R^-1 = R1^-1 @ R2^-1 ==> R = R2 @ R1 thus, R_conv @ R is new rotation matrix.
    R_conv = get_rot_mat_x(180)
    cam = Camera(R_conv @ R, t, sensor)

    # load image
    image_path = os.path.join(image_dir, f'{cam_position}.png')
    image = cv.imread(image_path)
    if image is None:
        raise Exception(f'Could not read image {image_path}')

    # load depth map
    depth_min = meta['cameras'][cam_position]['depth']['min']
    depth_max = meta['cameras'][cam_position]['depth']['max']
    depth_path = os.path.join(depth_dir, f'{cam_position}.png')
    depth = cv.imread(depth_path, cv.IMREAD_GRAYSCALE)
    if depth is None:
        raise Exception(f'Could not read depth {depth_path}')

    depth = depth / 255
    depth = depth * (depth_max - depth_min) + depth_min
    depth = depth * scale_factor  # apply normalization

    return cam, image, depth


def draw_epipolar_geometry_between(cam1, cam2, img1, img2):
    F = cam1.fundamental_matrix_between(cam2)

    def draw_epipolar_geometry(point, color):
        point = torch.as_tensor(point, dtype=torch.float32)
        line = F.T @ homogenize_vec(point)
        cv.circle(img1, (int(point[0]), int(point[1])), radius=5, color=color, thickness=5)
        draw_epipolar_line(img2, line, color)
        return img1, img2

    return draw_epipolar_geometry


def test_epipolar_geometry_blender():
    model_dir = "/Users/uveyisakbas/Desktop/blender/dataset/0001"

    position_1, position_2 = 'front', 'top'
    cam1, img1, _ = load_blender_dataset(model_dir, position_1)
    cam2, img2, _ = load_blender_dataset(model_dir, position_2)

    draw = draw_epipolar_geometry_between(cam1, cam2, img1, img2)
    draw([280, 230], Colors.RED.value)

    cv.imshow(position_1, img1)
    cv.imshow(position_2, img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_depth_maps_blender():
    model_dir = "/Users/uveyisakbas/Desktop/blender/dataset/0001"

    normalize = True
    depth_th = 1.7 if normalize else 800

    cam1, img1, depth1 = load_blender_dataset(model_dir, 'front', normalize=normalize)
    is_valid = torch.tensor(depth1, dtype=torch.float32).flatten() < depth_th
    points1 = cam1.transform_to_world(cam1.create_point_cloud_by_depth_map(depth1)[is_valid])

    cam2, img2, depth2 = load_blender_dataset(model_dir, 'back', normalize=normalize)
    is_valid = torch.tensor(depth2, dtype=torch.float32).flatten() < depth_th
    points2 = cam2.transform_to_world(cam2.create_point_cloud_by_depth_map(depth2)[is_valid])

    cam3, img3, depth3 = load_blender_dataset(model_dir, 'top', normalize=normalize)
    is_valid = torch.tensor(depth3, dtype=torch.float32).flatten() < depth_th
    points3 = cam3.transform_to_world(cam3.create_point_cloud_by_depth_map(depth3)[is_valid])

    cam4, img4, depth4 = load_blender_dataset(model_dir, 'bottom',normalize=normalize)
    is_valid = torch.tensor(depth4, dtype=torch.float32).flatten() < depth_th
    points4 = cam4.transform_to_world(cam4.create_point_cloud_by_depth_map(depth4)[is_valid])

    points = torch.cat((points1, points2, points3, points4), dim=0)

    cloud = trimesh.PointCloud(points)
    trimesh.Scene(cloud).show()

    print('done')


def test_epipolar_lines():
    """ Test finding epipolar lines given the points on the reference image. """
    # TODO did not try to run the code after modification, double check.

    scene_ = Scene(scene_dir='scenes/scene_1')
    cameras, images = scene_.cameras, scene_.images

    cam1, cam2 = cameras['front'], cameras['top']
    img1, img2 = images['front'], images['top']

    img_w, img_h = cam1.sensor_resolution

    draw = draw_epipolar_geometry_between(cam1, cam2, img1, img2)

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
    scene_ = Scene(scene_dir='scenes/scene_1')
    cameras, images, masks, radius = scene_.cameras, scene_.images, scene_.masks, 420

    img_front, img_top = images['front'], images['top']
    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cam_front.sensor_resolution

    p1 = torch.tensor([img_w / 2 + 1, img_h / 2])
    cv.circle(img_front, (int(p1[0]), int(p1[1])), radius=5, color=Colors.RED.value, thickness=5)

    depth = 450
    p2 = cam_front.map_image_to_image(p1, depth, cam_top).squeeze()
    p2_expected = torch.tensor([641.0714, 335.8847])
    err_message = f'test_image_point_to_image_point failed: Expected:{p2_expected} | Actual: {p2} '
    assert torch.allclose(p2, p2_expected), err_message

    cv.circle(img_top, (int(p2[0]), int(p2[1])), radius=5, color=Colors.RED.value, thickness=5)

    print(f'P1:{p1} | P2:{p2} | img:{(img_w, img_h)} | depth:{depth}')

    cv.imshow('top', img_top)
    cv.imshow('front', img_front)
    cv.waitKey()
    cv.destroyAllWindows()


def test_depth_from_two_points():
    """ Test finding depth from two points which correspond to same point in 3D. """

    scene_ = Scene(scene_dir='scenes/scene_1')
    cameras = scene_.cameras

    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cam_front.sensor_resolution

    X1 = torch.tensor([img_w / 2 + 1, img_h / 2])
    X2 = torch.tensor([641.0714111328, 335.8846740723])

    F = cam_front.fundamental_matrix_between(cam_top)

    # Check epipolar constraint
    close_to_zero = homogenize_vec(X1).view(-1, 1).T @ F @ homogenize_vec(X2)
    zero = torch.tensor([0.0])
    print(close_to_zero)
    assert torch.allclose(close_to_zero, zero,
                          atol=1e-4), f'Assertion Failed: Expected:{close_to_zero} | Actual: {zero}'
    print('Depth between 2 points is: ', cam_front.get_depth_from_two_points(cam_top, X1, X2))


def test_depth_to_line_segments():
    """ Test finding an epipolar line segment given a point and an estimated depth value. """

    scene_ = Scene(scene_dir='scenes/scene_1')
    cameras, images, masks, radius = scene_.cameras, scene_.images, scene_.masks, 420

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

    point_top_max = cameras['front'].map_image_to_image(point, point_depth_max, cameras['top']).squeeze()
    point_top_min = cameras['front'].map_image_to_image(point, point_depth_min, cameras['top']).squeeze()

    point_top_max_expected = torch.tensor([535.4022, 358.7291])
    err_message = f'Assertion failed: Expected {point_top_max_expected} | Actual {point_top_max}'
    assert torch.allclose(point_top_max, point_top_max_expected), err_message

    point_top_min_expected = torch.tensor([549.6879, 486.9599])
    err_message = f'Assertion failed: Expected {point_top_min_expected} | Actual {point_top_min}'
    assert torch.allclose(point_top_min, point_top_min_expected), err_message

    point_top_max_, point_top_min_ = tuple(map(int, point_top_max)), tuple(map(int, point_top_min))

    cv.circle(img_front, point_, radius=10, color=Colors.RED.value, thickness=10)
    cv.line(img_top, point_top_max_, point_top_min_, color=Colors.RED.value, thickness=5)

    print(f'Point Depth: {point_depth} | Max: {point_depth_max} | Min: {point_depth_min}')
    cv.imshow('Top', img_top)
    cv.imshow('Front', img_front)
    cv.waitKey()


def test_transforming_relative_depth():
    """ Test linear transformation of relative depth by using absolute depth values of selected points. """

    scene_ = Scene(scene_dir='scenes/scene_1')
    cameras, images = scene_.cameras, scene_.images

    img_front, img_top = images['front'], images['top']
    cam_front, cam_top = cameras['front'], cameras['top']
    img_w, img_h = cameras['front'].sensor_resolution

    def helper_func(p_, d_, color=Colors.RED.value, log=False):
        cv.circle(img_front, (int(p_[0]), int(p_[1])), radius=5, color=color, thickness=5)
        p_t = cam_front.map_image_to_image(p_, d_, cam_top).squeeze()
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


def test_point_cloud_and_trimesh():
    """ Test the point cloud and mesh creation using relative depth, image and mask."""
    scene_ = Scene(scene_dir='scenes/scene_1')
    cameras, images, masks = scene_.cameras, scene_.images, scene_.masks

    relative_depth = 1 / torch.as_tensor(predict_depth(images['front']), dtype=torch.float32)
    # relative_depth = torch.as_tensor(masks['front'], dtype=torch.float32) * 420
    # relative_depth = relative_depth + (1 - masks['front']) * 840
    relative_depth = relative_depth.flatten()

    depth = relative_depth

    W, H = cameras['front'].sensor_resolution
    x = torch.arange(0, W, dtype=torch.float32)
    y = torch.arange(0, H, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    ones = torch.ones_like(grid_x)
    img_points = torch.stack((grid_x, grid_y, ones))

    cam_points = depth * (torch.inverse(cameras['front'].K) @ img_points)
    cam_points = cam_points.T

    # Put into a unit sphere.
    cam_points = cam_points - torch.mean(cam_points, dim=0)
    cam_points = cam_points / torch.norm(cam_points, dim=1).max()

    point_cloud = PointCloud(cam_points)

    SHOW, EXPORT = False, False

    if SHOW:
        point_cloud.show()

    if EXPORT:
        point_cloud.export(file_obj='fish_point_cloud.obj', file_type='obj')

    cam_points = cam_points.view(H, W, 3)

    print(images['front'].shape, cam_points.shape)

    vertices, faces, face_colors = points_3d_to_trimesh(images['front'], pts3d=cam_points, valid=masks['front'])
    fish = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)

    # dimensions1 = [0.1, 0.1, 0.1]
    # box = trimesh.creation.box(extents=dimensions1)
    # scene = trimesh.Scene([fish, box])

    scene = trimesh.Scene([fish])
    scene.show()


# test_epipolar_lines()
# test_image_point_to_image_point()
# test_depth_from_two_points()
# test_depth_to_line_segments()
# test_transforming_relative_depth()
# test_point_cloud_and_trimesh()
# test_epipolar_geometry_blender()
test_depth_maps_blender()
