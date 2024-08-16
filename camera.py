import torch
import math
from dataclasses import dataclass
from epipolar_geometry import fundamental_matrix


def get_rot_mat_x(deg):
    rad = deg / 180 * math.pi
    return torch.tensor([
        [1, 0, 0],
        [0, math.cos(rad), -math.sin(rad)],
        [0, math.sin(rad), math.cos(rad)]
    ], dtype=torch.float32)


def homogenize(matrix):
    column_number = matrix.shape[1]
    last_row = torch.ones(column_number).unsqueeze(0)
    return torch.cat((matrix, last_row), 0)


def homogenize_vec(vector):
    return torch.cat((vector, torch.tensor([1])), dim=0)


def create_transformation_matrix(R, t):
    return torch.cat((R.T, -R.T @ t.unsqueeze(1)), dim=1)


def create_projection_matrix(R, t, K):
    trans_mat = create_transformation_matrix(R, t)
    project_mat = K @ trans_mat
    return project_mat


@dataclass
class Sensor:
    # TODO Find more accurate camera parameters.
    focal_length: float = 8
    size: tuple = (11.33, 7.13)
    resolution: tuple = (1280, 800)


class Camera:
    def __init__(self, R, t, sensor: Sensor = None):
        self.R = R
        self.t = t
        self.sensor = sensor if sensor is not None else Sensor()
        self.K = self.calibration_matrix(self.sensor)

    @property
    def sensor_resolution(self):
        return self.sensor.resolution

    @staticmethod
    def calibration_matrix(sensor: Sensor):
        img_w, img_h = sensor.resolution
        fy = sensor.focal_length * img_h / sensor.size[1]
        fx = fy
        return torch.tensor([
            [fx, 0, img_w / 2],
            [0, fy, img_h / 2],
            [0, 0, 1]
        ], dtype=torch.float32)

    def transformation_between(self, camera: 'Camera'):
        """
        The world reference system is associated to the first camera (self).
        The second camera offset by a rotation R and by a translation t.

        :param camera: Target camera
        :return:
        """
        R = self.R.T @ camera.R
        t = self.R.T @ (camera.t - self.t)
        return R, t

    def transformation_matrix_to_world(self):
        # TODO Use only rotation and translation. Do not need to create a hypothetical world camera.
        # Hypothetical world camera
        world_camera = Camera(torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))
        R, t = self.transformation_between(world_camera)
        return create_transformation_matrix(R, t)

    def transform_to_world(self, points: torch.Tensor) -> torch.Tensor:
        N, THREE = points.shape
        assert THREE == 3, 'Points must have shape of (N,3) where N is the number of points.'
        points = homogenize(points.T)  # 4xN
        trans_mat = self.transformation_matrix_to_world()
        points_world = trans_mat @ points
        return points_world.T

    def projection_between(self, camera: 'Camera'):
        """
        The world reference system is associated to the first camera (self).
        The matrix projects the points in reference coordinate system onto the image coordinates of the target camera.

        :param camera: Target camera
        :return:
        """
        R, t = self.transformation_between(camera)
        return create_projection_matrix(R, t, camera.K)

    def map_image_to_cam(self, points, depths):
        """
        Maps points on the image plane to the camera coordinate system by using depth information.

        :param points: Points on the image plane
        :param depths: Depths of the points
        :return: 3D points on the camera coordinate system
        """

        depths = torch.as_tensor(depths, dtype=torch.float32)
        points = torch.as_tensor(points, dtype=torch.float32)
        if points.dim() == 1:
            points = points.unsqueeze(0)

        N, TWO = points.shape
        assert TWO == 2, 'Points must have shape of (N,2) where N is the number of points.'

        points = homogenize(points.T)

        cam_points = (torch.inverse(self.K) @ points) * depths
        cam_points = cam_points.T

        return cam_points

    def map_image_to_image(self, points, depths, camera: 'Camera'):
        """
        Maps points on the reference image plane in to the points on the target image plane.

        :param points: Points on the reference image plane
        :param depths: Depths of points on the reference image plane
        :param camera: Target camera
        :return: 2D points on the target image plane
        """
        cam_points = self.map_image_to_cam(points, depths).T

        target_image_points = self.projection_between(camera) @ homogenize(cam_points)
        target_image_points = (target_image_points / target_image_points[2, :])[:2]  # Normalize by dividing z value.
        target_image_points = target_image_points.T
        return target_image_points

    def create_point_cloud_by_depth_map(self, depth_map):
        depths = depth_map.flatten()

        W, H = self.sensor_resolution
        x = torch.arange(0, W, dtype=torch.float32)
        y = torch.arange(0, H, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

        img_points = torch.stack((grid_x.flatten(), grid_y.flatten())).T
        cam_points = self.map_image_to_cam(img_points, depths)
        return cam_points

    def get_depth_from_two_points(self, camera: 'Camera', point1, point2):
        # TODO handle zero cases
        # TODO make sure at least one of the dividend elements will be nonzero
        # TODO prove that the third element of the dividend will be always zero

        X1 = homogenize_vec(point1)
        X2 = homogenize_vec(point2)
        R, t = self.transformation_between(camera)
        K1, K2 = self.K, camera.K
        e = torch.tensor([0, 0, 1], dtype=torch.float32)  # Only get the third element: e @ R.T @ t = (R.T @ t)[2]

        quotient = R.T @ t - (e @ R.T @ t) * torch.inverse(K2) @ X2
        dividend = (R.T @ torch.inverse(K1) @ X1) - (e @ R.T @ torch.inverse(K1) @ X1) * torch.inverse(K2) @ X2
        arg_max_abs = torch.argmax(torch.abs(dividend))

        depth = float('inf')
        try:
            depth = quotient[arg_max_abs] / dividend[arg_max_abs]
        except ZeroDivisionError:
            print('Finding depth from two points failed: division by zero.')

        return depth

    def fundamental_matrix_between(self, camera: 'Camera'):
        R, t = self.transformation_between(camera)
        return fundamental_matrix(R, t, self.K, camera.K)
