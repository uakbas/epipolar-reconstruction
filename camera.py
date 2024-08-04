import torch
import math
from dataclasses import dataclass


def get_rot_mat_x(deg):
    rad = deg / 180 * math.pi
    return torch.tensor([
        [1, 0, 0],
        [0, math.cos(rad), -math.sin(rad)],
        [0, math.sin(rad), math.cos(rad)]
    ], dtype=torch.float32)


def homogenize(matrix):
    column_number = matrix.shape[1]
    last_row = torch.zeros(column_number)
    last_row[-1] = 1
    last_row = last_row.unsqueeze(0)
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
    focal_length: float = 8
    size: tuple = (11.33, 7.13)
    resolution: tuple = (1280, 800)


class Camera:
    def __init__(self, R, t, sensor: Sensor = None):
        self.R = R
        self.t = t
        self.sensor = sensor if sensor is not None else Sensor()
        self.K = self.calibration_matrix(self.sensor)

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

    def projection_between(self, camera: 'Camera'):
        """
        The world reference system is associated to the first camera (self).
        The matrix projects the points in reference coordinate system onto the image coordinates of the target camera.

        :param camera: Target camera
        :return:
        """
        R, t = self.transformation_between(camera)
        return create_projection_matrix(R, t, camera.K)

    def map_image_plane_to_3d(self, point, depth):
        """
        Maps a point on the image plane to the camera coordinate system by using depth information.

        :param point: A point on the image plane
        :param depth: Depth of the point
        :return: 3D point on the camera coordinate system
        """
        # K^-1 @ X_image @ d --> X_camera
        return (torch.inverse(self.K) @ homogenize_vec(point)) * depth
