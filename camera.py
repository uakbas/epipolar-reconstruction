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
    mat = torch.cat((R, -R.T @ t.unsqueeze(1)), dim=1)
    return homogenize(mat)


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
        # Transformation between 2 cameras.
        R = self.R.T @ camera.R
        t = self.R.T @ (camera.t - self.t)
        return R, t
