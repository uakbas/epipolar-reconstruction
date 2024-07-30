import torch
import math
from dataclasses import dataclass


def getRotMatX(deg):
    rad = deg / 180 * math.pi
    return torch.tensor([
        [1, 0, 0],
        [0, math.cos(rad), -math.sin(rad)],
        [0, math.sin(rad), math.cos(rad)]
    ], dtype=torch.float32)


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
