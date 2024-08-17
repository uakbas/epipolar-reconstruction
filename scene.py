import os.path
import torch
import trimesh
import cv2 as cv
import numpy as np
from numpy.polynomial import Polynomial as Poly
from typing import Optional, Dict
from collections import namedtuple
from camera import get_rot_mat_x, Camera
from depth_prediction import predict_depth
import geometry as geo
import epipolar_geometry as ege
import visualization_tools as vt
from visualization_tools import Colors


class Scene:
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.image_dir = os.path.join(scene_dir, 'images')
        self.mask_dir = os.path.join(scene_dir, 'masks')
        self.mesh_mask_dir = os.path.join(scene_dir, 'mesh_masks')

        self.positions = ['front', 'top', 'back', 'bottom']

        self.cameras = self.load_cameras()
        self.images = self.load_images()
        self.masks = self.load_masks()
        self.mesh_masks = self.load_mesh_masks()
        self.point_clouds: Dict[str, Optional[torch.Tensor]] = {pos: None for pos in self.positions}

    @property
    def available_point_clouds(self):
        return {pos: cloud for pos, cloud in self.point_clouds.items() if cloud is not None}

    @staticmethod
    def load_cameras():
        radius = 420  # Distance to the center of the unit.
        return {
            'front': Camera(get_rot_mat_x(0), torch.tensor([0, 0, -radius], dtype=torch.float32)),
            'top': Camera(get_rot_mat_x(270), torch.tensor([0, -radius, 0], dtype=torch.float32)),
            'back': Camera(get_rot_mat_x(180), torch.tensor([0, 0, radius], dtype=torch.float32)),
            'bottom': Camera(get_rot_mat_x(90), torch.tensor([0, radius, 0], dtype=torch.float32)),
        }

    def load_images(self):
        image_names = self.positions
        images = {}
        for name in image_names:
            image_path = os.path.join(self.image_dir, f'{name}.png')
            image = cv.imread(image_path)
            if image is None:
                print(f'Could not read image {image_path}')
                continue

            images[name] = image

        return images

    def load_masks(self):
        mask_names = self.positions
        masks = {}
        for name in mask_names:
            mask_path = os.path.join(self.mask_dir, f'{name}.png')
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if mask is None:
                print(f'Could not read mask {mask_path}')
                continue

            _, mask_binary = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
            mask_binary = mask_binary / 255

            masks[name] = mask_binary

        return masks

    def load_mesh_masks(self):
        mask_names = self.positions
        masks = {}
        for name in mask_names:
            mask_path = os.path.join(self.mesh_mask_dir, f'{name}.png')
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if mask is None:
                masks[name] = None
            else:
                _, mask_binary = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
                mask_binary = mask_binary / 255

                masks[name] = mask_binary

        return masks

    def convert_to_meshes(self):
        meshes = {}
        for position, cloud in self.available_point_clouds.items():
            img = self.images[position]
            mask = self.mesh_masks[position] if self.mesh_masks[position] is not None else self.masks[position]
            vertices, faces, face_colors = vt.points_3d_to_trimesh(img, cloud, valid=mask)
            meshes[position] = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)

        return meshes

    def masked_point_clouds(self):
        masked_point_clouds = {}
        for position, cloud in self.available_point_clouds.items():
            mask = torch.as_tensor(self.masks[position], dtype=torch.bool).flatten()
            masked_point_clouds[position] = cloud[mask]

        return masked_point_clouds

    def generate_point_clouds(self):
        Configuration = namedtuple('Configuration', ['position', 'helper', 'scaling_ratio'])
        configurations = [
            Configuration(position='front', helper='top', scaling_ratio=0.7),
            Configuration(position='back', helper='top', scaling_ratio=0.7),
            Configuration(position='top', helper='front', scaling_ratio=0.7),
        ]
        for configuration in configurations:
            self.point_clouds[configuration.position] = self.create_point_cloud(*configuration)

    @staticmethod
    def get_border_types(position, position_helper):
        # TODO implement this function using the camera positions.
        if position == 'front' and position_helper == 'top':
            return 'middle', 'max'
        if position == 'top' and position_helper == 'front':
            return 'middle', 'min'
        if position == 'back' and position_helper == 'top':
            return 'middle', 'min'

        raise ValueError(f'Not implemented for the position pair: {(position, position_helper)}')

    def create_point_cloud(self, position, position_helper, scaling_ratio=0.7, visualize=False):
        """
        Create a mesh for position, utilizing position_helper.

        :param position: The position for which to create the mesh.
        :param position_helper: The position which helps to create mesh.
        :param scaling_ratio: The ratio of what percentage the relative depth is scaled by absolute depth.
        :param visualize: Show visualization of the process if True.
        :return: trimesh.Trimesh
        """
        cam: Camera = self.cameras[position]
        cam_helper: Camera = self.cameras[position_helper]
        img, img_helper = self.images[position], self.images[position_helper]
        mask, mask_helper = self.masks[position], self.masks[position_helper]
        border_type, border_type_helper = self.get_border_types(position, position_helper)

        border_points_helper, _ = geo.get_horizontal_borders(mask_helper, border_type_helper)
        border_points, border_idxs = geo.get_horizontal_borders(mask, border_type)

        # Find coefficients of the polynomial which passes through the border points
        poly_coefficients_helper = geo.fit_polynomial(border_points_helper)

        # A line is [a, b, c] ==> 0 = ax + by + c
        lines = cam.map_points_to_epipolar_lines(cam_helper, border_points)  # Nx3
        # Find intersections between epipolar lines and the polynomial on the helper image.
        intersections = geo.get_poly_intersections_for_lines(lines, poly_coefficients_helper, img_helper)  # Nx2

        n_intersection = intersections.shape[0]
        assert n_intersection == border_points.shape[0], 'Number of intersections and border points do not match.'

        depths_abs = torch.zeros(n_intersection)
        for i in range(n_intersection):
            depths_abs[i] = cam.get_depth_from_two_points(cam_helper, border_points[i], intersections[i])

        depth_map_rel = 1 / torch.as_tensor(predict_depth(img), dtype=torch.float32)
        depths_rel = depth_map_rel[border_points.to(torch.int)[:, 1], border_points.to(torch.int)[:, 0]]

        scale_coefficients = (depths_abs / depths_rel) * scaling_ratio
        translation_coefficients = depths_abs * (1 - scaling_ratio)

        mask_w = mask.shape[1]
        scale_coefficients_complete = torch.zeros(mask_w)
        scale_coefficients_complete[border_idxs] = scale_coefficients

        translation_coefficients_complete = torch.zeros(mask_w)
        translation_coefficients_complete[border_idxs] = translation_coefficients

        depth_map_abs = depth_map_rel * scale_coefficients_complete + translation_coefficients_complete

        if visualize:
            img_, img_helper_ = np.copy(img), np.copy(img_helper)
            vt.draw_points(img_helper_, border_points_helper)
            vt.draw_polynomial(img_helper_, Poly(poly_coefficients_helper))

            for i in np.arange(0, n_intersection, 100).tolist():
                print('Depth: ', cam.get_depth_from_two_points(cam_helper, border_points[i], intersections[i]))
                cv.circle(img_, (int(border_points[i][0]), int(border_points[i][1])), 5, Colors.GREEN.value, 5)
                cv.circle(img_helper_, (int(intersections[i][0]), int(intersections[i][1])), 5, Colors.GREEN.value, 5)
                ege.draw_epipolar_line(img_helper_, lines[i])

            cv.destroyAllWindows()
            cv.imshow(f'Create Point Cloud: {position}', img_)
            cv.imshow(f'Create Point Cloud: {position_helper}', img_helper_)
            cv.waitKey()

        cam_points = cam.create_point_cloud_by_depth_map(depth_map_abs)
        return cam.transform_to_world(cam_points)
