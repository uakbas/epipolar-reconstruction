import os.path
import torch
import trimesh
import cv2 as cv
import numpy as np
from numpy.polynomial import Polynomial as Poly
from typing import Optional, Dict, List, Tuple
from collections import namedtuple
from camera import get_rot_mat_x, Camera
from depth_prediction import predict_depth
import geometry as geo
import epipolar_geometry as ege
import visualization_tools as vt
from visualization_tools import Colors
from voxel_voter import get_votes_for_volume


class Scene:
    DepthConfiguration = namedtuple('Configuration', ['position', 'helper', 'scaling_ratio'])

    def __init__(self, scene_dir, depth_configurations: Optional[List[Tuple[str, str, float]]] = None):
        self.scene_dir = scene_dir
        self.image_dir = os.path.join(scene_dir, 'images')
        self.mask_dir = os.path.join(scene_dir, 'masks')
        self.mesh_mask_dir = os.path.join(scene_dir, 'mesh_masks')

        self.positions = ['front', 'top', 'back', 'bottom']

        self.cameras = self.load_cameras()
        self.images = self.load_images()
        self.masks = self.load_masks()
        self.mesh_masks = self.load_mesh_masks()

        if depth_configurations is not None:
            confs = []
            for conf in depth_configurations:
                position, helper, scaling_ratio = conf
                if position not in self.positions or helper not in self.positions:
                    raise ValueError(f'position and helper must be in the positions: {self.positions}')
                confs.append(self.DepthConfiguration(position, helper, scaling_ratio))
            self.depth_configurations = confs
        else:
            self.depth_configurations = [
                self.DepthConfiguration(position='front', helper='top', scaling_ratio=0.7),
                self.DepthConfiguration(position='back', helper='top', scaling_ratio=0.7),
                self.DepthConfiguration(position='top', helper='front', scaling_ratio=0.7),
                self.DepthConfiguration(position='bottom', helper='back', scaling_ratio=0.7),
            ]
        self.depth_maps: Dict[str, Optional[torch.Tensor]] = {pos: None for pos in self.positions}
        self.generate_depth_maps()

        self.point_clouds: Dict[str, Optional[torch.Tensor]] = {pos: None for pos in self.positions}

    @property
    def configurations_str(self):
        return '_'.join(
            [f'{conf.position}{str(conf.scaling_ratio).replace('.', '')}' for conf in self.depth_configurations])

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

    def generate_depth_maps(self):
        visualize = False
        for configuration in self.depth_configurations:
            position, position_helper, scaling_ratio = configuration
            depth_map = self.get_depth_map(position, position_helper, scaling_ratio, visualize)
            self.depth_maps[position] = depth_map

    @staticmethod
    def get_border_types(position, position_helper):
        # TODO implement this function using the camera positions.
        if position == 'front' and position_helper == 'top':
            return 'middle', 'max'
        if position == 'top' and position_helper == 'front':
            return 'middle', 'min'
        if position == 'back' and position_helper == 'top':
            return 'middle', 'min'
        if position == 'bottom' and position_helper == 'back':
            return 'middle', 'min'

        raise ValueError(f'Not implemented for the position pair: {(position, position_helper)}')

    def get_depth_map(self, position, position_helper, scaling_ratio, visualize):
        """
        Derive depth map for position by utilizing position_helper

        :param position: The position for which to derive depth map.
        :param position_helper: The position which helps to derive depth map.
        :param scaling_ratio: The ratio of what percentage the relative depth is scaled by absolute depth.
        :param visualize: Show visualization of the process if True.
        """
        cam: Camera = self.cameras[position]
        cam_helper: Camera = self.cameras[position_helper]
        img, img_helper = self.images[position], self.images[position_helper]
        mask, mask_helper = self.masks[position], self.masks[position_helper]
        border_type, border_type_helper = self.get_border_types(position, position_helper)

        border_points_helper, _ = geo.get_horizontal_borders(mask_helper, border_type_helper)

        border_points, _ = geo.get_horizontal_borders(mask, border_type)
        border_points_aligned, border_idxs = geo.align_points_by_polynomial(border_points, mask)
        if visualize:
            img_ = np.copy(img)
            vt.draw_points(img_, border_points, color=Colors.RED.value)
            vt.draw_points(img_, border_points_aligned, color=Colors.GREEN.value)
            cv.imshow(f'Aligned Border Points | {position}', img_)

        border_points = border_points_aligned  # Rename

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

            cv.imshow(f'Get Depth Map: {position}', img_)
            cv.imshow(f'Get Depth Map: {position_helper}', img_helper_)
            cv.waitKey()
            cv.destroyAllWindows()

        return depth_map_abs

    def generate_point_clouds(self):
        # TODO use only the positions which have depth map.
        positions = self.positions
        for position in positions:
            self.point_clouds[position] = self.create_point_cloud(position)

    def create_point_cloud(self, position):
        # TODO return None incase depth map is missing.
        cam = self.cameras[position]
        depth_map = self.depth_maps[position]

        assert depth_map is not None, f'No depth map for {position}'

        cam_points = cam.create_point_cloud_by_depth_map(depth_map)
        return cam.transform_to_world(cam_points)

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

    def generate_voxel_volume_by_voting(self):
        votes = []
        points_worlds = []
        for position in self.positions:
            depth_map = self.depth_maps[position]
            vote, points_world = self.create_voxel_volume(position, depth_map)
            votes.append(vote)
            points_worlds.append(points_world)

        points_world = points_worlds[0]
        total_votes = torch.sum(torch.stack(votes), dim=0)
        is_valid_voxel = total_votes > 1
        valid_voxel_points = points_world[is_valid_voxel]
        return valid_voxel_points

    def create_voxel_volume(self, position, depth_map, radius=420, depth_margin=50, sampling_frequency=2):
        volume_left_lim, volume_right_lim = -radius + 1, radius + 1
        x_line = torch.arange(volume_left_lim, volume_right_lim, sampling_frequency)
        y_line = torch.arange(volume_left_lim, volume_right_lim, sampling_frequency)
        z_line = torch.arange(volume_left_lim, volume_right_lim, sampling_frequency)
        volume = torch.stack(torch.meshgrid([x_line, y_line, z_line], indexing='ij'), dim=-1)
        x_len, y_len, z_len, THREE = volume.shape

        mask = torch.as_tensor(self.masks[position])
        cam: Camera = self.cameras[position]
        P = cam.projection_matrix

        depth_map_max, depth_map_min = (depth_map + depth_margin), (depth_map - depth_margin)
        depth_maps = torch.stack([depth_map_min, depth_map_max], dim=-1)

        volume_votes = get_votes_for_volume(volume, P, mask, depth_maps)

        return volume_votes.flatten(), volume.view(x_len * y_len * z_len, 3)
