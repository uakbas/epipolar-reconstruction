import torch
import trimesh
import os
import numpy as np
import cv2 as cv

from camera import Camera, get_rot_mat_x, Sensor


class VolumePhotoGenerator:
    # Volume is created using the axes in the y, x, z order.
    # For a convenient imagination, we could consider y goes down, x goes right and z goes inside.
    # This is to imagine the volume as 3D tensor.
    # As for cameras, everything is same. Just take the front camera as reference.
    # Rotate the scene around the origin so that the point of view matches with the front camera.

    # TODO Make the volume sampled step by step instead of fixed volume radius.
    # TODO This will handle the scaling process at the same time.

    def __init__(self):
        # Set cameras.
        cam_distance = 420
        sensor = Sensor(focal_length=9.2, size=(11.33, 7.13), resolution=(256, 160))
        self.cameras = {
            'front': Camera(get_rot_mat_x(0), torch.tensor([0, 0, -cam_distance], dtype=torch.float32), sensor),
            'top': Camera(get_rot_mat_x(270), torch.tensor([0, -cam_distance, 0], dtype=torch.float32), sensor),
            'back': Camera(get_rot_mat_x(180), torch.tensor([0, 0, cam_distance], dtype=torch.float32), sensor),
            'bottom': Camera(get_rot_mat_x(90), torch.tensor([0, cam_distance, 0], dtype=torch.float32), sensor),
        }

        # It is chosen as multiple of 16 to make it scalable.
        # This is smaller than cam_distance. In this case, the areas close to cameras will be out of use.
        self.volume_radius = 384  # 16 * 8 * 3

        y_line = torch.arange(-self.volume_radius, self.volume_radius, dtype=torch.float32)
        x_line = torch.arange(-self.volume_radius, self.volume_radius, dtype=torch.float32)
        z_line = torch.arange(-self.volume_radius, self.volume_radius, dtype=torch.float32)

        # 1st dim: y (down) | 2nd dim: x (right) | 3rd dim: z (forward)
        grid_y, grid_x, grid_z = torch.meshgrid([y_line, x_line, z_line], indexing='ij')

        # Each point in the volume has (x, y, z) coordinates.
        self.volume = torch.stack([grid_x, grid_y, grid_z], dim=-1)

        # For each point ==> Occupied 1 | Empty 0 | All zeros initially.
        self._volume_occupancy = torch.zeros(*self.volume.shape[:3])

    @property
    def scene_object(self):
        return self.volume[self.volume_occupancy == 1]

    @property
    def volume_dims(self):
        return torch.as_tensor(self.volume_occupancy.shape, dtype=torch.int)

    @property
    def volume_occupancy(self):
        return self._volume_occupancy

    @volume_occupancy.setter
    def volume_occupancy(self, value):
        # Reset volume occupancy
        self._volume_occupancy = torch.zeros_like(self._volume_occupancy)
        # Set volume occupancy
        self._volume_occupancy = value

    @property
    def scene_object(self):
        return self.volume[self.volume_occupancy == 1]

    @property
    def volume_dims(self):
        return torch.as_tensor(self.volume_occupancy.shape, dtype=torch.int)

    def set_occupancy_by_points_3d(self, points):
        assert points.ndim == 2 and points.shape[1] == 3, 'Points must have shape of (N, 3)'

        self.reset_occupancy()

        volume_dims = self.volume_dims // 2
        displacement = volume_dims[[1, 0, 2]]  # (y, x, z) --> (x, y, z)
        points = torch.round(points).to(dtype=torch.int) + displacement

        volume_occupancy = torch.zeros_like(self.volume_occupancy)
        volume_occupancy[points[:, 1], points[:, 0], points[:, 2]] = 1
        self.volume_occupancy = volume_occupancy

    def scale_occupancy(self, scale=1):
        assert scale == 1 or scale % 2 == 0, 'Invalid scale value.'
        assert (self.volume_dims % scale == 0).all(), 'Shape values must be multiple of the scale.'

        occ = self.volume_occupancy
        shape = torch.stack([self.volume_dims // scale, torch.full_like(self.volume_dims, scale)], dim=0).T.flatten()
        # occ_scaled = occ.view(*shape).sum(dim=(1, 3, 5)) > (scale * scale * scale) / 2
        occ_scaled = occ.view(*shape).sum(dim=(1, 3, 5)) > 0
        occ_scaled = occ_scaled.to(torch.int32)

        return occ_scaled

    def load_object(self, file: str, max_norm):
        assert max_norm < torch.max(self.volume_dims // 2) * 0.95, 'Max norm is too big.'

        mesh = trimesh.load(file)

        # Centralize the mesh around origin.
        mesh.apply_translation(-mesh.center_mass)

        # Scale the mesh to make sure it is within volume limits.
        mesh.apply_scale(max_norm / mesh.extents.max())

        points = torch.tensor(mesh.voxelized(pitch=1).points, dtype=torch.float32)

        # Adjust for opencv convention: x,y,z to -x,-y,z.
        points[:, 0:2] = -points[:, 0:2]

        self.set_occupancy_by_points_3d(points)

    def visualize(self, colors: np.ndarray = None):
        scene = trimesh.Scene()

        scene.add_geometry(trimesh.PointCloud(self.scene_object.numpy(), colors))
        scene.add_geometry(trimesh.creation.axis(origin_size=5))

        cam_front = trimesh.creation.box(extents=(10, 10, 50))
        cam_front.apply_translation([0, 0, -100])
        scene.add_geometry(cam_front)

        scene.show()

    def visualize_occ(self, scale=1):
        occ_scaled = self.scale_occupancy(scale)
        voxel_grid = trimesh.voxel.VoxelGrid(occ_scaled.numpy())
        voxel_grid.show()

    def shoot_by_position(self, position: torch, **kwargs):
        perturbation_distance = kwargs.get('perturbation_distance', 0.5)

        cam: Camera = self.cameras[position]

        # Create an empty image.
        W, H = cam.sensor.resolution
        image = torch.zeros((H, W)).numpy()

        # Perturbate the original points to decrease aliasing effect.
        points = self.scene_object.reshape(-1, 3)
        perturbation = torch.full_like(points, fill_value=perturbation_distance)
        points = torch.cat([points, points + perturbation, points - perturbation], dim=0)

        # Project onto the image plane.
        image_points, _ = cam.project(points)
        image_points = torch.round(image_points).to(torch.int)  # pixel coordinates

        # @formatter:off
        # Use only the image points which are inside the image plane.
        is_inside_image = (0 <= image_points[:, 0]) & (image_points[:, 0] < W) & (0 <= image_points[:, 1]) & (image_points[:, 1] < H)
        image_points = image_points[is_inside_image]
        # @formatter:on

        # Set the projected image points.
        image[image_points[:, 1], image_points[:, 0]] = 255

        # Apply dilation to remove aliasing.
        image = cv.dilate(image, np.ones((4, 4), np.uint8), iterations=1)
        image = cv.GaussianBlur(image, (5, 5), 0)
        image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]

        return image

    def shoot(self, target_dir, show=False):
        for i, position in enumerate(self.cameras.keys()):
            image = self.shoot_by_position(position)
            if show:
                cv.imshow(position, image)
            else:
                cv.imwrite(os.path.join(target_dir, f'{i}_{position}.png'), image)

        if show:
            cv.waitKey()

    def export_occupancy(self, save_dir, scale=1):
        occ_scaled = self.scale_occupancy(scale)
        torch.save(occ_scaled, os.path.join(save_dir, f'volume_occupancy_{scale}x.pt'))

    def export_scene_object(self, save_dir):
        torch.save(
            {
                'volume_radius': self.volume_radius,
                'scene_object': self.scene_object,
            },
            os.path.join(save_dir, 'scene_conf.pt')
        )

    def reuse_scene_object(self):
        # TODO Test.
        distance = self.volume_radius
        points = self.scene_object.to(dtype=torch.int)

        vol_occupancy_shape = [2 * distance, 2 * distance, 2 * distance]
        vol_occupancy = torch.zeros(vol_occupancy_shape)

        indexes = points + torch.tensor([distance, distance, distance])  # x, y, z
        vol_occupancy[indexes[:, 1], indexes[:, 0], indexes[:, 2]] = 1

        self.volume_occupancy = vol_occupancy


def create_bunny_data():
    object_dir = os.path.join('scene_objects', 'bunny')
    scale_coefficient = 8

    gen = VolumePhotoGenerator()
    gen.load_object(os.path.join(object_dir, 'bun_zipper.ply'), max_norm=192)

    gen.shoot(show=False, target_dir=object_dir)
    # gen.export_scene_object(object_dir)
    gen.export_occupancy(object_dir, scale=scale_coefficient)
    gen.visualize_occ(scale=scale_coefficient)
    # gen.visualize()


def create_utah_teapot_data():
    object_dir = os.path.join('scene_objects', 'utah_teapot')
    gen = VolumePhotoGenerator()
    gen.load_object(os.path.join(object_dir, 'utah_teapot.obj'), max_norm=192)
    gen.shoot(show=False, target_dir=object_dir)
    gen.visualize_occ(scale=4)


def main():
    create_bunny_data()
    # create_utah_teapot_data()


if '__main__' == __name__:
    main()
