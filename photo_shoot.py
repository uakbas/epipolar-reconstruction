import torch
import trimesh
import numpy as np
import cv2 as cv
import os

from camera import Camera, get_rot_mat_x


def visualize_points(points: np.ndarray, colors: np.ndarray = None):
    scene = trimesh.Scene()
    cloud = trimesh.PointCloud(points, colors)
    scene.add_geometry(cloud)
    scene.add_geometry(trimesh.creation.axis(origin_size=5))

    # front
    cam1 = trimesh.creation.box(extents=(10, 10, 50))
    cam1.apply_translation([100, 0, 0])
    # scene.add_geometry(cam1)

    # top
    cam2 = trimesh.creation.box(extents=(10, 10, 5))
    cam2.apply_translation([0, -100, 0])
    # scene.add_geometry(cam2)

    scene.show()
    # scene.export('test.obj', 'obj')


def create_sphere(volume: torch.Tensor, radius):
    points = volume.reshape(-1, 3)
    # sampling_step=1
    # sampling_mask = torch.zeros(points.shape[0], dtype=torch.int)
    # sampling_mask[::sampling_step] = 1
    return points[points.norm(dim=-1) < radius]


class FourCamScene:

    def __init__(self):
        self.cam_distance = 420
        self.cameras = {
            'front': Camera(get_rot_mat_x(0), torch.tensor([0, 0, -self.cam_distance], dtype=torch.float32)),
            'top': Camera(get_rot_mat_x(270), torch.tensor([0, -self.cam_distance, 0], dtype=torch.float32)),
            'back': Camera(get_rot_mat_x(180), torch.tensor([0, 0, self.cam_distance], dtype=torch.float32)),
            'bottom': Camera(get_rot_mat_x(90), torch.tensor([0, self.cam_distance, 0], dtype=torch.float32)),
        }

        self.limits = [(-(self.cam_distance - padding), (self.cam_distance - padding) + 1) for padding in [1, 2, 3]]

        y_line = torch.arange(*self.limits[0], dtype=torch.float32)
        x_line = torch.arange(*self.limits[1], dtype=torch.float32)
        z_line = torch.arange(*self.limits[2], dtype=torch.float32)

        grid_y, grid_x, grid_z = torch.meshgrid([y_line, x_line, z_line], indexing='ij')
        self.volume = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        self.volume_occupancy = torch.zeros(*self.volume.shape[:3])

    @property
    def scene_object(self):
        return self.volume[self.volume_occupancy == 1]

    @property
    def volume_dims(self):
        return torch.as_tensor(self.volume_occupancy.shape, dtype=torch.int)

    def visualize(self):
        visualize_points(self.scene_object.numpy())

    def set_occupancy_by_3d_points(self, points):
        points = torch.round(points).to(dtype=torch.int)
        volume_dims = self.volume_dims // 2
        displacement = volume_dims[[1, 0, 2]]  # (y, x, z) --> (x, y, z)
        points = points + displacement

        self.volume_occupancy = torch.zeros_like(self.volume_occupancy)  # TODO remove
        self.volume_occupancy[points[:, 1], points[:, 0], points[:, 2]] = 1

    def generate_sphere(self, sphere_radius):
        points = create_sphere(self.volume, radius=sphere_radius)
        self.set_occupancy_by_3d_points(points)

    def load_object(self, file: str, max_norm):
        mesh = trimesh.load(file)

        mesh.apply_translation(-mesh.center_mass)
        mesh.apply_scale(max_norm / mesh.extents.max())

        points = torch.tensor(mesh.voxelized(pitch=1).points, dtype=torch.float32)
        points[:, 0:2] = -points[:, 0:2]  # adjust for opencv convention: x,y,z to -x,-y,z

        self.set_occupancy_by_3d_points(points)

    """
    def translate(self, translation):
        self.scene_object = self.scene_object + torch.as_tensor(translation)
    """

    def shoot_by_position(self, position: torch, **kwargs):
        perturbation_distance = kwargs.get('perturbation_distance', 0.5)

        cam: Camera = self.cameras[position]

        # create an empty image
        W, H = cam.sensor.resolution
        image = torch.zeros((H, W)).numpy()

        # perturbate the original points to decrease aliasing effect
        points = self.scene_object.reshape(-1, 3)
        perturbation = torch.full_like(points, fill_value=perturbation_distance)
        points = torch.cat([points, points + perturbation, points - perturbation], dim=0)

        # project onto the image plane
        image_points, _ = cam.project(points)
        image_points = torch.round(image_points).to(torch.int)  # pixel coordinates

        # is_inside the image plane
        is_inside_image = (0 <= image_points[:, 0]) & (image_points[:, 0] < W) & (
                0 <= image_points[:, 1]) & (image_points[:, 1] < H)

        # set the projected image points
        image_points = image_points[is_inside_image]
        image[image_points[:, 1], image_points[:, 0]] = 255

        # apply dilation to remove aliasing
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

    def export_occupancy(self, save_dir):
        torch.save(self.volume_occupancy, os.path.join(save_dir, 'volume_occupancy.pt'))


def create_bunny_data():
    four_cam_scene = FourCamScene()
    object_dir = os.path.join('scene_objects', 'bunny')
    four_cam_scene.load_object(os.path.join(object_dir, 'bun_zipper.ply'), max_norm=200)
    four_cam_scene.shoot(show=False, target_dir=object_dir)
    four_cam_scene.export_occupancy(object_dir)
    four_cam_scene.visualize()


def main():
    create_bunny_data()


if '__main__' == __name__:
    main()
