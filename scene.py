import os.path
import cv2 as cv
import torch
from camera import Camera, get_rot_mat_x

radius = 420  # Distance to the center of the unit.

cameras = {
    'front': Camera(get_rot_mat_x(0), torch.tensor([0, 0, -radius], dtype=torch.float32)),
    'top': Camera(get_rot_mat_x(270), torch.tensor([0, -radius, 0], dtype=torch.float32)),
    'back': Camera(get_rot_mat_x(180), torch.tensor([0, 0, radius], dtype=torch.float32)),
    'bottom': Camera(get_rot_mat_x(90), torch.tensor([0, radius, 0], dtype=torch.float32)),
}

scene_dir = 'scenes/scene_1'

image_dir = os.path.join(scene_dir, 'images')
image_names = ['front', 'top', 'back', 'bottom']
images = {}
for name in image_names:
    image_path = os.path.join(image_dir, f'{name}.png')
    image = cv.imread(image_path)
    if image is None:
        print(f'Could not read image {image_path}')
        continue

    images[name] = image

mask_dir = os.path.join(scene_dir, 'masks')
mask_names = image_names
masks = {}

for name in mask_names:
    mask_path = os.path.join(mask_dir, f'{name}.png')
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    _, mask_binary = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    mask_binary = mask_binary / 255

    if mask is None:
        print(f'Could not read mask {mask_path}')
        continue

    masks[name] = mask_binary
