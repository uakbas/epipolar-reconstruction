import os.path
import cv2 as cv
import torch
from camera import Camera, getRotMatX

radius = 420  # Distance to the center of the unit.

cameras = {
    'front': Camera(getRotMatX(0), torch.tensor([0, 0, -radius], dtype=torch.float32)),
    'top': Camera(getRotMatX(270), torch.tensor([0, -radius, 0], dtype=torch.float32)),
    'back': Camera(getRotMatX(180), torch.tensor([0, 0, radius], dtype=torch.float32)),
    'bottom': Camera(getRotMatX(90), torch.tensor([0, radius, 0], dtype=torch.float32)),
}

image_dir = 'images'
image_names = ['front', 'top', 'back', 'bottom']
images = {}
for name in image_names:
    image_path = os.path.join(image_dir, f'{name}.png')
    image = cv.imread(image_path)
    if image is None:
        print(f'Could not read image {image_path}')
        continue

    images[name] = image
