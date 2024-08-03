import torch
import numpy as np
import matplotlib
import cv2 as cv
from depth.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'depth/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


def predict_depth(image):
    return model.infer_image(image)


def show_depth(depth, grayscale=False):
    depth_ = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_ = depth_.astype(np.uint8)

    if grayscale:
        depth_ = np.repeat(depth_[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth_ = (cmap(depth_)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    cv.imshow('Depth Map', depth_)
    cv.waitKey(0)
