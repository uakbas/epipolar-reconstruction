import torch
from camera import homogenize
from typing import List


def create_volume(limits: List[tuple], sampling_step):
    x_lim, y_lim, z_lim = limits
    x_line = torch.arange(*x_lim, sampling_step)
    y_line = torch.arange(*y_lim, sampling_step)
    z_line = torch.arange(*z_lim, sampling_step)
    return torch.stack(torch.meshgrid([x_line, y_line, z_line], indexing='ij'), dim=-1)


def get_volume_vote(volume, projection_mat, mask, depth_maps):
    x_len, y_len, z_len, THREE = volume.shape
    assert THREE == 3, "Volume must have shape of (x_len, y_len, z_len, 3)"

    depth_map_H, depth_map_W, TWO = depth_maps.shape
    assert TWO == 2, "Depth map must have shape of (H, W, 2)"
    depth_map_min = depth_maps[:, :, 0]
    depth_map_max = depth_maps[:, :, 1]

    mask_H, mask_W = mask.shape
    P = projection_mat

    n_points = x_len * y_len * z_len
    points_world = volume.view(n_points, THREE)

    points_projected = P @ homogenize(points_world.T)
    depths = points_projected[2]
    points_projected = (points_projected[:2] / depths).T
    points_projected = points_projected.to(torch.int)

    # Check if the projected points are inside the image
    is_visible = ((0 <= points_projected[:, 0]) & (points_projected[:, 0] < mask_W) &
                  (0 <= points_projected[:, 1]) & (points_projected[:, 1] < mask_H))
    idx_visible_points = torch.nonzero(is_visible).flatten()

    points_projected_visible = points_projected[is_visible]
    depths_visible = depths[is_visible]

    # Check if visible points are on the positive region of the mask
    is_inside_mask = mask[points_projected_visible[:, 1], points_projected_visible[:, 0]] == 1

    # Check if visible points are between max and min depths

    depths_max = depth_map_max[points_projected_visible[:, 1], points_projected_visible[:, 0]]
    depths_min = depth_map_min[points_projected_visible[:, 1], points_projected_visible[:, 0]]
    is_inside_depth_interval = (depths_min <= depths_visible) & (depths_visible <= depths_max)

    is_valid = is_inside_mask & is_inside_depth_interval

    idx_valid_points = idx_visible_points[is_valid]

    votes = torch.zeros(n_points)
    votes[idx_valid_points] = 1

    volume_votes = votes.view(x_len, y_len, z_len)
    return volume_votes
