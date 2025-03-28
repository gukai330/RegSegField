from nerfstudio.utils.poses import inverse
import torch

def unpack_4x4_transform(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

def is_in_frustum(camera, points_3d, ):
    c2w = camera.camera_to_worlds
    w2c = inverse(c2w)
    rotation, translation = unpack_4x4_transform(w2c)
    points_cam = points_3d @ rotation.T + translation

    fx = camera.fx
    fy = camera.fy
    cx = camera.cx
    cy = camera.cy

    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = -points_cam[..., 2]

    pixel_i = fx * ( x / ( z + 1e-12)) + cx
    pixel_j = -fy * ( y / ( z + 1e-12)) + cy

    pixel_i = torch.round(pixel_i).long()
    pixel_j = torch.round(pixel_j).long()

    is_in = (pixel_i >= 0) & (pixel_i < camera.width) & (pixel_j >= 0) & (pixel_j < camera.height)

    return is_in

def is_in_mask(camera, points_3d, mask):
    c2w = camera.camera_to_worlds
    w2c = inverse(c2w)
    rotation, translation = unpack_4x4_transform(w2c)
    points_cam = points_3d @ rotation.T + translation

    fx = camera.fx
    fy = camera.fy
    cx = camera.cx
    cy = camera.cy

    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = -points_cam[..., 2]

    pixel_i = fx * ( x / ( z + 1e-12)) + cx
    pixel_j = -fy * ( y / ( z + 1e-12)) + cy

    pixel_i = torch.round(pixel_i).long()
    pixel_j = torch.round(pixel_j).long()

    is_in_frustum = (pixel_i >= 0) & (pixel_i < camera.width) & (pixel_j >= 0) & (pixel_j < camera.height)

    res = torch.zeros_like(pixel_i).bool()

    res[is_in_frustum] = mask[pixel_j[is_in_frustum], pixel_i[is_in_frustum]]

    return res