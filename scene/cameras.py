#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, fov2focal
from utils.viewer_utils import projection_from_intrinsics

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, bg, image_width, image, image_height, image_path,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 timestep=None, data_device = "cuda",
                 cx=None, cy=None, fx=None, fy=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep
        self.cx = cx if cx is not None else image_width / 2
        self.cy = cy if cy is not None else image_height / 2
        self.fx = fx if fx is not None else fov2focal(FoVx, image_width)
        self.fy = fy if fy is not None else fov2focal(FoVy, image_height)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)

        K = torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])
        self.projection_matrix = projection_from_intrinsics(
            K[None],
            (self.image_height, self.image_width),
            self.znear,
            self.zfar,
        )[0]
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep,
                 cx=None, cy=None, fx=None, fy=None):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep
        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2
        self.fx = fx if fx is not None else fov2focal(fovx, width)
        self.fy = fy if fy is not None else fov2focal(fovy, height)

