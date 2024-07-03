import torch
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def preprocess_render(batch):
    H, W = batch['H'][0], batch['W'][0]
    extrs = batch["cam_extrinsics"]
    intrs = batch["cam_intrinsics"]
    # znear, zfar = batch["znear"], batch["zfar"]
    znear, zfar = 0.5, 10000
    B = extrs.shape[0]  # `
    
    proj_mat = [getProjectionMatrix(znear, zfar, intrs[i], H, W).transpose(0, 1) for i in range(B)]
    world_view_transform = [
        getWorld2View2(extrs[i][:3, :3].reshape(3, 3).transpose(1, 0), extrs[i][:3, 3]).transpose(0, 1) for i in
        range(B)]
    proj_mat = torch.stack(proj_mat, dim=0)  # [4,4]
    # print("proj mat = ", proj_mat)

    world_view_transform = torch.stack(world_view_transform, dim=0)  # [4,4]
   
    full_proj_transform = (world_view_transform.bmm(proj_mat))
    camera_center = world_view_transform.inverse()[:, 3, :3]

    FovX = [torch.FloatTensor([focal2fov(intrs[i][0, 0], W)]) for i in range(B)]

    # print("111",FovX[0])
    FovY = [torch.FloatTensor([focal2fov(intrs[i][1, 1], H)]) for i in range(B)]

    return {"projection_matrix": proj_mat,
            "world_view_transform": world_view_transform,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "H": torch.ones(B) * H,
            "W": torch.ones(B) * W,
            "FovX": torch.stack(FovX, dim=0),
            "FovY": torch.stack(FovY, dim=0)
            }

def depth2pc(depth, extrinsic, intrinsic):
    B, C, H, W = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, H-0.5, H, device=depth.device), torch.linspace(0.5, W-0.5, W, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)