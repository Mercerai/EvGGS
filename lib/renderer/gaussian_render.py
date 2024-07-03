import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(cam, idx, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=pts_xyz.device)
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=pts_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(cam['FovX'][idx] * 0.5)
    tanfovy = math.tan(cam['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam['H'][idx]),
        image_width=int(cam['W'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam['world_view_transform'][idx],
        projmatrix=cam['full_proj_transform'][idx],
        sh_degree=3,
        campos=cam['camera_center'][idx],
        prefiltered=False,
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, _ = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)
    # print("render image shape : ", rendered_image.shape)

    return rendered_image

# def pts2render(cam, pcd, gs_scaling, gs_opacity, gs_rotation, bg_color):
#     bs = pcd.shape[0]
#     # print(gs_data)
#     render_novel_list = []
#     for i in range(bs):
#         xyz_i = pcd[i, :3, :].permute(1, 0)
#         rgb_i = pcd[i, 3:6, :].permute(1, 0)
#         scale_i = gs_scaling[i].permute(1, 0)
#         opacity_i = gs_opacity[i].permute(1, 0)
#         rot_i = gs_rotation[i].permute(1, 0)
#         render_novel_i = render(cam, i, xyz_i, rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
#         render_novel_list.append(render_novel_i)

#     return torch.stack(render_novel_list, dim=0)

def pts2render(data, bg_color):
    bs = data['lview']['img'].shape[0]
    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        for view in ['lview', 'rview']:
            valid_i = data[view]['pts_valid'][i, :].bool() 
            xyz_i = data[view]['pts'][i, :, :]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            rot_i = data[view]['rot'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[view]['scale'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = data[view]['opacity'][i, :, :, :].permute(1, 2, 0).view(-1, 1)

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 1))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0).repeat((1,3))
        # pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        render_novel_i = render(data["target"], i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))

    return torch.concat(render_novel_list, dim=0)
