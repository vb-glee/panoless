import os
import math
import numpy as np

import torch
import torchvision
import torch.nn.functional as F

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from scene.gaussian_model import GaussianModel

from utils.point_utils import depth_to_normal

from utils.color_utils import *

from utils.general_utils import get_env_rayd1, get_env_rayd2

use_feature = True

def reflection(rayd, normal):
    refl = rayd - 2*normal*torch.sum(rayd*normal, dim=-1, keepdim=True)
    return refl

def sample_cubemap_color(rays_d, env_map):
    H,W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1,3)))
    outcolor = outcolor.reshape(H,W,3).permute(2,0,1)
    return outcolor

def render_env_map(pc: GaussianModel):
    env_cood1 = sample_cubemap_color(get_env_rayd1(512,1024), pc.envmap)
    env_cood2 = sample_cubemap_color(get_env_rayd2(512,1024), pc.envmap)
    
    env_res = {
        'env_cood1': env_cood1, 
        'env_cood2': env_cood2
    }
    
    if pc.envmap is not None:
        pc.envmap.ensure_usage_map()
        if pc.envmap.usage_map is not None:
            usage_counts = pc.envmap.usage_map.float()
            usage_vis = torch.log1p(usage_counts)
            if (usage_vis > 0).sum() > 0:
                p90 = torch.quantile(usage_vis[usage_vis > 0], 0.90)
                usage_vis = usage_vis / p90.clamp_min(1e-8)
            else:
                usage_vis = usage_vis / usage_vis.max().clamp_min(1e-8)
            usage_vis = F.avg_pool2d(usage_vis.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze()
            usage_vis = usage_vis.unsqueeze(0).repeat(3, 1, 1)
            env_res['env_usage'] = usage_vis
    
    return env_res

def get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS):
    return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, env_start_iter=None, cam_idx=0, scaling_modifier=1.0, iteration=0, save_dir=None, track_env_usage=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings_black = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=use_feature,
    )
    
    rasterizer_black = GaussianRasterizer(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    rets =  {}

    gs_albedo = pc.get_albedo
    gs_roughness = pc.get_roughness
    gs_feature = pc.get_language_feature
    
    input_ts = torch.cat([gs_roughness, gs_feature], dim=-1)
    
    albedo_map, out_ts, radii, allmap = rasterizer_black(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = gs_albedo,
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = F.normalize(render_normal, dim=0)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()
    
    viewdirs = viewpoint_camera.rays_d.to("cuda")
    normals_full = render_normal.permute(1,2,0)
    refl_dirs = reflection(viewdirs, normals_full)

    env_colors = torch.sigmoid(pc.get_envmap(refl_dirs.reshape(-1, 3)))
    env_colors = env_colors.reshape(image_height, image_width, 3)
    env_colors = srgb2linear(env_colors)

    out_ts = out_ts.permute(1,2,0)

    albedo_map = albedo_map.permute(1,2,0)

    with torch.no_grad():
        select_index = (render_alpha.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]

    roughness_sel = out_ts[..., :1].reshape(-1, 1)[select_index]
    alpha_sel     = render_alpha.reshape(-1, 1)[select_index]

    use_env = (env_start_iter is None) or (iteration >= env_start_iter)
    refl_sel = F.normalize(refl_dirs, dim=-1).reshape(-1, 3)[select_index].contiguous()

    if use_env and track_env_usage and hasattr(pc, "envmap") and (pc.envmap is not None) and pc.envmap.training:
        hit_w = (alpha_sel * (1.0 - roughness_sel).pow(2.0)).squeeze(1)
        hit_w = hit_w.clamp_(0, 1).detach()
        _ = pc.envmap(refl_sel, track_usage=hit_w)

    albedo_map = albedo_map.reshape(-1, 3)[select_index]

    env_color_sel = env_colors.reshape(-1, 3)[select_index]
    if getattr(pipe, 'rho_weighted_env', False):
        refl_strength = (1.0 - roughness_sel).pow(2.0).clamp(0, 1)
        spec_light = refl_strength * env_color_sel
    else:
        spec_light = env_color_sel
    diff_light = albedo_map
    
    pbr_rgb_linear = spec_light + diff_light
    pbr_rgb_linear = torch.clamp(pbr_rgb_linear, min=0., max=1.)

    pbr_rgb = linear2srgb(pbr_rgb_linear)
    pbr_rgb = torch.clamp(pbr_rgb, min=0., max=1.)
        
    output_rgb = torch.zeros(image_height, image_width, 3).cuda()
    output_rgb.reshape(-1, 3)[select_index] = pbr_rgb
    output_rgb = output_rgb.permute(2,0,1)

    output_rgb_linear = torch.zeros(image_height, image_width, 3).cuda()
    output_rgb_linear.reshape(-1, 3)[select_index] = pbr_rgb_linear
    output_rgb_linear = output_rgb_linear.permute(2,0,1)
    
    rets.update({
        'pbr_rgb': output_rgb,
        'pbr_rgb_linear': output_rgb_linear,

        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }) 
        
    if save_dir:
        with torch.no_grad():
            
            output_spec = torch.zeros(image_height, image_width, 3).cuda()
            output_spec.reshape(-1, 3)[select_index] = linear2srgb(spec_light)
            output_spec = output_spec.permute(2,0,1)
            
            output_diff = torch.zeros(image_height, image_width, 3).cuda()
            output_diff.reshape(-1, 3)[select_index] = linear2srgb(diff_light)
            output_diff = output_diff.permute(2,0,1)

            import os
            os.makedirs(f"{save_dir}render_alpha/", exist_ok=True)
            os.makedirs(f"{save_dir}render_normal/", exist_ok=True)
            os.makedirs(f"{save_dir}surf_depth/", exist_ok=True)
            os.makedirs(f"{save_dir}surf_normal/", exist_ok=True)
            os.makedirs(f"{save_dir}gt_image/", exist_ok=True)
            os.makedirs(f"{save_dir}feature_image/", exist_ok=True)
            os.makedirs(f"{save_dir}roughness/", exist_ok=True)
            os.makedirs(f"{save_dir}pbr_rgb/", exist_ok=True)
            os.makedirs(f"{save_dir}spec_light/", exist_ok=True)
            os.makedirs(f"{save_dir}diff_light/", exist_ok=True)
    
            torchvision.utils.save_image(render_alpha, f"{save_dir}/render_alpha/render_alpha_{cam_idx:03d}.png")
            torchvision.utils.save_image(((render_normal+1)/2)*render_alpha, f"{save_dir}/render_normal/render_normal_{cam_idx:03d}.png")
            
            surf_depth = surf_depth / surf_depth.max()
            torchvision.utils.save_image(surf_depth, f"{save_dir}surf_depth/surf_depth_{cam_idx:03d}.png")
            torchvision.utils.save_image((surf_normal+1)/2, f"{save_dir}surf_normal/surf_normal_{cam_idx:03d}.png")
            
            gt_image_copy = viewpoint_camera.original_image.cuda()
            torchvision.utils.save_image(gt_image_copy, f"{save_dir}gt_image/gt_image_{cam_idx:03d}.png")
            
            torchvision.utils.save_image(((out_ts[..., 1:].permute(2,0,1))[:3]+1)/2, f"{save_dir}feature_image/feature_image_{cam_idx:03d}.png")
            torchvision.utils.save_image(out_ts[..., :1].repeat(1,1,3).permute(2,0,1), f"{save_dir}roughness/roughness_{cam_idx:03d}.png")
            
            torchvision.utils.save_image(output_rgb*render_alpha, f"{save_dir}pbr_rgb/pbr_rgb_{cam_idx:03d}.png")
            torchvision.utils.save_image(output_spec, f"{save_dir}spec_light/spec_light_{cam_idx:03d}.png")
            torchvision.utils.save_image(output_diff, f"{save_dir}diff_light/diff_light_{cam_idx:03d}.png")

    return rets

