import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import math
import torch
import numpy as np
from tqdm import tqdm

from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss, ssim, binary_cross_entropy
from utils.general_utils import safe_state
from utils.color_utils import *
from gaussian_renderer import *

from utils.image_utils import psnr
from lpipsPyTorch import LPIPS
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    
    scene = Scene(dataset, gaussians, resolution_scales=[1.0])
    
    gaussians.training_setup(opt)
    lpips_fn = LPIPS(net_type='vgg').cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    print('Training set length', len(viewpoint_stack))
        
    ema_loss_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.apply_stage_schedule(iteration, opt.env_start_iter, opt.joint_training_start)
        if iteration % 1000 == 0 and (iteration <= opt.env_start_iter or iteration > opt.joint_training_start):
            gaussians.oneupSHdegree()

        data_idx = np.random.randint(len(viewpoint_stack))
        
        viewpoint_cam = viewpoint_stack[data_idx]
        
        bg = torch.rand((3), device="cuda")
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, opt.env_start_iter, iteration=iteration, track_env_usage=True)
        
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        gt_rgba = viewpoint_cam.original_image.cuda()
        gt_rgb_linear = srgb2linear(gt_rgba[:3,...])
        gt_image = gt_rgb_linear * gt_rgba[3:,...] + (1-gt_rgba[3:,...]) * bg[:, None, None]
            
        loss = 0.0
        
        pbr_rgb_linear = render_pkg["pbr_rgb_linear"] * render_pkg["rend_alpha"] + (1-render_pkg["rend_alpha"]) * bg[:, None, None]
        Ll1 = l1_loss(pbr_rgb_linear, gt_image)
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pbr_rgb_linear, gt_image))
        loss += loss_pbr

        if iteration < 3000 and not opt.no_alpha_loss:
            gt_mask = viewpoint_cam.original_image.cuda()[3:,...]
            alpha_loss = binary_cross_entropy(render_pkg["rend_alpha"], gt_mask)
            loss += alpha_loss

        lambda_normal = opt.lambda_normal if (iteration > 0 and not opt.no_normal_loss) else 0.0

        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        loss = loss + normal_loss

        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if (iteration <= opt.env_start_iter or iteration > opt.joint_training_start) and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
            if (iteration in testing_iterations):
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, bg, iteration), opt, lpips_fn)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                
def prepare_output_and_logger(args):
    dataset_name = args.source_path.split('/')[-1]
    if not args.model_path:
        args.model_path = os.path.join("./output/refnerf/", dataset_name)
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def normal_angular_error_metrics(rend_normal, gt_normal, rend_alpha, alpha_thresh=0.5, gt_norm_thresh=0.5):
    valid = (rend_alpha.squeeze(0) > alpha_thresh) & (gt_normal.norm(dim=0) > gt_norm_thresh)
    if not valid.any():
        return None, None
    dot = torch.clamp((rend_normal * gt_normal).sum(dim=0), -1.0, 1.0)
    angle_deg = torch.acos(dot) * (180.0 / math.pi)
    angles = angle_deg[valid]
    return angles.mean().item(), angles.median().item()


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, opt, lpips_fn):
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        env_res = render_env_map(scene.gaussians)
        for env_name in env_res.keys():
            if tb_writer:
                tb_writer.add_image("#envmap/{}".format(env_name), env_res[env_name], global_step=iteration)
            torchvision.utils.save_image(env_res[env_name], f"{scene.model_path}/envmap_{env_name}_{iteration:06d}.png")

        validation_configs = ({'name': 'test',  'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = psnr_test = ssim_test = lpips_test = 0.0
                sum_mae = sum_median_ae = 0.0
                count_normal = 0
                save_path = f"{scene.model_path}/iteration-{iteration}/"
                os.makedirs(save_path, exist_ok=True)
                for cam_idx, viewpoint in enumerate(config['cameras']):
                    bg_eval = torch.rand((3), device="cuda")
                    render_pkg = renderFunc(viewpoint, scene.gaussians, renderArgs[0], bg_eval, opt.env_start_iter, cam_idx, iteration=renderArgs[2], save_dir=save_path, track_env_usage=False)

                    image_linear = render_pkg["pbr_rgb_linear"] * render_pkg["rend_alpha"] + (1-render_pkg["rend_alpha"]) * bg_eval[:, None, None]
                    image_linear = torch.clamp(image_linear, 0.0, 1.0)

                    gt_rgba = viewpoint.original_image.cuda()
                    gt_rgb_linear = srgb2linear(gt_rgba[:3,...])
                    gt_image_linear = gt_rgb_linear * gt_rgba[3:,...] + (1-gt_rgba[3:,...]) * bg_eval[:, None, None]
                    gt_image_linear = torch.clamp(gt_image_linear, 0.0, 1.0)

                    image_srgb = torch.clamp(linear2srgb(image_linear), 0.0, 1.0)
                    gt_srgb    = torch.clamp(linear2srgb(gt_image_linear), 0.0, 1.0)

                    l1_test    += l1_loss(image_srgb, gt_srgb).mean().double()
                    psnr_test  += psnr(image_srgb, gt_srgb).mean().double()
                    ssim_test  += ssim(image_srgb[None], gt_srgb[None]).mean().double()
                    lpips_test += lpips_fn(image_srgb[None], gt_srgb[None]).mean().double()

                    gt_normal_path = os.path.join(scene.source_path, config['name'], viewpoint.image_name + "_normal.npy")
                    if os.path.exists(gt_normal_path):
                        import torch.nn.functional as F_
                        gt_np = np.load(gt_normal_path)
                        gt_n = torch.from_numpy(gt_np).float().cuda().permute(2, 0, 1)
                        H, W = render_pkg["rend_normal"].shape[1], render_pkg["rend_normal"].shape[2]
                        if gt_n.shape[1] != H or gt_n.shape[2] != W:
                            gt_n = F_.interpolate(gt_n.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                        gt_n = F_.normalize(gt_n, dim=0)
                        mae, med_ae = normal_angular_error_metrics(render_pkg["rend_normal"], gt_n, render_pkg["rend_alpha"])
                        if mae is not None:
                            sum_mae += mae
                            sum_median_ae += med_ae
                            count_normal += 1

                n = len(config['cameras'])
                p, s, lp, l1 = psnr_test/n, ssim_test/n, lpips_test/n, l1_test/n
                log = f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1:.4f}  PSNR {p:.2f}  SSIM {s:.4f}  LPIPS {lp:.4f}"
                if count_normal > 0:
                    log += f"  normal_mae {sum_mae/count_normal:.4f}  normal_median_ae {sum_median_ae/count_normal:.4f}"
                print(log)

                if iteration == testing_iterations[-1] and config['name'] == 'test':
                    metrics_path = os.path.join(scene.model_path, 'metrics.txt')
                    with open(metrics_path, 'w') as f:
                        f.write(f'psnr={p:.4f} ssim={s:.4f} lpips={lp:.4f} l1={l1:.4f}\n')
                        if count_normal > 0:
                            f.write(f'normal_mae={sum_mae/count_normal:.4f} normal_median_ae={sum_median_ae/count_normal:.4f}\n')
                    print(f"Saved metrics → {metrics_path}")

        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=
                        [10_000, 20_000, 30_000, 37_000, 45_000]
                       )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=
                        [10_000, 20_000, 30_000, 37_000, 45_000]
                       )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)
    print("\nTraining complete.")
