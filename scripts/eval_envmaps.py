import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import LPIPS

_lpips_fn = LPIPS(net_type='vgg').cuda()


def _load(path):
    t = TF.to_tensor(Image.open(path).convert("RGB")).float().cuda().clamp(0.0, 1.0)
    return t


def _other_to_gt(gt, other):
    if other.shape[1:] != gt.shape[1:]:
        other = TF.resize(other, [gt.shape[1], gt.shape[2]], antialias=True)
    return other.clamp(0.0, 1.0)


def get_psnr(gt_path, other_path):
    gt = _load(gt_path)
    other = _other_to_gt(gt, _load(other_path))
    with torch.no_grad():
        return psnr(other[None], gt[None]).mean().item()


def get_lpips(gt_path, other_path):
    gt = _load(gt_path)
    other = _other_to_gt(gt, _load(other_path))
    with torch.no_grad():
        return _lpips_fn(other[None], gt[None]).item()


def get_l1(gt_path, other_path):
    gt = _load(gt_path)
    other = _other_to_gt(gt, _load(other_path))
    with torch.no_grad():
        return l1_loss(other, gt).item()


def get_ssim(gt_path, other_path):
    gt = _load(gt_path)
    other = _other_to_gt(gt, _load(other_path))
    with torch.no_grad():
        return ssim(other[None], gt[None]).item()


def main():
    import json
    gt_dir   = os.path.join(_root, "envs")
    data_dir = os.path.join(_root, "local", "data")
    scenes = ("vase", "cola", "mirror")
    out = {}
    for method in sorted(os.listdir(data_dir)):
        subdir = os.path.join(data_dir, method)
        if not os.path.isdir(subdir):
            continue
        out[method] = {}
        for scene in scenes:
            gt_path    = os.path.join(gt_dir, f"{scene}.png")
            other_path = os.path.join(subdir, f"{scene}.png")
            if not os.path.isfile(gt_path) or not os.path.isfile(other_path):
                continue
            out[method][scene] = {
                "psnr":  round(get_psnr(gt_path, other_path), 4),
                "lpips": round(get_lpips(gt_path, other_path), 4),
                "l1":    round(get_l1(gt_path, other_path), 6),
                "ssim":  round(get_ssim(gt_path, other_path), 4),
            }
    out_path = os.path.join(_root, "local", "comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


def main_ablation():
    import json
    gt_path      = os.path.join(_root, "envs", "vase.png")
    ablation_dir = os.path.join(_root, "local", "ablation")
    if not os.path.isfile(gt_path):
        return
    out = {}
    for name in sorted(os.listdir(ablation_dir)):
        if not name.endswith(".png"):
            continue
        other_path = os.path.join(ablation_dir, name)
        key = os.path.splitext(name)[0]
        out[key] = {
            "psnr":  round(get_psnr(gt_path, other_path), 4),
            "lpips": round(get_lpips(gt_path, other_path), 4),
            "l1":    round(get_l1(gt_path, other_path), 6),
            "ssim":  round(get_ssim(gt_path, other_path), 4),
        }
    out_path = os.path.join(_root, "local", "ablations.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


def main_quick(gt_path, pred_path):
    print(f"PSNR:  {get_psnr(gt_path, pred_path):.4f}")
    print(f"SSIM:  {get_ssim(gt_path, pred_path):.4f}")
    print(f"LPIPS: {get_lpips(gt_path, pred_path):.4f}")
    print(f"L1:    {get_l1(gt_path, pred_path):.6f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--gt",   type=str, default=None)
    p.add_argument("--pred", type=str, default=None)
    args = p.parse_args()
    if args.gt and args.pred:
        main_quick(args.gt, args.pred)
    else:
        if args.full or not args.ablation:
            main()
        if args.ablation or not args.full:
            main_ablation()
