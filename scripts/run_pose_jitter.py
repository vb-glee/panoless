#!/usr/bin/env python3
import os
import sys
import csv
import glob
import subprocess

import torch
import torchvision.transforms.functional as TF
from PIL import Image

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import LPIPS

SIGMAS      = [0.0, 0.5, 1.0, 2.0]
SCENE_PATH  = "custom/vase_t45-120_p0-60_plus_center"
GT_ENVMAP   = os.path.join(_root, "envs", "vase.png")
OUT_DIR     = os.path.join(_root, "outputs", "pose_jitter")
SEED        = 42
TRAIN_ITERS = 45_000
ENVMAP_COORD = "env_cood2"


def load_img(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).float().cuda().clamp(0.0, 1.0)


def compute_metrics(pred_path, gt_path, lpips_fn):
    gt   = load_img(gt_path)
    pred = load_img(pred_path)
    if pred.shape[1:] != gt.shape[1:]:
        pred = TF.resize(pred, [gt.shape[1], gt.shape[2]], antialias=True).clamp(0.0, 1.0)
    with torch.no_grad():
        return (
            psnr(pred[None], gt[None]).mean().item(),
            ssim(pred[None], gt[None]).item(),
            lpips_fn(pred[None], gt[None]).item(),
            l1_loss(pred, gt).item(),
        )


def find_envmap(model_path, coord, iteration):
    exact = os.path.join(model_path, f"envmap_{coord}_{iteration:06d}.png")
    if os.path.isfile(exact):
        return exact
    candidates = sorted(glob.glob(os.path.join(model_path, f"envmap_{coord}_*.png")))
    return candidates[-1] if candidates else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lpips_fn = LPIPS(net_type="vgg").cuda()

    rows = []
    for sigma in SIGMAS:
        model_path = os.path.join(OUT_DIR, f"vase_sigma_{sigma:.1f}")
        cmd = [
            sys.executable, os.path.join(_root, "train.py"),
            "-s", SCENE_PATH,
            "--model_path", model_path,
            "--pose_jitter_deg", str(sigma),
            "--pose_jitter_seed", str(SEED),
            "--iterations", str(TRAIN_ITERS),
            "--eval",
            "--albedo_bias", "0",
        ]
        print(f"\n{'='*70}\n[pose_jitter] σ={sigma}°  →  {model_path}\n{'='*70}")
        subprocess.run(cmd, check=True)

        pred_path = find_envmap(model_path, ENVMAP_COORD, TRAIN_ITERS)
        if pred_path is None:
            print(f"  WARNING: no {ENVMAP_COORD} envmap found in {model_path}")
            rows.append((sigma, float("nan"), float("nan"), float("nan"), float("nan")))
            continue

        ep, es, el, el1 = compute_metrics(pred_path, GT_ENVMAP, lpips_fn)
        print(f"  PSNR={ep:.4f}  SSIM={es:.4f}  LPIPS={el:.4f}  L1={el1:.6f}")
        rows.append((sigma, ep, es, el, el1))

    csv_path = os.path.join(OUT_DIR, "vase.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma_deg", "env_PSNR", "env_SSIM", "env_LPIPS", "env_L1"])
        for sigma, p, s, lp, l1 in rows:
            writer.writerow([f"{sigma:.1f}", f"{p:.4f}", f"{s:.4f}", f"{lp:.4f}", f"{l1:.6f}"])

    print(f"\nResults → {csv_path}")
    print("\nsigma_deg | env_PSNR | env_SSIM | env_LPIPS | env_L1")
    print("-" * 57)
    for sigma, p, s, lp, l1 in rows:
        print(f"  {sigma:4.1f}    |  {p:7.4f} |  {s:7.4f} |   {lp:7.4f} | {l1:.6f}")


if __name__ == "__main__":
    main()
