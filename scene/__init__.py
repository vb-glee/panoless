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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


def _perturb_pose(R, T, sigma_deg, scene_radius, rng):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-8
    angle_rad = np.deg2rad(sigma_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    dR = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
    R_new = R @ dR.T
    cam_center = -(R @ T)
    cam_center_new = cam_center + rng.standard_normal(3) * (0.002 * sigma_deg * scene_radius)
    T_new = -(R_new.T @ cam_center_new)
    return R_new.astype(np.float32), T_new.astype(np.float32)


def _apply_pose_jitter(scene_info, sigma_deg, scene_radius, seed=42):
    rng = np.random.default_rng(seed)
    perturbed = []
    for cam in scene_info.train_cameras:
        R_new, T_new = _perturb_pose(cam.R, cam.T, sigma_deg, scene_radius, rng)
        perturbed.append(cam._replace(R=R_new, T=T_new))
    return scene_info._replace(train_cameras=perturbed)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            raise ValueError("Could not recognize scene type!")

        sigma_deg = getattr(args, 'pose_jitter_deg', 0.0)
        if sigma_deg > 0.0:
            seed = getattr(args, 'pose_jitter_seed', 42)
            radius = scene_info.nerf_normalization['radius']
            print(f"[pose_jitter] σ={sigma_deg}° on {len(scene_info.train_cameras)} train cameras (seed={seed})")
            scene_info = _apply_pose_jitter(scene_info, sigma_deg, radius, seed=seed)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)   # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print('resolution_scales', resolution_scales)
        
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(
                self.model_path,
               "point_cloud",
               "iteration_" + str(self.loaded_iter),
               "point_cloud.ply"
            ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]