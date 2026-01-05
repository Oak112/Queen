"""
Scene loading and camera construction for QUEEN.

This file is based on the upstream QUEEN implementation and lightly modified
to ensure that, for COLMAP-based scenes, the train/test camera split respects
`ModelParams.test_indices` (so it aligns with `MultiViewVideoDataset`).
"""

import os
import json
import torch
from typing import List

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.cameras import SequentialCamera
from arguments import ModelParams
from utils.camera_utils import sequentialCameraList_from_camInfos, camera_to_JSON, updateCam


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        train_image_data=None,
        test_image_data=None,
        load_iteration=None,
        shuffle: bool = True,
        resolution_scales=[1.0],
        N_video_views: int = 300,
        verbose: bool = False,
    ):
        """Initialize a Scene from a dataset path.

        :param args: Model parameters (including source_path, test_indices, etc.).
        :param gaussians: GaussianModel instance to be associated with the scene.
        """

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            if verbose:
                print(
                    "Loading trained model at iteration {}".format(
                        self.loaded_iter
                    )
                )

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        # --- Scene type detection & loading ---------------------------------
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # MODIFIED: pass `test_indices` into the Colmap reader so that the
            # resulting train/test camera lists match the MultiViewVideoDataset
            # split used in train.py.
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.eval,
                test_indices=args.test_indices,
            )
            dataset_type = "colmap"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["Dynerf"](
                args.source_path, args.test_indices, N_video_views
            )
            dataset_type = "dynerf"
        elif os.path.exists(os.path.join(args.source_path, "train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["Panoptic"](args.source_path)
            dataset_type = "panoptic"
        elif os.path.exists(os.path.join(args.source_path, "models.json")):
            scene_info = sceneLoadTypeCallbacks["Immersive"](
                args.source_path, args.test_indices, args.world_scale
            )
            dataset_type = "immersive"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if verbose:
                print(
                    "Found transforms_train.json file, assuming Blender data set!"
                )
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
            dataset_type = "blender"
        else:
            assert False, "Could not recognize scene type!"

        if verbose:
            print(
                f"DEBUG: Scene::__init__(): parsed scene_info as type '{dataset_type}'"
            )

        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # --- Build camera lists ---------------------------------------------
        for resolution_scale in resolution_scales:
            if verbose:
                print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = sequentialCameraList_from_camInfos(
                scene_info.train_cameras,
                resolution_scale,
                args,
                train_image_data,
            )
            if verbose:
                print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = sequentialCameraList_from_camInfos(
                scene_info.test_cameras,
                resolution_scale,
                args,
                test_image_data,
            )
            if verbose:
                print("Loading Video Cameras")
            self.video_cameras[resolution_scale] = sequentialCameraList_from_camInfos(
                scene_info.video_cameras,
                resolution_scale,
                args,
            )

        # --- Optionally export cameras to JSON ------------------------------
        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.video_cameras:
                camlist.extend(scene_info.video_cameras)
            for cam_id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(cam_id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        # --- Initialize Gaussian attributes ---------------------------------
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            # initialize Gaussian attributes from a point cloud parsed in scene_info
            # usually does not carry full Gaussian attributes, but only xyz and
            # base rgb (as SH dc)
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, self.cameras_extent, ignore_colors=False
            )
            if verbose:
                print(
                    "Scene::__init__(): gaussians initialized by create_from_pcd()"
                )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def save(self, iteration, mask=None, save_point_cloud: bool = False):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask)
        if save_point_cloud:
            # save a "canonical one" for the frame
            self.gaussians.save_ply(
                os.path.join(self.model_path, "point_cloud.ply"), mask
            )

    def save_compressed(self, iteration, latent_args=None):
        if iteration == -1:
            point_cloud_path = os.path.join(self.model_path, "compressed")
        else:
            point_cloud_path = os.path.join(
                self.model_path, f"point_cloud/iteration_{iteration}"
            )
        self.gaussians.save_compressed_pkl(
            os.path.join(point_cloud_path, "point_cloud.pkl"), latent_args
        )

    def save_flow(self, iteration, camera, mask=None):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply_flow(
            os.path.join(point_cloud_path, "point_cloud.ply"), camera, mask
        )

    def updateCameraImages(
        self,
        args,
        train_image_data,
        test_image_data,
        frame_idx,
        resolution_scales=1.0,
    ):
        for resolution_scale in resolution_scales:
            for idx, camera in enumerate(self.train_cameras[resolution_scale]):
                image = train_image_data["image"][idx]
                updateCam(
                    args,
                    image,
                    train_image_data["path"][idx],
                    frame_idx,
                    camera,
                    resolution_scale,
                )
            for idx, camera in enumerate(self.test_cameras[resolution_scale]):
                image = test_image_data["image"][idx]
                updateCam(
                    args,
                    image,
                    test_image_data["path"][idx],
                    frame_idx,
                    camera,
                    resolution_scale,
                )

    def resetCameraCache(self):
        for camera in self.getTrainCameras():
            camera.colors_precomp = torch.zeros(self.gaussians._xyz.shape[0], 3).to(
                self.gaussians._features_dc
            )

    def getTrainCameras(self, scale: float = 1.0) -> List[SequentialCamera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale: float = 1.0) -> List[SequentialCamera]:
        return self.test_cameras[scale]

    def getVideoCameras(self, scale: float = 1.0) -> List[SequentialCamera]:
        return self.video_cameras[scale]
