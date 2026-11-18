"""LoFTR correspondences with the source project's MAGSAC geometry filter."""

from pathlib import Path
from threading import Lock

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from src.v1.Functions import summarize_matches


def load_torch_image(image, max_side, device):
    """Resize RGB input without the original fixed-aspect-ratio distortion."""
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_width = max(8, int(width * scale) // 8 * 8)
    new_height = max(8, int(height * scale) // 8 * 8)
    image = image.resize((new_width, new_height))
    pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
    return K.color.rgb_to_grayscale(tensor)


def FormatingOutputs(correspondences):
    """Return a flat inlier mask, including safe zero/insufficient-match cases."""
    points0 = correspondences["keypoints0"].detach().cpu().numpy()
    points1 = correspondences["keypoints1"].detach().cpu().numpy()
    inliers = np.zeros(len(points0), dtype=bool)
    geometry_estimated = False
    if len(points0) >= 8:
        try:
            matrix, mask = cv2.findFundamentalMat(
                points0, points1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
            )
        except cv2.error:
            matrix, mask = None, None
        if matrix is not None and mask is not None and len(mask) == len(points0):
            inliers = mask.ravel() > 0
            geometry_estimated = bool(np.isfinite(matrix).all())
            if not geometry_estimated:
                inliers[:] = False
    return inliers, geometry_estimated


class SimilarityService:
    def __init__(self, settings):
        self.settings = settings
        self.matcher = None
        self.device = None
        self._lock = Lock()

    @property
    def loaded(self):
        return self.matcher is not None

    def _load_model(self):
        if self.matcher is not None:
            return
        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LOFTR_DEVICE=cuda was requested but CUDA is unavailable.")
        if self.settings.weights_path:
            path = Path(self.settings.weights_path).expanduser()
            if not path.is_file():
                raise RuntimeError("LOFTR_WEIGHTS must point to an existing LoFTR checkpoint.")
            matcher = KF.LoFTR(pretrained=None)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint)
            matcher.load_state_dict(state_dict)
        else:
            # Kornia downloads the chosen upstream checkpoint on the first comparison.
            matcher = KF.LoFTR(pretrained=self.settings.pretrained)
        self.matcher = matcher.eval().to(device)
        self.device = device

    def compare(self, image1, image2):
        # One model instance per process, with serialized initialization and inference.
        with self._lock:
            self._load_model()
            inputs = {
                "image0": load_torch_image(image1, self.settings.max_image_side, self.device),
                "image1": load_torch_image(image2, self.settings.max_image_side, self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(inputs)
            inliers, geometry_estimated = FormatingOutputs(correspondences)
            confidence = correspondences["confidence"].detach().cpu().tolist()
            result = summarize_matches(confidence, inliers)
            result["GeometryEstimated"] = geometry_estimated
            return result
