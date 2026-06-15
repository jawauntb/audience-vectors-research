"""
VideoQualityScorer — quality score via R3D-18 similarity.

Uses torchvision.models.video.r3d_18 (already installed with torchvision) as
a video feature extractor (512-dim). The quality score is the cosine
similarity between the feature vector of the evaluated video and the
centroid of the reference distribution (neutral videos, alpha=0).

High score → video visually close to the baseline without steering.
Low score → steering pushed the video out of distribution.

Why R3D-18 instead of classic FVD?
FVD requires batches of >= 16 videos to estimate the distribution.
In the BO loop we evaluate one video at a time. The cosine distance to
the reference centroid is a stable approximation, computationally
cheap and compatible with online scoring.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoQualityScorer:
    """
    Video quality scorer based on cosine similarity with R3D-18.

    Extracts 512-dimensional features with the pretrained R3D-18 backbone
    and computes the cosine similarity between the evaluated video and the
    centroid of the neutral reference videos (alpha=0).
    """

    def __init__(self, device: str = "cuda") -> None:
        """
        Initializes the R3D-18 model and prepares it for feature extraction.

        Args:
            device: PyTorch device ('cuda' or 'cpu').
        """
        self.device = device

        # Load R3D-18 with pretrained weights
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Remove the final fully-connected layer — keep only the feature extractor (512-dim)
        model.fc = nn.Identity()
        model = model.to(device)
        model.eval()
        self._model = model

        self._ref_mean: np.ndarray | None = None

    def _load_video_tensor(self, video_path: Path, n_frames: int = 16) -> torch.Tensor:
        """
        Loads a video and returns a tensor in the format expected by R3D-18.

        Samples n_frames equally spaced frames, resizes to
        112×112 with PIL and applies the standard Kinetics-400 normalization.

        Args:
            video_path: Path to the video file (.mp4).
            n_frames:   Number of frames to sample (default: 16).

        Returns:
            Tensor of shape (1, C, T, H, W) in float32, normalized.
        """
        from PIL import Image

        frames: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            total = stream.frames or 25
            # Compute equally spaced sampling indices
            indices = set(
                int(round(i * (total - 1) / max(n_frames - 1, 1)))
                for i in range(n_frames)
            )
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    img = frame.to_image().convert("RGB")
                    img = img.resize((112, 112), Image.BILINEAR)
                    frames.append(np.array(img))
                if len(frames) >= n_frames:
                    break

        # Pad with the last frame if there are fewer than n_frames
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames else np.zeros((112, 112, 3), dtype=np.uint8))

        # (T, H, W, C) → (C, T, H, W) → (1, C, T, H, W)
        arr = np.stack(frames[:n_frames])  # (T, H, W, C)
        t = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)

        # Kinetics-400 normalization (R3D-18 standard)
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        t = (t - mean) / std

        return t.unsqueeze(0)  # (1, C, T, H, W)

    def _extract_features(self, video_path: Path) -> np.ndarray:
        """
        Extracts an L2-normalized feature vector (512,) for a video.

        Args:
            video_path: Path to the video file.

        Returns:
            Numpy array of shape (512,), L2-normalized.
        """
        tensor = self._load_video_tensor(video_path).to(self.device)
        with torch.no_grad():
            feats = self._model(tensor)  # (1, 512)
        feat = feats.squeeze(0).cpu().numpy()  # (512,)
        # L2 normalization
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def fit_reference(self, video_paths: list[Path]) -> None:
        """
        Fits the reference distribution with neutral videos (alpha=0).

        Extracts features for all videos, computes the centroid and
        L2-normalizes it for use in cosine similarity.

        Args:
            video_paths: List of paths to neutral reference videos.
        """
        feats = np.stack([self._extract_features(p) for p in video_paths])
        ref_mean = feats.mean(axis=0)
        norm = np.linalg.norm(ref_mean)
        if norm > 0:
            ref_mean = ref_mean / norm
        self._ref_mean = ref_mean

    def score(self, video_path: Path) -> float:
        """
        Returns the quality score for a video.

        The score is the cosine similarity between the video features and the
        reference centroid. Values close to 1.0 indicate that the
        video is visually similar to the baseline without steering.

        Args:
            video_path: Path to the video to evaluate.

        Returns:
            Score in [-1, 1]. Returns 0.0 if the scorer has not been fitted.
        """
        if self._ref_mean is None:
            return 0.0
        feat = self._extract_features(video_path)
        return float(np.dot(feat, self._ref_mean))

    def save(self, path: Path) -> None:
        """
        Saves the reference centroid to disk (.npz).

        Args:
            path: Destination path for the .npz file.
        """
        np.savez(str(path), ref_mean=self._ref_mean)

    def load(self, path: Path) -> None:
        """
        Loads the reference centroid from disk.

        Args:
            path: Path to the .npz file saved by `save()`.
        """
        data = np.load(str(path))
        self._ref_mean = data["ref_mean"]

    @property
    def is_fitted(self) -> bool:
        """Returns True if the scorer has been fitted with reference videos."""
        return self._ref_mean is not None
