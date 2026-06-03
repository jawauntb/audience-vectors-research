"""CLIPScore evaluator — visual fidelity objective.

Computes mean cosine similarity between CLIP-ViT-H frame embeddings and the
conditioning prompt. Used as Objective 2 (visual fidelity) in the BO.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


class CLIPScorer:
    """Scores a video against a text prompt using CLIP-ViT-H/14."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def _encode_frames(self, frames: list[Image.Image]) -> Tensor:
        tensors = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        feats = self.model.encode_image(tensors)
        return F.normalize(feats, dim=-1)  # (T, 1024)

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> Tensor:
        tokens = self.tokenizer([prompt]).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)  # (1, 1024)

    def _extract_frames(self, video_path: Path, max_frames: int = 8) -> list[Image.Image]:
        frames: list[Image.Image] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            total = stream.frames or 25
            step = max(1, total // max_frames)
            for i, frame in enumerate(container.decode(video=0)):
                if i % step == 0:
                    frames.append(frame.to_image())
                if len(frames) >= max_frames:
                    break
        return frames

    def score(self, video_path: Path, prompt: str) -> float:
        """Return mean CLIP cosine similarity across sampled frames."""
        frames = self._extract_frames(video_path)
        if not frames:
            return 0.0
        frame_feats = self._encode_frames(frames)   # (T, 1024)
        text_feat = self._encode_text(prompt)        # (1, 1024)
        sims = (frame_feats @ text_feat.T).squeeze(-1)  # (T,)
        return float(sims.mean().item())

    def score_batch(self, video_paths: list[Path], prompt: str) -> Tensor:
        scores = [self.score(p, prompt) for p in video_paths]
        return torch.tensor(scores, dtype=torch.double)

    @torch.no_grad()
    def encode_image_for_steering(self, image: Image.Image) -> Tensor:
        """Return (1024,) CLIP embedding for a conditioning frame — used by SVDGenerator."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        return F.normalize(feat, dim=-1).squeeze(0)  # (1024,)
