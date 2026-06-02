"""Stable Video Diffusion generator on Modal, with CLIP-image steering.

SVD-XT (img2vid) takes a single input image, encodes it with CLIP-ViT-H-14
(1024-dim) for cross-attention conditioning AND with the VAE for spatial
conditioning. We patch the CLIP-image embedding before it reaches the U-Net:

    image_embedding ← image_embedding + alpha * v_mem_clip_h

Returns the generated 25-frame mp4 as bytes.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile

import modal

from audience_vectors.modal_app.app import MODAL_REGION, app, env_secrets
from audience_vectors.modal_app.image_factory import (
    SVD_CACHE_DIR,
    SVD_REPO_ID,
    svd_image,
)

logger = logging.getLogger(__name__)


svd_volume = modal.Volume.from_name("svd-weights-v1", create_if_missing=True)
svd_outputs_volume = modal.Volume.from_name("svd-outputs-v1", create_if_missing=True)

SVD_OUTPUTS_MOUNT = "/svd-outputs"


@app.function(
    image=svd_image,
    volumes={SVD_CACHE_DIR: svd_volume},
    secrets=env_secrets,
    timeout=60 * 60,
    cpu=4.0,
    memory=16 * 1024,
)
def populate_svd_weights() -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    print(f"[svd] downloading {SVD_REPO_ID} …")
    p = snapshot_download(SVD_REPO_ID)
    print(f"[svd] cached at {p}")
    svd_volume.commit()


@app.cls(
    region=MODAL_REGION,
    image=svd_image,
    gpu="H100",
    volumes={
        SVD_CACHE_DIR: svd_volume,
        SVD_OUTPUTS_MOUNT: svd_outputs_volume,
    },
    timeout=30 * 60,
    min_containers=0,
    max_containers=20,
    scaledown_window=180,
    enable_memory_snapshot=False,
    secrets=env_secrets,
)
class SVDGenerator:
    """Stable Video Diffusion wrapped for Modal."""

    @modal.enter()
    def load_model(self) -> None:
        import torch  # noqa: PLC0415
        from diffusers import StableVideoDiffusionPipeline  # noqa: PLC0415
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        svd_volume.reload()
        local = snapshot_download(SVD_REPO_ID)
        print(f"[svd] loading from {local}")
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            local, torch_dtype=torch.float16, variant="fp16",
        )
        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        print(f"[svd] ready")

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        *,
        steering_vector: list[float] | None = None,
        alpha: float = 0.0,
        num_frames: int = 25,
        num_inference_steps: int = 25,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        fps: int = 7,
        seed: int | None = None,
        output_label: str = "untitled",
    ) -> bytes:
        """Generate a video from a seed image, optionally steering the CLIP-image
        embedding before it reaches the U-Net cross-attention."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # SVD-XT expects 1024x576
        image = image.resize((1024, 576))

        generator = (
            torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        )

        # Monkey-patch the image encoder output if steering is requested.
        original_encode = self.pipe._encode_image
        if steering_vector is not None and abs(alpha) > 1e-6:
            v = torch.tensor(steering_vector, dtype=torch.float16, device="cuda")
            v = v / (v.norm() + 1e-8)

            def steered_encode(image, device, num_videos_per_prompt, do_classifier_free_guidance):
                emb = original_encode(image, device, num_videos_per_prompt, do_classifier_free_guidance)
                # emb shape: (B, 1, 1024) for SVD; broadcast steering vector
                steered = emb + alpha * v
                return steered

            self.pipe._encode_image = steered_encode

        try:
            with torch.no_grad():
                result = self.pipe(
                    image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    generator=generator,
                )
            frames = result.frames[0]
        finally:
            self.pipe._encode_image = original_encode

        import imageio  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        imageio.mimsave(
            tmp, [np.asarray(im) for im in frames], fps=fps, codec="libx264", quality=8,
        )
        with open(tmp, "rb") as fh:
            video_bytes = fh.read()
        os.unlink(tmp)

        out_path = f"{SVD_OUTPUTS_MOUNT}/{output_label}.mp4"
        with open(out_path, "wb") as fh:
            fh.write(video_bytes)
        svd_outputs_volume.commit()

        return video_bytes


__all__ = ["SVDGenerator", "populate_svd_weights"]
