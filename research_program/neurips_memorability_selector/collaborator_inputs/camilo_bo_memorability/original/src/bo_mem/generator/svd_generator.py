"""Video generator with SVD-XT + memorability steering.

Main flow
---------
1. Loads the Stable Video Diffusion XT pipeline (fp16, lazy — only on the 1st call).
2. Reads the direction vector v_mem_clip (1024-dim) from the TRIBE-CLIP adapter.
3. At generation time, injects alpha × v_mem_clip into the CLIP embedding of the
   conditioning image — an "activation steering" technique on the final CLIP projection.
4. Exports the frames as MP4 with PyAV.

Why monkey-patch _encode_image?
--------------------------------
SVD uses a CLIP-ViT-H encoder to turn the conditioning image into an
embedding (B, 1, 1024) that guides the entire diffusion. By adding the
memorability direction *after* the final projection, we operate exactly in the space where
the v_mem_clip vector was learned (via linear regression over TRIBE activations).

Usage on Kaggle (GPU with limited RAM)
--------------------------------------
To avoid OOM on the Kaggle T4 (~13 GB RAM), use cpu_offload=True:
    gen = SVDGenerator(cpu_offload=True)
    gen._load_pipeline()

This uses enable_model_cpu_offload() from diffusers/accelerate: the model stays
on the CPU and only the active subcomponent is moved to the GPU during inference.
Peak VRAM: ~3 GB (vs. ~16 GB without offload). Slower, but stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class SVDGenerator:
    # Default adapter path within the repository
    _DEFAULT_ADAPTER = (
        Path(__file__).parent.parent.parent.parent / "artifacts" / "tribe_clip_adapter.pt"
    )

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_frames: int = 25,
        adapter_path: Path | None = None,
        cpu_offload: bool = False,
    ) -> None:
        """
        Args:
            model_id:     Model ID on the HuggingFace Hub.
            device:       "cuda" or "cpu".
            dtype:        Precision — torch.float16 recommended for GPU.
            num_frames:   Number of frames in the generated video (SVD default = 25).
            adapter_path: Path to tribe_clip_adapter.pt.
                          If None, uses artifacts/tribe_clip_adapter.pt relative to the repo.
            cpu_offload:  If True, uses enable_model_cpu_offload() instead of .to(device).
                          Reduces peak VRAM from ~16 GB to ~3 GB. Requires accelerate.
                          Use True on the Kaggle T4; False on GPUs with VRAM >= 24 GB.
        """
        self.device = device
        self.dtype = dtype
        self.num_frames = num_frames
        self.cpu_offload = cpu_offload
        self._pipe = None
        self._vmem_clip: Tensor | None = None
        self._model_id = model_id

        # Load the direction vector from the adapter if available
        resolved = adapter_path or self._DEFAULT_ADAPTER
        if resolved.exists():
            self._load_adapter(resolved)

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_adapter(self, adapter_path: Path) -> None:
        """Loads v_mem_clip_h_via_adapter from the checkpoint and normalizes to unit norm."""
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        raw = ckpt["v_mem_clip_h_via_adapter"]       # tensor or array (1024,)
        arr = np.array(raw, dtype=np.float32)
        v = torch.from_numpy(arr)
        self._vmem_clip = v / v.norm()               # unit-normed (1024,)

    def _load_pipeline(self) -> None:
        """Loads the SVD-XT pipeline from the HuggingFace Hub (lazy — called on the 1st generation).

        Memory strategies:
          - low_cpu_mem_usage=True: streams from disk, avoids full staging in RAM.
          - cpu_offload=True: keeps the model on the CPU, moves subcomponents to the GPU
            during inference. Required on the Kaggle T4 (13 GB RAM / 16 GB VRAM).
          - cpu_offload=False + .to(device): loads everything into VRAM. Fast, but requires
            enough VRAM (~9 GB for SVD-XT fp16).
        """
        import gc

        from diffusers import StableVideoDiffusionPipeline

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            low_cpu_mem_usage=True,   # stream from disk → peak RAM ~4 GB (vs ~18 GB)
        )

        if self.cpu_offload:
            # Each subcomponent is moved to the GPU only during its forward pass.
            # Do NOT call .to(device) together with cpu_offload — conflict.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)

        pipe.enable_attention_slicing()   # reduces peak VRAM during attention
        self._pipe = pipe

    # ── Public API ──────────────────────────────────────────────────────────────

    def set_vmem_clip(self, vmem_clip: Tensor) -> None:
        """Sets the direction vector externally (normalized internally)."""
        self._vmem_clip = vmem_clip / vmem_clip.norm()

    def generate(
        self,
        conditioning_image: Image.Image,
        alphas: Tensor,
        guidance_scale: float = 3.0,
        seed: int = 0,
        num_inference_steps: int = 25,
        output_path: Path | None = None,
    ) -> Path:
        """Generates a video from a conditioning image with memorability steering.

        Args:
            conditioning_image: Input RGB image (PIL).
            alphas:             Tensor of steering coefficients. The total_alpha =
                                sum(alphas) is added to the CLIP embedding along the v_mem direction.
            guidance_scale:     CFG scale (classifier-free guidance). Typical: 1–15.
            seed:               Noise generator seed (reproducibility).
            num_inference_steps: Denoising steps (25 = standard quality, 10 = fast).
            output_path:        Output MP4 path. If None, saves to /tmp/.

        Returns:
            Path of the generated video.
        """
        # Load pipeline on the 1st call
        if self._pipe is None:
            self._load_pipeline()

        # Monkey-patch _encode_image to inject the steering vector
        # after the final CLIP projection (1024-dim).
        # The patch is reverted after generation so it does not affect subsequent calls.
        original_encode = self._pipe._encode_image

        if self._vmem_clip is not None:
            total_alpha = float(alphas.sum().item())
            v_dir = self._vmem_clip

            def steered_encode(image, device, num_videos_per_prompt, do_cfg):
                # emb: (batch, 1, 1024) after CLIP projection
                emb = original_encode(image, device, num_videos_per_prompt, do_cfg)
                # Move the direction to the same device/dtype as the embedding (works with cpu_offload)
                direction = v_dir.to(emb.device, dtype=emb.dtype)
                return emb + total_alpha * direction.unsqueeze(0).unsqueeze(0)

            self._pipe._encode_image = steered_encode

        generator = torch.manual_seed(seed)

        result = self._pipe(
            conditioning_image,
            num_frames=self.num_frames,
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=guidance_scale,
            max_guidance_scale=guidance_scale,
        )

        # Restore the original _encode_image
        self._pipe._encode_image = original_encode

        frames = result.frames[0]   # list of PIL.Image
        output_path = output_path or Path(f"/tmp/svd_gen_{seed}.mp4")
        self._save_frames_as_video(frames, output_path)
        return output_path

    # ── Utilities ───────────────────────────────────────────────────────────────

    @staticmethod
    def _save_frames_as_video(frames: list[Image.Image], path: Path, fps: int = 6) -> None:
        """Exports a list of PIL.Image as an H.264 MP4 with PyAV."""
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = frames[0].width
            stream.height = frames[0].height
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                video_frame = av.VideoFrame.from_image(frame)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
