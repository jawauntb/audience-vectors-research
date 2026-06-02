"""CogVideoX-5B generator on Modal, with optional conditioning-space steering.

Two functions:
  - `populate_cogvideox_weights`: one-shot weight cache populator.
  - `CogVideoXGenerator`: `@app.cls` class. Loads CogVideoX once; per call you
    pass a prompt + optional (steering_vector, alpha). Returns the generated mp4
    as bytes.

Conditioning-space steering: after the T5-XXL text encoder produces
`prompt_embeds (B, L, 4096)`, we add `alpha * steering_vector` (broadcast to
the L token dimension) BEFORE handing the embedding to the diffusion transformer.
This is the LLM-style activation-steering recipe applied to a video diffusion
text encoder.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile

import modal

from audience_vectors.modal_app.app import MODAL_REGION, app, env_secrets
from audience_vectors.modal_app.image_factory import (
    COGVIDEOX_CACHE_DIR,
    COGVIDEOX_REPO_ID,
    cogvideox_image,
)

logger = logging.getLogger(__name__)


cogvideox_volume = modal.Volume.from_name(
    "cogvideox-weights-v1", create_if_missing=True,
)
gen_outputs_volume = modal.Volume.from_name(
    "cogvideox-outputs-v1", create_if_missing=True,
)

GEN_OUTPUTS_MOUNT = "/cogvideox-outputs"


@app.function(
    image=cogvideox_image,
    volumes={COGVIDEOX_CACHE_DIR: cogvideox_volume},
    secrets=env_secrets,
    timeout=60 * 60,
    cpu=4.0,
    memory=16 * 1024,
)
def populate_cogvideox_weights() -> None:
    """Pre-download CogVideoX-5B + tokenizer + T5-XXL into the cache volume."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    print(f"[cogvideox] downloading {COGVIDEOX_REPO_ID} …")
    p = snapshot_download(COGVIDEOX_REPO_ID, local_dir_use_symlinks=False)
    print(f"[cogvideox] cached at: {p}")
    cogvideox_volume.commit()


@app.cls(
    region=MODAL_REGION,
    image=cogvideox_image,
    # H200 (sm_90, 141 GB). Account doesn't have B200 access; H200 is ~30% faster than
    # H100 and comfortably fits CogVideoX-5B + T5-XXL with batch_size>1. Torch 2.7 cu126
    # is forward-compatible on Hopper, so the image works unchanged if we get B200 later.
    gpu="H100",
    volumes={
        COGVIDEOX_CACHE_DIR: cogvideox_volume,
        GEN_OUTPUTS_MOUNT: gen_outputs_volume,
    },
    timeout=30 * 60,
    min_containers=0,
    # Burst up to 20 B200 workers in parallel for sweep jobs. Each one holds a
    # full CogVideoX-5B in VRAM (~25 GB), so memory is fine on B200 (180 GB).
    max_containers=20,
    scaledown_window=180,
    enable_memory_snapshot=False,
    secrets=env_secrets,
)
class CogVideoXGenerator:
    """CogVideoX-5B wrapped for Modal. Load weights once, generate many."""

    @modal.enter()
    def load_model(self) -> None:
        import torch  # noqa: PLC0415
        from diffusers import CogVideoXPipeline  # noqa: PLC0415
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        cogvideox_volume.reload()
        local = snapshot_download(COGVIDEOX_REPO_ID, local_dir_use_symlinks=False)
        print(f"[cogvideox] loading from {local}")
        self.pipe = CogVideoXPipeline.from_pretrained(local, torch_dtype=torch.bfloat16)
        # Memory savers — even B200 doesn't need them, but they don't hurt latency much
        self.pipe.enable_model_cpu_offload = False  # keep on GPU; B200 has 180GB
        self.pipe.to("cuda")
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        # Stash text-encoder output dim for steering vector validation
        self.text_embed_dim = 4096  # T5-XXL on CogVideoX
        print(f"[cogvideox] ready (text dim={self.text_embed_dim})")

    @modal.method()
    def predict_text_embedding(self, prompt: str) -> dict:
        """Encode a prompt with CogVideoX's T5 and return the embedding tensor.

        Used at adapter-training time: we encode all 1022 BMD captions on
        Modal so we have (TRIBE features, T5 embedding) pairs.
        """
        import torch  # noqa: PLC0415

        with torch.no_grad():
            embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                device="cuda",
                dtype=torch.bfloat16,
            )
        # embeds shape: (1, L, 4096) — mean-pool over sequence dim
        emb = embeds[0].mean(dim=0).float().cpu().numpy()
        return {
            "prompt": prompt,
            "embedding": emb.tolist(),
            "shape": list(emb.shape),
        }

    @modal.method()
    def generate(
        self,
        prompt: str,
        *,
        steering_vector: list[float] | None = None,
        alpha: float = 0.0,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: int | None = None,
        output_label: str = "untitled",
    ) -> bytes:
        """Generate a video, optionally with conditioning-space steering.

        steering_vector + alpha → injected into text encoder output:
            prompt_embeds[i] += alpha * steering_vector  (broadcast over L tokens)
        Returns the resulting mp4 as bytes (also persisted to the outputs volume).
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        generator = (
            torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        )

        # Encode prompt to get embeds (and the negative if CFG)
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt="static, blurry, slideshow, frozen",
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                device="cuda",
                dtype=torch.bfloat16,
            )

        # Conditioning-space steering — inject alpha · v_steer into the prompt
        # embedding's token dimension. v is (4096,); we broadcast across L tokens.
        if steering_vector is not None and abs(alpha) > 1e-6:
            v = torch.tensor(
                steering_vector, dtype=torch.bfloat16, device="cuda",
            )
            if v.shape[-1] != prompt_embeds.shape[-1]:
                raise ValueError(
                    f"steering_vector dim {v.shape[-1]} != text encoder dim {prompt_embeds.shape[-1]}"
                )
            v_norm = v / (v.norm() + 1e-8)
            prompt_embeds = prompt_embeds + alpha * v_norm  # broadcast over (B, L)

        # Generate
        with torch.no_grad():
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        frames = result.frames[0]  # list of PIL Images

        # Export to mp4
        import imageio  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        imageio.mimsave(tmp, [np.asarray(im) for im in frames], fps=8, codec="libx264", quality=8)
        with open(tmp, "rb") as fh:
            video_bytes = fh.read()
        os.unlink(tmp)

        # Persist to the outputs volume so we can re-run TRIBE without re-uploading
        out_path = f"{GEN_OUTPUTS_MOUNT}/{output_label}.mp4"
        with open(out_path, "wb") as fh:
            fh.write(video_bytes)
        gen_outputs_volume.commit()

        return video_bytes


__all__ = ["CogVideoXGenerator", "populate_cogvideox_weights"]
