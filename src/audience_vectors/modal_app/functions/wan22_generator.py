"""Wan2.2 generator on Modal B200.

This is the first high-quality open-weight video target for the next steering
round. It intentionally starts as a reliable official-CLI wrapper:

- cache Wan2.2 weights in a Modal Volume
- generate T2V, I2V, or TI2V clips on B200
- persist outputs to a Modal Volume and return mp4 bytes

Once generation + TRIBE scoring are moving, the next layer is adding hooks inside
Wan's DiT / conditioning path. Keeping the initial wrapper close to the official
CLI makes that next patch easier to validate against unmodified Wan outputs.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal

from audience_vectors.modal_app.app import MODAL_REGION, app, env_secrets
from audience_vectors.modal_app.image_factory import (
    WAN22_CACHE_DIR,
    WAN22_MODEL_REPOS,
    WAN22_REPO_DIR,
    wan22_image,
)

wan22_volume = modal.Volume.from_name("wan22-weights-v1", create_if_missing=True)
wan22_outputs_volume = modal.Volume.from_name(
    "wan22-outputs-v1", create_if_missing=True
)

WAN22_OUTPUTS_MOUNT = "/wan22-outputs"
WAN22_HF_HUB_CACHE = f"{WAN22_CACHE_DIR}/hub"


def _configure_hf_cache() -> None:
    """Force Wan downloads into the mounted Modal volume.

    The app forwards local env vars as Modal secrets, so a developer machine's
    HF_HOME can otherwise override the image-level cache path.
    """
    os.environ["HF_HOME"] = WAN22_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = WAN22_HF_HUB_CACHE
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    Path(WAN22_HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)


def _safe_output_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


def _model_repo_for_task(task: str) -> str:
    try:
        return WAN22_MODEL_REPOS[task]
    except KeyError as exc:
        choices = ", ".join(sorted(WAN22_MODEL_REPOS))
        raise ValueError(
            f"unsupported Wan2.2 task {task!r}; choose one of {choices}"
        ) from exc


def _build_generate_command(
    *,
    prompt: str,
    task: str,
    size: str,
    ckpt_dir: str,
    output_path: str,
    image_path: str | None,
    frame_num: int | None,
    sample_steps: int | None,
    sample_guide_scale: float | None,
    sample_shift: float | None,
    seed: int | None,
    offload_model: bool | None,
    use_prompt_extend: bool,
    prompt_extend_method: str,
    extra_args: list[str] | None,
) -> list[str]:
    cmd = [
        "python",
        "generate.py",
        "--task",
        task,
        "--size",
        size,
        "--ckpt_dir",
        ckpt_dir,
        "--prompt",
        prompt,
        "--save_file",
        output_path,
    ]
    if seed is not None:
        cmd += ["--base_seed", str(seed)]
    if image_path is not None:
        cmd += ["--image", image_path]
    if frame_num is not None:
        cmd += ["--frame_num", str(frame_num)]
    if sample_steps is not None:
        cmd += ["--sample_steps", str(sample_steps)]
    if sample_guide_scale is not None:
        cmd += ["--sample_guide_scale", str(sample_guide_scale)]
    if sample_shift is not None:
        cmd += ["--sample_shift", str(sample_shift)]
    if offload_model is not None:
        cmd += ["--offload_model", str(offload_model)]
    if use_prompt_extend:
        cmd += [
            "--use_prompt_extend",
            "--prompt_extend_method",
            prompt_extend_method,
            "--prompt_extend_target_lang",
            "en",
        ]
    if extra_args:
        cmd += extra_args
    return cmd


@app.function(
    image=wan22_image,
    volumes={WAN22_CACHE_DIR: wan22_volume},
    secrets=env_secrets,
    timeout=4 * 60 * 60,
    cpu=8.0,
    memory=64 * 1024,
)
def populate_wan22_weights(task: str = "ti2v-5B") -> dict[str, str]:
    """Pre-download a Wan2.2 checkpoint into the Modal cache volume."""
    _configure_hf_cache()
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    repo_id = _model_repo_for_task(task)
    print(f"[wan2.2] downloading {repo_id} for task={task} ...")
    path = snapshot_download(repo_id, cache_dir=WAN22_HF_HUB_CACHE)
    wan22_volume.commit()
    print(f"[wan2.2] cached at {path}")
    return {"task": task, "repo_id": repo_id, "path": path}


@app.function(
    image=wan22_image,
    timeout=10 * 60,
    cpu=2.0,
    memory=8 * 1024,
)
def smoke_wan22_import() -> str:
    """Verify the official Wan repo imports inside the built Modal image."""
    proc = subprocess.run(
        ["python", "-c", "import wan; print('wan import ok')"],
        cwd=WAN22_REPO_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Wan import failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


@app.cls(
    region=MODAL_REGION,
    image=wan22_image,
    gpu="B200",
    volumes={
        WAN22_CACHE_DIR: wan22_volume,
        WAN22_OUTPUTS_MOUNT: wan22_outputs_volume,
    },
    timeout=4 * 60 * 60,
    cpu=16.0,
    memory=128 * 1024,
    min_containers=0,
    max_containers=8,
    scaledown_window=300,
    enable_memory_snapshot=False,
    secrets=env_secrets,
)
class Wan22Generator:
    """Official Wan2.2 CLI wrapped as a Modal class on B200."""

    @modal.enter()
    def load(self) -> None:
        _configure_hf_cache()
        wan22_volume.reload()
        Path(WAN22_OUTPUTS_MOUNT).mkdir(parents=True, exist_ok=True)
        self.repo_dir = WAN22_REPO_DIR
        print("[wan2.2] ready; official repo at", self.repo_dir)

    @modal.method()
    def generate(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        task: str = "ti2v-5B",
        size: str = "1280*704",
        frame_num: int | None = None,
        sample_steps: int | None = None,
        sample_guide_scale: float | None = None,
        sample_shift: float | None = None,
        seed: int | None = None,
        offload_model: bool | None = False,
        output_label: str = "wan22",
        use_prompt_extend: bool = False,
        prompt_extend_method: str = "local_qwen",
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a Wan2.2 mp4.

        Defaults target TI2V-5B because it is the fastest high-quality open path
        for text+image-to-video experiments. For pure high-quality I2V, pass
        task="i2v-A14B", size="1280*720".
        """
        _configure_hf_cache()
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        repo_id = _model_repo_for_task(task)
        ckpt_dir = snapshot_download(repo_id, cache_dir=WAN22_HF_HUB_CACHE)
        safe_label = _safe_output_label(output_label)
        output_path = f"{WAN22_OUTPUTS_MOUNT}/{safe_label}.mp4"

        image_path: str | None = None
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="wan22-input-")
        try:
            temp_dir = Path(temp_dir_ctx.name)
            if image_bytes is not None:
                image_path = str(temp_dir / "seed_image.png")
                Path(image_path).write_bytes(image_bytes)

            cmd = _build_generate_command(
                prompt=prompt,
                task=task,
                size=size,
                ckpt_dir=ckpt_dir,
                output_path=output_path,
                image_path=image_path,
                frame_num=frame_num,
                sample_steps=sample_steps,
                sample_guide_scale=sample_guide_scale,
                sample_shift=sample_shift,
                seed=seed,
                offload_model=offload_model,
                use_prompt_extend=use_prompt_extend,
                prompt_extend_method=prompt_extend_method,
                extra_args=extra_args,
            )

            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            print("[wan2.2] running:", " ".join(cmd))
            proc = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "[wan2.2] generation failed\n"
                    f"returncode={proc.returncode}\n"
                    f"stdout:\n{proc.stdout[-6000:]}\n"
                    f"stderr:\n{proc.stderr[-6000:]}"
                )

            video_bytes = Path(output_path).read_bytes()
            wan22_outputs_volume.commit()
            return {
                "task": task,
                "repo_id": repo_id,
                "ckpt_dir": ckpt_dir,
                "size": size,
                "frame_num": frame_num,
                "sample_steps": sample_steps,
                "sample_guide_scale": sample_guide_scale,
                "sample_shift": sample_shift,
                "seed": seed,
                "offload_model": offload_model,
                "output_label": safe_label,
                "modal_output_path": output_path,
                "bytes": len(video_bytes),
                "video_bytes": video_bytes,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        finally:
            temp_dir_ctx.cleanup()


__all__ = ["Wan22Generator", "populate_wan22_weights", "smoke_wan22_import"]
