"""Wan2.2 LoRA training helpers on Modal B200.

The first target is deliberately modest: cache the TRIBE-selected winner clips
and run a tiny Wan2.2 I2V low-noise LoRA SFT smoke. This is proxy distillation,
not a final human-validated preference model.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

from audience_vectors.modal_app.app import MODAL_REGION, app, env_secrets
from audience_vectors.modal_app.image_factory import (
    WAN22_LORA_CACHE_DIR,
    wan22_lora_image,
)

wan22_lora_cache_volume = modal.Volume.from_name(
    "wan22-lora-cache-v1", create_if_missing=True
)
wan22_lora_data_volume = modal.Volume.from_name(
    "wan22-lora-data-v1", create_if_missing=True
)
wan22_lora_outputs_volume = modal.Volume.from_name(
    "wan22-lora-outputs-v1", create_if_missing=True
)

WAN22_LORA_DATA_MOUNT = "/wan22-lora-data"
WAN22_LORA_OUTPUTS_MOUNT = "/wan22-lora-outputs"
WAN22_LORA_HF_CACHE = f"{WAN22_LORA_CACHE_DIR}/hub"

COMFY_WAN22_REPO_ID = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21_I2V_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-720P"
WAN22_I2V_LOW_NOISE = (
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
)
WAN22_I2V_HIGH_NOISE = (
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
)
WAN21_VAE = "split_files/vae/wan_2.1_vae.safetensors"
WAN_UMT5 = "models_t5_umt5-xxl-enc-bf16.pth"


def _configure_hf_cache() -> None:
    os.environ["HF_HOME"] = WAN22_LORA_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = WAN22_LORA_HF_CACHE
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    Path(WAN22_LORA_HF_CACHE).mkdir(parents=True, exist_ok=True)


def _module_file(module_name: str) -> str:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"could not find module file for {module_name}")
    return str(Path(module_file))


def _run(cmd: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    started = time.time()
    print("[wan22-lora] running:", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_s = time.time() - started
    payload = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
    }
    print(
        f"[wan22-lora] returncode={proc.returncode} elapsed={elapsed_s:.1f}s",
        flush=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(payload, indent=2))
    return payload


def _download_weights(include_high_noise: bool) -> dict[str, str]:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    filenames = {
        "t5": (WAN21_I2V_REPO_ID, WAN_UMT5),
        "vae": (COMFY_WAN22_REPO_ID, WAN21_VAE),
        "dit_low": (COMFY_WAN22_REPO_ID, WAN22_I2V_LOW_NOISE),
    }
    if include_high_noise:
        filenames["dit_high"] = (COMFY_WAN22_REPO_ID, WAN22_I2V_HIGH_NOISE)

    paths = {}
    for key, (repo_id, filename) in filenames.items():
        print(f"[wan22-lora] downloading/checking {repo_id}/{filename}", flush=True)
        paths[key] = hf_hub_download(
            repo_id,
            filename,
            cache_dir=WAN22_LORA_HF_CACHE,
        )
    wan22_lora_cache_volume.commit()
    return paths


def _write_dataset_config(
    dataset_root: Path,
    *,
    target_frames: int,
    width: int,
    height: int,
    source_fps: float,
) -> Path:
    videos_dir = dataset_root / "videos"
    cache_dir = dataset_root / "cache_latents"
    if not videos_dir.exists():
        raise FileNotFoundError(f"dataset videos dir missing: {videos_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = Path("/tmp/wan22_lora_dataset.toml")
    dataset_config.write_text(
        "\n".join(
            [
                "[general]",
                f"resolution = [{width}, {height}]",
                'caption_extension = ".txt"',
                "batch_size = 1",
                "enable_bucket = true",
                "bucket_no_upscale = false",
                "",
                "[[datasets]]",
                f'video_directory = "{videos_dir}"',
                f'cache_directory = "{cache_dir}"',
                f"target_frames = [{target_frames}]",
                'frame_extraction = "head"',
                f"source_fps = {source_fps:.1f}",
                "",
            ]
        )
    )
    return dataset_config


def _find_outputs(output_dir: Path, output_name: str) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(str(path) for path in output_dir.glob(f"{output_name}*.safetensors"))


def _latest_mp4(run_dir: Path) -> Path:
    mp4s = sorted(run_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not mp4s:
        raise FileNotFoundError(f"no mp4 generated in {run_dir}")
    return mp4s[-1]


def _tag_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


@app.function(
    image=wan22_lora_image,
    volumes={
        WAN22_LORA_CACHE_DIR: wan22_lora_cache_volume,
        WAN22_LORA_DATA_MOUNT: wan22_lora_data_volume,
        WAN22_LORA_OUTPUTS_MOUNT: wan22_lora_outputs_volume,
    },
    secrets=env_secrets,
    timeout=8 * 60 * 60,
    cpu=16.0,
    memory=128 * 1024,
)
def populate_wan22_lora_weights(include_high_noise: bool = False) -> dict[str, str]:
    """Pre-download Musubi-compatible Wan2.2 14B I2V training weights."""
    _configure_hf_cache()
    return _download_weights(include_high_noise=include_high_noise)


@app.function(
    region=MODAL_REGION,
    image=wan22_lora_image,
    gpu="B200",
    volumes={
        WAN22_LORA_CACHE_DIR: wan22_lora_cache_volume,
        WAN22_LORA_DATA_MOUNT: wan22_lora_data_volume,
        WAN22_LORA_OUTPUTS_MOUNT: wan22_lora_outputs_volume,
    },
    secrets=env_secrets,
    timeout=12 * 60 * 60,
    cpu=16.0,
    memory=128 * 1024,
)
def train_wan22_lora_smoke(
    dataset_name: str = "wan22_lora_sft_winners_50",
    output_name: str = "wan22_tribe_proxy_i2v_low_smoke",
    max_train_steps: int = 40,
    network_dim: int = 8,
    network_alpha: int = 8,
    learning_rate: float = 1e-4,
    target_frames: int = 81,
    width: int = 640,
    height: int = 352,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a tiny low-noise Wan2.2 I2V LoRA SFT smoke."""
    _configure_hf_cache()
    wan22_lora_data_volume.reload()
    dataset_root = Path(WAN22_LORA_DATA_MOUNT) / dataset_name
    output_dir = Path(WAN22_LORA_OUTPUTS_MOUNT) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = _download_weights(include_high_noise=False)
    dataset_config = _write_dataset_config(
        dataset_root,
        target_frames=target_frames,
        width=width,
        height=height,
        source_fps=24.0,
    )

    cache_latents = _run(
        [
            "python",
            _module_file("musubi_tuner.wan_cache_latents"),
            "--dataset_config",
            str(dataset_config),
            "--vae",
            weights["vae"],
            "--i2v",
            "--vae_dtype",
            "bf16",
            "--batch_size",
            "1",
            "--num_workers",
            "4",
            "--skip_existing",
            "--keep_cache",
        ]
    )
    wan22_lora_data_volume.commit()

    cache_text = _run(
        [
            "python",
            _module_file("musubi_tuner.wan_cache_text_encoder_outputs"),
            "--dataset_config",
            str(dataset_config),
            "--t5",
            weights["t5"],
            "--batch_size",
            "8",
            "--num_workers",
            "4",
            "--skip_existing",
            "--keep_cache",
        ]
    )
    wan22_lora_data_volume.commit()

    train = _run(
        [
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            "fp16",
            _module_file("musubi_tuner.wan_train_network"),
            "--task",
            "i2v-A14B",
            "--dit",
            weights["dit_low"],
            "--dataset_config",
            str(dataset_config),
            "--sdpa",
            "--mixed_precision",
            "fp16",
            "--fp8_base",
            "--optimizer_type",
            "adamw8bit",
            "--learning_rate",
            str(learning_rate),
            "--gradient_checkpointing",
            "--max_data_loader_n_workers",
            "2",
            "--persistent_data_loader_workers",
            "--network_module",
            "networks.lora_wan",
            "--network_dim",
            str(network_dim),
            "--network_alpha",
            str(network_alpha),
            "--timestep_sampling",
            "shift",
            "--discrete_flow_shift",
            "5.0",
            "--min_timestep",
            "0",
            "--max_timestep",
            "900",
            "--preserve_distribution_shape",
            "--max_train_steps",
            str(max_train_steps),
            "--save_every_n_steps",
            str(max_train_steps),
            "--seed",
            str(seed),
            "--output_dir",
            str(output_dir),
            "--output_name",
            output_name,
        ]
    )
    wan22_lora_outputs_volume.commit()

    result = {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "output_name": output_name,
        "max_train_steps": max_train_steps,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "learning_rate": learning_rate,
        "target": "Wan2.2 I2V-A14B low-noise only",
        "objective": "tribe_bmd_memorability_proxy_sft",
        "warning": "Proxy-selected clips, not direct human labels.",
        "weights": weights,
        "outputs": _find_outputs(output_dir, output_name),
        "cache_latents": cache_latents,
        "cache_text": cache_text,
        "train": train,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    wan22_lora_outputs_volume.commit()
    return result


@app.function(
    region=MODAL_REGION,
    image=wan22_lora_image,
    gpu="B200",
    volumes={
        WAN22_LORA_CACHE_DIR: wan22_lora_cache_volume,
        WAN22_LORA_DATA_MOUNT: wan22_lora_data_volume,
        WAN22_LORA_OUTPUTS_MOUNT: wan22_lora_outputs_volume,
    },
    secrets=env_secrets,
    timeout=12 * 60 * 60,
    cpu=16.0,
    memory=128 * 1024,
)
def generate_wan22_lora_eval(
    eval_name: str = "wan22_lora_eval_heldout_8",
    lora_output_name: str = "wan22_tribe_proxy_i2v_low_smoke_20",
    n: int = 2,
    infer_steps: int = 12,
    width: int = 640,
    height: int = 352,
    frames: int = 81,
    lora_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Generate base-vs-LoRA Wan2.2 I2V clips for a tiny held-out smoke eval."""
    _configure_hf_cache()
    wan22_lora_data_volume.reload()
    wan22_lora_outputs_volume.reload()

    eval_root = Path(WAN22_LORA_DATA_MOUNT) / eval_name
    prompts_path = eval_root / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)
    prompts = json.loads(prompts_path.read_text())[:n]

    lora_path = (
        Path(WAN22_LORA_OUTPUTS_MOUNT)
        / lora_output_name
        / f"{lora_output_name}.safetensors"
    )
    if not lora_path.exists():
        raise FileNotFoundError(lora_path)

    weights = _download_weights(include_high_noise=False)
    multiplier_tag = _tag_float(lora_multiplier)
    output_dir = (
        Path(WAN22_LORA_OUTPUTS_MOUNT)
        / (
            f"{lora_output_name}_{eval_name}_eval_"
            f"{len(prompts)}x2_s{infer_steps}_m{multiplier_tag}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_idx, row in enumerate(prompts):
        for variant in ("base", "lora"):
            label = f"{row['bmd_name']}_{variant}"
            run_dir = output_dir / f"_run_{label}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            cmd = [
                "python",
                _module_file("musubi_tuner.wan_generate_video"),
                "--task",
                "i2v-A14B",
                "--dit",
                weights["dit_low"],
                "--vae",
                weights["vae"],
                "--t5",
                weights["t5"],
                "--prompt",
                row["prompt"],
                "--image_path",
                str(eval_root / row["seed_image"]),
                "--video_size",
                str(height),
                str(width),
                "--video_length",
                str(frames),
                "--fps",
                "24",
                "--infer_steps",
                str(infer_steps),
                "--flow_shift",
                "5.0",
                "--guidance_scale",
                "5.0",
                "--seed",
                str(610000 + row_idx),
                "--save_path",
                str(run_dir),
                "--output_type",
                "video",
                "--attn_mode",
                "torch",
                "--fp8",
            ]
            if variant == "lora":
                cmd += [
                    "--lora_weight",
                    str(lora_path),
                    "--lora_multiplier",
                    str(lora_multiplier),
                ]

            run = _run(cmd)
            generated = _latest_mp4(run_dir)
            final_path = output_dir / f"{label}.mp4"
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(generated), final_path)
            rows.append(
                {
                    **row,
                    "variant": variant,
                    "label": label,
                    "video": str(final_path),
                    "infer_steps": infer_steps,
                    "lora_multiplier": lora_multiplier if variant == "lora" else 0.0,
                    "run": run,
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2))
    wan22_lora_outputs_volume.commit()
    return {
        "eval_name": eval_name,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "n_videos": len(rows),
        "rows": rows,
    }


@app.function(
    region=MODAL_REGION,
    image=wan22_lora_image,
    gpu="B200",
    volumes={
        WAN22_LORA_CACHE_DIR: wan22_lora_cache_volume,
        WAN22_LORA_DATA_MOUNT: wan22_lora_data_volume,
        WAN22_LORA_OUTPUTS_MOUNT: wan22_lora_outputs_volume,
    },
    secrets=env_secrets,
    timeout=12 * 60 * 60,
    cpu=16.0,
    memory=128 * 1024,
)
def generate_wan22_lora_best_of_n(  # noqa: C901
    eval_name: str = "wan22_lora_eval_fresh_picsum_8",
    lora_output_name: str = "wan22_tribe_proxy_i2v_low_r16_s150",
    n: int = 8,
    n_per_seed: int = 4,
    infer_steps: int = 12,
    width: int = 640,
    height: int = 352,
    frames: int = 81,
    lora_multiplier: float = 1.0,
    seed_base: int = 720000,
    start_idx: int = 0,
    end_idx: int | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Generate LoRA-only best-of-N candidates for preservation-gated selection."""
    _configure_hf_cache()
    wan22_lora_data_volume.reload()
    wan22_lora_outputs_volume.reload()

    eval_root = Path(WAN22_LORA_DATA_MOUNT) / eval_name
    prompts_path = eval_root / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)
    prompts = json.loads(prompts_path.read_text())[:n]
    if start_idx < 0 or start_idx > len(prompts):
        raise ValueError(f"start_idx out of range: {start_idx}")
    if end_idx is None:
        end_idx = len(prompts)
    if end_idx < start_idx or end_idx > len(prompts):
        raise ValueError(f"end_idx out of range: {end_idx}")

    lora_path = (
        Path(WAN22_LORA_OUTPUTS_MOUNT)
        / lora_output_name
        / f"{lora_output_name}.safetensors"
    )
    if not lora_path.exists():
        raise FileNotFoundError(lora_path)

    weights = _download_weights(include_high_noise=False)
    multiplier_tag = _tag_float(lora_multiplier)
    output_dir = (
        Path(WAN22_LORA_OUTPUTS_MOUNT)
        / (
            f"{lora_output_name}_{eval_name}_bon_"
            f"{len(prompts)}x{n_per_seed}_s{infer_steps}_m{multiplier_tag}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_idx, row in enumerate(prompts[start_idx:end_idx], start=start_idx):
        for sample_idx in range(n_per_seed):
            generation_seed = seed_base + row_idx * 100 + sample_idx
            label = f"{row['bmd_name']}_m{multiplier_tag}_n{sample_idx:02d}"
            final_path = output_dir / f"{label}.mp4"
            if skip_existing and final_path.exists():
                rows.append(
                    {
                        **row,
                        "variant": "lora",
                        "sample_index": sample_idx,
                        "generation_seed": generation_seed,
                        "label": label,
                        "video": str(final_path),
                        "infer_steps": infer_steps,
                        "lora_multiplier": lora_multiplier,
                        "skipped_existing": True,
                    }
                )
                continue
            run_dir = output_dir / f"_run_{label}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            cmd = [
                "python",
                _module_file("musubi_tuner.wan_generate_video"),
                "--task",
                "i2v-A14B",
                "--dit",
                weights["dit_low"],
                "--vae",
                weights["vae"],
                "--t5",
                weights["t5"],
                "--prompt",
                row["prompt"],
                "--image_path",
                str(eval_root / row["seed_image"]),
                "--video_size",
                str(height),
                str(width),
                "--video_length",
                str(frames),
                "--fps",
                "24",
                "--infer_steps",
                str(infer_steps),
                "--flow_shift",
                "5.0",
                "--guidance_scale",
                "5.0",
                "--seed",
                str(generation_seed),
                "--save_path",
                str(run_dir),
                "--output_type",
                "video",
                "--attn_mode",
                "torch",
                "--fp8",
                "--lora_weight",
                str(lora_path),
                "--lora_multiplier",
                str(lora_multiplier),
            ]
            run = _run(cmd)
            generated = _latest_mp4(run_dir)
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(generated), final_path)
            rows.append(
                {
                    **row,
                    "variant": "lora",
                    "sample_index": sample_idx,
                    "generation_seed": generation_seed,
                    "label": label,
                    "video": str(final_path),
                    "infer_steps": infer_steps,
                    "lora_multiplier": lora_multiplier,
                    "run": run,
                }
            )

    manifest_path = (
        output_dir / f"manifest_{start_idx:03d}_{end_idx:03d}.json"
        if start_idx != 0 or end_idx != len(prompts)
        else output_dir / "manifest.json"
    )
    manifest_path.write_text(json.dumps(rows, indent=2))
    wan22_lora_outputs_volume.commit()
    return {
        "eval_name": eval_name,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "n_videos": len(rows),
        "rows": rows,
    }


__all__ = [
    "generate_wan22_lora_best_of_n",
    "generate_wan22_lora_eval",
    "populate_wan22_lora_weights",
    "train_wan22_lora_smoke",
]
