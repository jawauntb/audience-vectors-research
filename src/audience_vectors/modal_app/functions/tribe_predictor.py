"""TRIBE v2 predictor on Modal — research-grade, simplified.

This is a stripped-down version of the production pattern in
superoptimizers (gic_modal/functions/neural_engagement_scoring.py).
Dropped relative to the production version:

  - SSRF guards / pinned-IP HTTPS connection handlers
  - Prod/dev environment splits + HF_HUB_OFFLINE enforcement
  - Logfire span instrumentation
  - whisperx uvx shim (model path patching) — re-add if text input matters
  - ffprobe duration validation guards
  - Modal observability decorators

Kept because they genuinely matter even in research:

  - Pinned HF revisions (drift = silently wrong activations)
  - Shared weights volume (cold-start without 10 GB of downloads)
  - @modal.enter(snap=True) + GPU memory snapshots
  - Suffix normalization for signed URLs ending in `.bin`
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from importlib import import_module
from pathlib import Path
from typing import Any

import modal  # type: ignore[import-not-found]
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from audience_vectors.modal_app.app import MODAL_REGION, app, env_secrets
from audience_vectors.modal_app.image_factory import (
    HF_CACHE_DIR,
    TRIBE_FEATURE_MODEL_PINS,
    TRIBE_HF_REPO_ID,
    TRIBE_HF_REVISION,
    WHISPERX_MODEL_REPO_ID,
    WHISPERX_MODEL_REVISION,
    tribe_image,
)

# Shared weights volume — populate once via `populate_tribe_weights`.
# Set `create_if_missing=True` for first run; flip to False once you've
# validated it and want loud failures on missing volumes.
tribe_weights_volume = modal.Volume.from_name(
    "tribe-v2-weights-v1", create_if_missing=True
)
bmd_videos_volume = modal.Volume.from_name(
    "bmd-videos-v1", create_if_missing=True,
)
BMD_VIDEOS_MOUNT = "/bmd-videos"


_TRIBE_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm"})
_TRIBE_TEXT_SUFFIXES = frozenset({".txt"})
_MAX_VIDEO_DURATION_SECONDS = 60.0
_MAX_REMOTE_DOWNLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB
_INTROSPECTION_PATTERNS = (
    "pos",
    "position",
    "rope",
    "rotary",
    "temporal",
    "time",
    "frame",
)
_MAX_FFT_NUMEL = 20_000_000


class VideoPredictionResult(BaseModel):
    """TRIBE v2 video output: per-frame (time, vertices) activation tensor.

    Service layer means across frames for a scalar score; keeps both
    tensors + duration so callers can window/aggregate as needed.
    """

    frames: list[list[float]]
    duration_seconds: float


class VideoPreflightResult(BaseModel):
    """Lightweight video-readiness check before expensive TRIBE prediction."""

    input_kind: str
    event_mode: str
    resolved_path: str
    tribe_path: str
    exists: bool
    size_bytes: int | None
    duration_seconds: float
    events_rows: int
    event_columns: list[str]
    step_seconds: dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_remote(url: str) -> str:
    """Stream a remote URL to /tmp, enforcing a size cap. Returns local path."""
    suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir="/tmp")
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "audience-vectors/0.1"})
    written = 0
    with urllib.request.urlopen(req, timeout=60) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > _MAX_REMOTE_DOWNLOAD_BYTES:
            raise RuntimeError(f"remote media too large: {declared} bytes")
        with open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                written += len(chunk)
                if written > _MAX_REMOTE_DOWNLOAD_BYTES:
                    raise RuntimeError("remote media exceeded size cap")
                out.write(chunk)
    return tmp


def _force_container_tmpdir() -> None:
    """Keep third-party audio/transcript tooling off host-specific temp paths."""
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = "/tmp"
    tempfile.tempdir = "/tmp"


def _resolve_local_path(path_or_url: str) -> tuple[str, bool]:
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme == "":
        return path_or_url, False
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URI scheme: {parsed.scheme}")
    return _download_remote(path_or_url), True


def _ensure_tribe_suffix(local_path: str) -> tuple[str, str | None]:
    """Symlink to `.mp4` if suffix doesn't match TRIBE's allowlist.

    Returns (path_to_use, cleanup_dir_or_None).
    """
    suffix = os.path.splitext(local_path)[1].lower()
    if suffix in _TRIBE_VIDEO_SUFFIXES:
        return local_path, None
    link_dir = tempfile.mkdtemp(prefix="tribe_video_", dir="/tmp")
    target = os.path.join(link_dir, "video.mp4")
    os.symlink(os.path.abspath(local_path), target)
    return target, link_dir


def _ensure_text_suffix(local_path: str) -> tuple[str, str | None]:
    """Symlink to `.txt` if suffix doesn't match TRIBE's text allowlist."""
    suffix = os.path.splitext(local_path)[1].lower()
    if suffix in _TRIBE_TEXT_SUFFIXES:
        return local_path, None
    link_dir = tempfile.mkdtemp(prefix="tribe_text_", dir="/tmp")
    target = os.path.join(link_dir, "stimulus.txt")
    os.symlink(os.path.abspath(local_path), target)
    return target, link_dir


def _probe_duration(local_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", local_path],
        capture_output=True, check=True, text=True, timeout=30,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise RuntimeError(f"non-positive video duration: {duration}")
    return duration


def _matches_introspection_name(name: str) -> bool:
    lower = name.lower()
    return any(pattern in lower for pattern in _INTROSPECTION_PATTERNS)


def _frequency_energy_1d(seq: Any) -> dict[str, float]:
    import numpy as np

    spectrum = np.fft.fft(seq, axis=0, norm="ortho")
    energy = (np.abs(spectrum) ** 2).sum(axis=1)
    total = float(energy.sum())
    freqs = np.abs(np.fft.fftfreq(seq.shape[0]))
    max_freq = float(freqs.max()) or 1.0
    rel = freqs / max_freq
    masks = {
        "dc": freqs == 0,
        "low_nonzero": (freqs > 0) & (rel <= 0.25),
        "mid": (rel > 0.25) & (rel <= 0.50),
        "high": rel > 0.50,
    }
    return {
        name: float(energy[mask].sum() / total) if total else 0.0
        for name, mask in masks.items()
    }


def _tensor_introspection(name: str, tensor: Any) -> dict[str, Any]:
    import numpy as np

    detached = tensor.detach().float().cpu()
    arr = detached.numpy()
    row: dict[str, Any] = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(tensor.dtype),
        "numel": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "norm": float(np.linalg.norm(arr.reshape(-1))) if arr.size else 0.0,
    }
    seq_info = _sequence_view(arr)
    if seq_info is not None and arr.size <= _MAX_FFT_NUMEL:
        seq, seq_axis = seq_info
        row["fft_sequence_axis"] = seq_axis
        row["fft_energy_1d_raw"] = _frequency_energy_1d(seq)
        centered = seq - seq.mean(axis=0, keepdims=True)
        row["fft_energy_1d_centered"] = _frequency_energy_1d(centered)
        seq_len = seq.shape[0]
        grid, has_prefix = _square_grid_from_sequence_length(seq_len)
        if grid is not None:
            patch_seq = seq[1:] if has_prefix else seq
            patches = patch_seq.reshape(grid, grid, -1)
            spectrum = np.fft.fftn(patches, axes=(0, 1), norm="ortho")
            energy = (np.abs(spectrum) ** 2).sum(axis=-1)
            total = float(energy.sum())
            fy = np.abs(np.fft.fftfreq(grid))
            fx = np.abs(np.fft.fftfreq(grid))
            yy, xx = np.meshgrid(fy, fx, indexing="ij")
            radius = np.sqrt(xx**2 + yy**2)
            rel = radius / (float(radius.max()) or 1.0)
            row["fft_energy_2d_patch_grid"] = {
                "grid": grid,
                "dc": float(energy[rel == 0].sum() / total) if total else 0.0,
                "low": float(energy[(rel > 0) & (rel <= 0.25)].sum() / total)
                if total
                else 0.0,
                "mid": float(energy[(rel > 0.25) & (rel <= 0.50)].sum() / total)
                if total
                else 0.0,
                "high": float(energy[rel > 0.50].sum() / total) if total else 0.0,
            }
    elif arr.size > _MAX_FFT_NUMEL:
        row["fft_skipped"] = f"numel>{_MAX_FFT_NUMEL}"
    return row


def _sequence_view(arr: Any) -> tuple[Any, int] | None:
    import numpy as np

    if arr.ndim == 1 and arr.shape[0] > 1:
        return arr[:, None], 0
    if arr.ndim < 2:
        return None
    axis = 0
    if arr.shape[0] == 1 and arr.ndim >= 3 and arr.shape[1] > 1:
        axis = 1
    if arr.shape[axis] <= 1:
        return None
    moved = np.moveaxis(arr, axis, 0)
    return moved.reshape(moved.shape[0], -1), axis


def _square_grid_from_sequence_length(seq_len: int) -> tuple[int | None, bool]:
    grid = int(round(seq_len ** 0.5))
    if grid * grid == seq_len:
        return grid, False
    grid = int(round((seq_len - 1) ** 0.5))
    if grid * grid == seq_len - 1:
        return grid, True
    return None, False


def _candidate_children(obj: Any) -> list[tuple[str, Any]]:
    items: dict[str, Any] = {}
    raw_dict = getattr(obj, "__dict__", None)
    if isinstance(raw_dict, dict):
        items.update(raw_dict)
    pydantic_private = getattr(obj, "__pydantic_private__", None)
    if isinstance(pydantic_private, dict):
        items.update(pydantic_private)
    pydantic_extra = getattr(obj, "__pydantic_extra__", None)
    if isinstance(pydantic_extra, dict):
        items.update(pydantic_extra)
    model_fields = getattr(type(obj), "model_fields", {})
    for name in model_fields:
        try:
            items.setdefault(name, getattr(obj, name))
        except AttributeError:
            pass
    return [(name, value) for name, value in sorted(items.items()) if value is not None]


def _prefixed(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _traversal_children(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [(_prefixed(prefix, str(key)), child) for key, child in value.items()]
    if isinstance(value, (list, tuple)):
        return [(_prefixed(prefix, str(idx)), child) for idx, child in enumerate(value)]
    return [
        (_prefixed(prefix, name), child)
        for name, child in _candidate_children(value)
        if not name.startswith("_") or name == "_model"
    ]


def _find_torch_roots(obj: Any) -> list[tuple[str, Any]]:
    import torch

    roots: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def visit(prefix: str, value: Any, depth: int) -> None:
        obj_id = id(value)
        if obj_id in seen:
            return
        if isinstance(value, torch.nn.Module):
            seen.add(obj_id)
            roots.append((prefix or "model", value))
            return
        if depth <= 0:
            return
        for child_prefix, child in _traversal_children(prefix, value):
            visit(child_prefix, child, depth - 1)

    visit("", obj, 3)
    return roots


def _wrapper_attribute_inventory(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in _candidate_children(obj):
        if name.startswith("_") and name != "_model":
            continue
        shape = list(value.shape) if hasattr(value, "shape") else None
        rows.append(
            {
                "name": name,
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "shape": shape,
            }
        )
    return rows


def _format_module_tree_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TRIBE Module Tree",
        "",
        f"- Model class: `{report['model_class']}`",
        f"- Torch root: `{report['torch_root']}`",
        f"- Modules listed: **{len(report['modules'])}**",
        "",
        "| name | type | parameters | buffers |",
        "|---|---|---:|---:|",
    ]
    for row in report["modules"]:
        lines.append(
            f"| `{row['name']}` | `{row['type']}` | "
            f"{int(row['parameters']):,} | {int(row['buffers']):,} |"
        )
    return "\n".join(lines) + "\n"


def _lookup_torch_module(torch_root: Any, module_name: str) -> Any:
    normalized = module_name.strip()
    if normalized in {"", "_model"}:
        return torch_root
    if normalized.startswith("_model."):
        normalized = normalized.removeprefix("_model.")
    modules = dict(torch_root.named_modules())
    if normalized not in modules:
        available = ", ".join(sorted(modules)[:20])
        raise RuntimeError(
            f"unknown TRIBE module {module_name!r}; first available modules: {available}"
        )
    return modules[normalized]


def _first_tensor(value: Any) -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _replace_tensor_in_sequence(items: Any, replacement: Any) -> list[Any]:
    rows = []
    replaced = False
    for item in items:
        if not replaced and _first_tensor(item) is not None:
            rows.append(_replace_first_tensor(item, replacement))
            replaced = True
        else:
            rows.append(item)
    return rows


def _replace_first_tensor(value: Any, replacement: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return replacement
    if isinstance(value, tuple):
        return tuple(_replace_tensor_in_sequence(value, replacement))
    if isinstance(value, list):
        return _replace_tensor_in_sequence(value, replacement)
    if isinstance(value, dict):
        rows = dict(value)
        for key, item in value.items():
            if _first_tensor(item) is not None:
                rows[key] = _replace_first_tensor(item, replacement)
                break
        return rows
    raise RuntimeError(f"cannot replace tensor inside output type {type(value)!r}")


def _hidden_sequence_axis(tensor: Any) -> int | None:
    if tensor.ndim < 2:
        return None
    if tensor.ndim == 2:
        return 0
    if int(tensor.shape[0]) == 1:
        return 1
    return tensor.ndim - 2


def _patch_hidden_tensor(tensor: Any, patch_mode: str, patch_scale: float) -> Any:
    mode = patch_mode.strip().lower()
    if mode == "none":
        return tensor
    if mode not in {"non_dc_scale", "dc_scale", "all_scale"}:
        raise RuntimeError(f"unsupported hidden patch mode: {patch_mode}")
    if mode == "all_scale":
        return tensor * float(patch_scale)
    seq_axis = _hidden_sequence_axis(tensor)
    if seq_axis is None:
        raise RuntimeError(f"cannot infer sequence axis for shape {list(tensor.shape)}")
    dc = tensor.mean(dim=seq_axis, keepdim=True)
    non_dc = tensor - dc
    if mode == "non_dc_scale":
        return dc + non_dc * float(patch_scale)
    return dc * float(patch_scale) + non_dc


def _compressed_numpy_bytes(name: str, tensor: Any) -> bytes:
    import io

    import numpy as np

    buffer = io.BytesIO()
    arr = tensor.detach().float().cpu().numpy().astype(np.float16)
    np.savez_compressed(buffer, **{name: arr})
    return buffer.getvalue()


def _direction_from_numpy_bytes(payload: bytes, tensor: Any) -> Any:
    import io

    import numpy as np
    import torch

    loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    if "direction" not in loaded:
        raise RuntimeError("direction payload must contain a `direction` array")
    direction = torch.as_tensor(loaded["direction"], device=tensor.device).to(
        dtype=tensor.dtype
    )
    if tensor.ndim == direction.ndim + 1 and int(tensor.shape[0]) == 1:
        direction = direction.unsqueeze(0)
    if tuple(direction.shape) != tuple(tensor.shape):
        raise RuntimeError(
            f"direction shape {list(direction.shape)} does not match hidden shape "
            f"{list(tensor.shape)}"
        )
    batch_size = int(direction.shape[0]) if tensor.ndim >= 3 and int(tensor.shape[0]) == 1 else 1
    norm = torch.linalg.vector_norm(direction.reshape(batch_size, -1), dim=1)
    if torch.any(norm <= 1e-6):
        raise RuntimeError("near-zero hidden direction")
    view_shape = [batch_size, *([1] * (direction.ndim - 1))]
    return direction / norm.reshape(view_shape)


def _patch_hidden_direction_tensor(
    tensor: Any,
    direction_payload: bytes,
    patch_alpha: float,
) -> Any:
    direction = _direction_from_numpy_bytes(direction_payload, tensor)
    batch_size = int(tensor.shape[0]) if tensor.ndim >= 3 and int(tensor.shape[0]) == 1 else 1
    flat_tensor = tensor.reshape(batch_size, -1)
    flat_direction = direction.reshape(batch_size, -1)
    projection = (flat_tensor * flat_direction).sum(dim=1)
    view_shape = [batch_size, *([1] * (tensor.ndim - 1))]
    return tensor - float(patch_alpha) * projection.reshape(view_shape) * direction


def _format_tribe_introspection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TRIBE Model Introspection",
        "",
        f"- Model class: `{report['model_class']}`",
        f"- Torch roots found: **{len(report['torch_roots'])}**",
        f"- Total parameters: **{report['total_parameters']:,}**",
        f"- Matched modules: **{len(report['matching_modules'])}**",
        f"- Matched parameters: **{len(report['matching_parameters'])}**",
        f"- Matched buffers: **{len(report['matching_buffers'])}**",
        "",
        "## Torch Roots",
        "",
    ]
    if report["torch_roots"]:
        lines.extend(f"- `{row['name']}`: `{row['type']}`" for row in report["torch_roots"])
    else:
        lines.append("No nested `torch.nn.Module` roots were found in the wrapper.")
    lines += [
        "",
        "## Matching Modules",
        "",
    ]
    if report["matching_modules"]:
        lines.extend(f"- `{name}`" for name in report["matching_modules"][:80])
    else:
        lines.append("No module names matched the positional/temporal patterns.")
    lines += [
        "",
        "## Matching Parameters And Buffers",
        "",
        "| kind | name | shape | centered FFT DC | centered FFT high |",
        "|---|---|---:|---:|---:|",
    ]
    rows: list[tuple[str, dict[str, Any]]] = [
        *[("parameter", row) for row in report["matching_parameters"]],
        *[("buffer", row) for row in report["matching_buffers"]],
    ]
    if not rows:
        lines.append("| none | - | - | - | - |")
    for kind, row in rows[:120]:
        centered = row.get("fft_energy_1d_centered", {})
        lines.append(
            f"| {kind} | `{row['name']}` | `{row['shape']}` | "
            f"{float(centered.get('dc', 0.0)):.3f} | "
            f"{float(centered.get('high', 0.0)):.3f} |"
        )
    lines += [
        "",
        "## Interpretation Note",
        "",
        "This is an architectural inventory, not a causal patch. If explicit positional, rotary, or temporal tensors are present, the next step is to save hidden states around the matching modules and rerun the memorability-direction FFT/patching tests internally.",
    ]
    if report["wrapper_attributes"]:
        lines += [
            "",
            "## Wrapper Attributes",
            "",
            "| name | type | shape |",
            "|---|---|---:|",
        ]
        for row in report["wrapper_attributes"][:120]:
            lines.append(f"| `{row['name']}` | `{row['type']}` | `{row['shape']}` |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# One-shot weight populator
# ---------------------------------------------------------------------------


@app.function(
    image=tribe_image,
    volumes={HF_CACHE_DIR: tribe_weights_volume},
    secrets=env_secrets,
    timeout=60 * 60,
    cpu=4.0,
    memory=16 * 1024,
)
def populate_tribe_weights() -> None:
    """Populate the shared HF cache volume.

    Run once after deploy:

        modal run audience_vectors.modal_app.functions.tribe_predictor::populate_tribe_weights

    `meta-llama/Llama-3.2-3B` is gated — your HF token must have accepted
    Meta's license. The token flows in via env_secrets.
    """
    for repo_id, revision in (
        (TRIBE_HF_REPO_ID, TRIBE_HF_REVISION),
        *TRIBE_FEATURE_MODEL_PINS,
        (WHISPERX_MODEL_REPO_ID, WHISPERX_MODEL_REVISION),
    ):
        snapshot_download(repo_id, revision=revision)
    tribe_weights_volume.commit()


@app.function(
    region=MODAL_REGION,
    image=tribe_image,
    volumes={HF_CACHE_DIR: tribe_weights_volume},
    secrets=env_secrets,
    gpu="H100!",
    timeout=20 * 60,
    cpu=4.0,
    memory=32 * 1024,
)
def introspect_tribe_model() -> dict[str, Any]:
    """Return TRIBE module/parameter metadata for positional critique follow-up."""
    from tribev2 import TribeModel  # type: ignore[import-not-found]

    tribe_path = snapshot_download(TRIBE_HF_REPO_ID, revision=TRIBE_HF_REVISION)
    model = TribeModel.from_pretrained(tribe_path, device="cuda")
    torch_roots = _find_torch_roots(model)
    modules = [
        f"{prefix}.{name}" if name else prefix
        for prefix, root in torch_roots
        for name, _module in root.named_modules()
        if _matches_introspection_name(f"{prefix}.{name}" if name else prefix)
    ]
    parameters = [
        _tensor_introspection(f"{prefix}.{name}", tensor)
        for prefix, root in torch_roots
        for name, tensor in root.named_parameters()
        if _matches_introspection_name(f"{prefix}.{name}")
    ]
    buffers = [
        _tensor_introspection(f"{prefix}.{name}", tensor)
        for prefix, root in torch_roots
        for name, tensor in root.named_buffers()
        if _matches_introspection_name(f"{prefix}.{name}")
    ]
    total_parameters = sum(
        int(param.numel()) for _prefix, root in torch_roots for param in root.parameters()
    )
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "torch_roots": [
            {"name": name, "type": f"{type(root).__module__}.{type(root).__qualname__}"}
            for name, root in torch_roots
        ],
        "total_parameters": total_parameters,
        "matching_modules": modules,
        "matching_parameters": parameters,
        "matching_buffers": buffers,
        "wrapper_attributes": _wrapper_attribute_inventory(model),
    }


@app.function(
    region=MODAL_REGION,
    image=tribe_image,
    volumes={HF_CACHE_DIR: tribe_weights_volume},
    secrets=env_secrets,
    gpu="H100!",
    timeout=20 * 60,
    cpu=4.0,
    memory=32 * 1024,
)
def inspect_tribe_module_tree() -> dict[str, Any]:
    """Return the full nested torch module tree for choosing hook targets."""
    from tribev2 import TribeModel  # type: ignore[import-not-found]

    tribe_path = snapshot_download(TRIBE_HF_REPO_ID, revision=TRIBE_HF_REVISION)
    model = TribeModel.from_pretrained(tribe_path, device="cuda")
    torch_roots = _find_torch_roots(model)
    if not torch_roots:
        raise RuntimeError("no torch root found inside TRIBE wrapper")
    root_name, root = torch_roots[0]
    rows = []
    for name, module in root.named_modules():
        full_name = root_name if not name else f"{root_name}.{name}"
        own_parameters = sum(
            int(param.numel()) for param in module.parameters(recurse=False)
        )
        own_buffers = sum(int(buf.numel()) for buf in module.buffers(recurse=False))
        rows.append(
            {
                "name": full_name,
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                "parameters": own_parameters,
                "buffers": own_buffers,
            }
        )
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "torch_root": root_name,
        "modules": rows,
    }


@app.local_entrypoint()
def inspect_tribe_module_tree_cli(
    out_json: str = "data/reports/tribe_module_tree.json",
    out_md: str = "data/reports/tribe_module_tree.md",
) -> None:
    """Run Modal-side module tree inspection and save local reports."""
    report = inspect_tribe_module_tree.remote()
    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_format_module_tree_markdown(report))
    print(f"[tribe-module-tree] wrote {json_path}")
    print(f"[tribe-module-tree] wrote {md_path}")


@app.local_entrypoint()
def introspect_tribe_model_cli(
    out_json: str = "data/reports/tribe_model_introspection.json",
    out_md: str = "data/reports/tribe_model_introspection.md",
) -> None:
    """Run Modal-side TRIBE introspection and save local reports."""
    report = introspect_tribe_model.remote()
    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_format_tribe_introspection_markdown(report))
    print(f"[tribe-introspection] wrote {json_path}")
    print(f"[tribe-introspection] wrote {md_path}")


# ---------------------------------------------------------------------------
# Predictor — class-based so weights load once per warm container.
# ---------------------------------------------------------------------------


@app.cls(
    region=MODAL_REGION,
    image=tribe_image,
    # B200 (sm_100) needs torch>=2.7; TRIBE pins torch<2.7 so we use H100 (sm_90).
    # Strict `H100!` prevents Modal from auto-upgrading to H200 (kernel mismatch).
    gpu="H100!",
    volumes={
        HF_CACHE_DIR: tribe_weights_volume,
        BMD_VIDEOS_MOUNT: bmd_videos_volume,
    },
    timeout=20 * 60,
    min_containers=0,
    scaledown_window=300,
    # Snapshots cached a stale view of the bmd-videos volume; disable for now.
    enable_memory_snapshot=False,
    secrets=env_secrets,
)
class TribeV2Predictor:
    """TRIBE v2 wrapped for Modal. Load once, predict many times."""

    @modal.enter()
    def load_model(self) -> None:
        # Inline import — keeps module-import cheap for orchestration paths.
        from tribev2 import TribeModel  # type: ignore[import-not-found]

        _force_container_tmpdir()
        # Refresh BMD videos volume so containers see the latest uploaded files.
        bmd_videos_volume.reload()

        tribe_path = snapshot_download(TRIBE_HF_REPO_ID, revision=TRIBE_HF_REVISION)
        self.model = TribeModel.from_pretrained(tribe_path, device="cuda")

    def _video_events_dataframe(self, tribe_path: str, *, audio_only: bool):
        if not audio_only:
            return self.model.get_events_dataframe(video_path=tribe_path)

        import pandas as pd  # inline import to keep module-load light

        demo_utils = import_module("tribev2.demo_utils")
        event = {
            "type": "Video",
            "filepath": str(tribe_path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return demo_utils.get_audio_and_text_events(
            pd.DataFrame([event]),
            audio_only=True,
        )

    def _predict_video_impl(
        self,
        video_path_or_url: str,
        *,
        audio_only: bool = False,
    ) -> VideoPredictionResult:
        """Predict per-vertex brain activations for a video stimulus."""
        import numpy as np  # inline import to keep module-load light

        _force_container_tmpdir()
        # Uploaded generated/YouTube clips can arrive after a warm container starts.
        # Reload before path resolution so Modal volume-backed scoring is not stale.
        bmd_videos_volume.reload()
        local_path, is_temp = _resolve_local_path(video_path_or_url)
        cleanup_dir: str | None = None
        try:
            duration = _probe_duration(local_path)
            if duration > _MAX_VIDEO_DURATION_SECONDS:
                raise ValueError(
                    f"video too long: {duration:.1f}s > {_MAX_VIDEO_DURATION_SECONDS:.0f}s"
                )
            tribe_path, cleanup_dir = _ensure_tribe_suffix(local_path)
            events = self._video_events_dataframe(tribe_path, audio_only=audio_only)
            preds, _segments = self.model.predict(events, verbose=False)
            return VideoPredictionResult(
                frames=np.asarray(preds).tolist(),
                duration_seconds=duration,
            )
        finally:
            if cleanup_dir is not None:
                try:
                    os.unlink(os.path.join(cleanup_dir, "video.mp4"))
                    os.rmdir(cleanup_dir)
                except OSError:
                    pass
            if is_temp:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

    def _preflight_video_impl(
        self,
        video_path_or_url: str,
        *,
        input_kind: str,
        audio_only: bool = False,
    ) -> VideoPreflightResult:
        """Validate path resolution, duration probing, and TRIBE event creation."""
        import time

        _force_container_tmpdir()
        timings: dict[str, float] = {}

        started = time.monotonic()
        bmd_videos_volume.reload()
        timings["volume_reload"] = time.monotonic() - started

        started = time.monotonic()
        local_path, is_temp = _resolve_local_path(video_path_or_url)
        timings["resolve_local_path"] = time.monotonic() - started

        cleanup_dir: str | None = None
        try:
            started = time.monotonic()
            duration = _probe_duration(local_path)
            timings["probe_duration"] = time.monotonic() - started

            started = time.monotonic()
            tribe_path, cleanup_dir = _ensure_tribe_suffix(local_path)
            timings["ensure_suffix"] = time.monotonic() - started

            started = time.monotonic()
            events = self._video_events_dataframe(tribe_path, audio_only=audio_only)
            timings["get_events_dataframe"] = time.monotonic() - started

            return VideoPreflightResult(
                input_kind=input_kind,
                event_mode="audio_only" if audio_only else "full",
                resolved_path=local_path,
                tribe_path=tribe_path,
                exists=os.path.exists(local_path),
                size_bytes=os.path.getsize(local_path)
                if os.path.exists(local_path)
                else None,
                duration_seconds=float(duration),
                events_rows=int(len(events)),
                event_columns=[str(column) for column in events.columns],
                step_seconds=timings,
            )
        finally:
            if cleanup_dir is not None:
                try:
                    os.unlink(os.path.join(cleanup_dir, "video.mp4"))
                    os.rmdir(cleanup_dir)
                except OSError:
                    pass
            if is_temp:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

    @modal.method()
    def predict_video(
        self,
        video_path_or_url: str,
        audio_only: bool = False,
    ) -> VideoPredictionResult:
        """Predict per-vertex brain activations for a video stimulus."""
        return self._predict_video_impl(video_path_or_url, audio_only=audio_only)

    @modal.method()
    def predict_video_bytes(
        self,
        video_bytes: bytes,
        suffix: str = ".mp4",
        audio_only: bool = False,
    ) -> VideoPredictionResult:
        """Predict per-vertex brain activations for an uploaded video payload."""
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir="/tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as out:
                out.write(video_bytes)
            return self._predict_video_impl(tmp, audio_only=audio_only)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    @modal.method()
    def preflight_video(
        self,
        video_path_or_url: str,
        audio_only: bool = False,
    ) -> VideoPreflightResult:
        """Validate a video path/URL without running expensive TRIBE prediction."""
        return self._preflight_video_impl(
            video_path_or_url,
            input_kind="path",
            audio_only=audio_only,
        )

    @modal.method()
    def preflight_video_bytes(
        self,
        video_bytes: bytes,
        suffix: str = ".mp4",
        audio_only: bool = False,
    ) -> VideoPreflightResult:
        """Validate an uploaded video payload without running TRIBE prediction."""
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir="/tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as out:
                out.write(video_bytes)
            return self._preflight_video_impl(
                tmp,
                input_kind="bytes",
                audio_only=audio_only,
            )
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _predict_text_impl(self, text_path_or_url: str) -> VideoPredictionResult:
        """Predict per-vertex brain activations for a text stimulus.

        TRIBE v2 builds text events by converting text to speech and then
        transcribing it into word-level timings. The output shape matches video
        prediction: one brain-vector row per retained TR window.
        """
        import numpy as np  # inline import to keep module-load light

        bmd_videos_volume.reload()
        local_path, is_temp = _resolve_local_path(text_path_or_url)
        cleanup_dir: str | None = None
        try:
            tribe_path, cleanup_dir = _ensure_text_suffix(local_path)
            events = self.model.get_events_dataframe(text_path=tribe_path)
            preds, segments = self.model.predict(events, verbose=False)
            if len(preds) == 0:
                raise ValueError("text produced no retained TRIBE segments")
            duration = 0.0
            for segment in segments:
                start = float(getattr(segment, "offset", 0.0))
                seg_duration = float(getattr(segment, "duration", 0.0))
                duration = max(duration, start + seg_duration)
            if duration <= 0:
                duration = float(len(preds))
            return VideoPredictionResult(
                frames=np.asarray(preds).tolist(),
                duration_seconds=duration,
            )
        finally:
            if cleanup_dir is not None:
                try:
                    os.unlink(os.path.join(cleanup_dir, "stimulus.txt"))
                    os.rmdir(cleanup_dir)
                except OSError:
                    pass
            if is_temp:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

    @modal.method()
    def predict_text(self, text_path_or_url: str) -> VideoPredictionResult:
        """Predict per-vertex brain activations for a text stimulus."""
        return self._predict_text_impl(text_path_or_url)

    @modal.method()
    def predict_video_time_pos_scale(
        self,
        video_path_or_url: str,
        time_pos_scale: float,
    ) -> VideoPredictionResult:
        """Predict after temporarily scaling TRIBE's learned time-position table.

        `time_pos_scale=1.0` is the normal model. `0.0` is a direct ablation of
        the learned temporal positional embedding. The parameter is restored after
        each call, so each request is self-contained.
        """
        import torch  # inline import to keep module-load light

        torch_model = getattr(self.model, "_model", None)
        time_pos_embed = getattr(torch_model, "time_pos_embed", None)
        if time_pos_embed is None:
            raise RuntimeError("TRIBE model has no _model.time_pos_embed parameter")

        with torch.no_grad():
            original = time_pos_embed.detach().clone()
            time_pos_embed.copy_(original * float(time_pos_scale))
        try:
            return self._predict_video_impl(video_path_or_url)
        finally:
            with torch.no_grad():
                time_pos_embed.copy_(original)

    @modal.method()
    def predict_video_hidden_patch(
        self,
        video_path_or_url: str,
        hook_module: str = "_model.encoder",
        patch_mode: str = "none",
        patch_scale: float = 1.0,
        rotary_inv_freq_scale: float = 1.0,
        capture_hidden: bool = False,
    ) -> dict[str, Any]:
        """Predict while optionally patching a hidden state or rotary frequencies.

        The default captures or patches the encoder output, which is downstream of
        the rotary attention stack and upstream of the fMRI prediction head.
        `patch_mode="non_dc_scale", patch_scale=0.0` keeps only the per-sequence
        mean of the hooked tensor. `rotary_inv_freq_scale=0.0` directly removes
        rotary frequency variation for the duration of the call.
        """
        import torch  # inline import to keep module-load light

        torch_model = getattr(self.model, "_model", None)
        if torch_model is None:
            raise RuntimeError("TRIBE wrapper has no private _model torch root")
        module = _lookup_torch_module(torch_model, hook_module)
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = _first_tensor(output)
            if tensor is None:
                return output
            captured["hidden_shape"] = list(tensor.shape)
            captured["hidden_dtype"] = str(tensor.dtype)
            captured["sequence_axis"] = _hidden_sequence_axis(tensor)
            if capture_hidden:
                captured["hidden_npz"] = _compressed_numpy_bytes("hidden", tensor)
            patched = _patch_hidden_tensor(tensor, patch_mode, patch_scale)
            if patched is tensor:
                return output
            return _replace_first_tensor(output, patched)

        rotary_inv_freq = None
        original_rotary_inv_freq = None
        if float(rotary_inv_freq_scale) != 1.0:
            rotary = getattr(getattr(torch_model, "encoder", None), "rotary_pos_emb", None)
            rotary_inv_freq = getattr(rotary, "inv_freq", None)
            if rotary_inv_freq is None:
                raise RuntimeError("TRIBE encoder has no rotary_pos_emb.inv_freq buffer")
            with torch.no_grad():
                original_rotary_inv_freq = rotary_inv_freq.detach().clone()
                rotary_inv_freq.copy_(
                    original_rotary_inv_freq * float(rotary_inv_freq_scale)
                )

        handle = module.register_forward_hook(hook)
        try:
            result = self._predict_video_impl(video_path_or_url)
        finally:
            handle.remove()
            if rotary_inv_freq is not None and original_rotary_inv_freq is not None:
                with torch.no_grad():
                    rotary_inv_freq.copy_(original_rotary_inv_freq)
        if "hidden_shape" not in captured:
            raise RuntimeError(f"hook module {hook_module!r} did not capture a tensor")
        return {
            "frames": result.frames,
            "duration_seconds": result.duration_seconds,
            "hook_module": hook_module,
            "patch_mode": patch_mode,
            "patch_scale": float(patch_scale),
            "rotary_inv_freq_scale": float(rotary_inv_freq_scale),
            **captured,
        }

    @modal.method()
    def capture_video_hiddens(
        self,
        video_path_or_url: str,
        hook_modules: list[str],
    ) -> dict[str, Any]:
        """Predict once while capturing hidden tensors at several modules."""
        torch_model = getattr(self.model, "_model", None)
        if torch_model is None:
            raise RuntimeError("TRIBE wrapper has no private _model torch root")
        captures: dict[str, Any] = {}
        handles = []

        def make_hook(hook_module: str) -> Any:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = _first_tensor(output)
                if tensor is None:
                    return
                captures[hook_module] = {
                    "hidden_shape": list(tensor.shape),
                    "hidden_dtype": str(tensor.dtype),
                    "sequence_axis": _hidden_sequence_axis(tensor),
                    "hidden_npz": _compressed_numpy_bytes("hidden", tensor),
                }

            return hook

        for hook_module in hook_modules:
            module = _lookup_torch_module(torch_model, hook_module)
            handles.append(module.register_forward_hook(make_hook(hook_module)))
        try:
            result = self._predict_video_impl(video_path_or_url)
        finally:
            for handle in handles:
                handle.remove()
        missing = [hook_module for hook_module in hook_modules if hook_module not in captures]
        if missing:
            raise RuntimeError(f"hook modules did not capture tensors: {missing}")
        return {
            "frames": result.frames,
            "duration_seconds": result.duration_seconds,
            "captures": captures,
        }

    @modal.method()
    def predict_video_hidden_direction_patch(
        self,
        video_path_or_url: str,
        hook_module: str,
        direction_npz: bytes,
        patch_alpha: float = 1.0,
    ) -> dict[str, Any]:
        """Predict after removing/amplifying one learned hidden direction.

        `patch_alpha=1.0` removes the projection onto `direction_npz`.
        `patch_alpha=-1.0` amplifies the existing projection by one extra copy.
        """
        torch_model = getattr(self.model, "_model", None)
        if torch_model is None:
            raise RuntimeError("TRIBE wrapper has no private _model torch root")
        module = _lookup_torch_module(torch_model, hook_module)
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = _first_tensor(output)
            if tensor is None:
                return output
            captured["hidden_shape"] = list(tensor.shape)
            captured["hidden_dtype"] = str(tensor.dtype)
            captured["sequence_axis"] = _hidden_sequence_axis(tensor)
            patched = _patch_hidden_direction_tensor(tensor, direction_npz, patch_alpha)
            return _replace_first_tensor(output, patched)

        handle = module.register_forward_hook(hook)
        try:
            result = self._predict_video_impl(video_path_or_url)
        finally:
            handle.remove()
        if "hidden_shape" not in captured:
            raise RuntimeError(f"hook module {hook_module!r} did not capture a tensor")
        return {
            "frames": result.frames,
            "duration_seconds": result.duration_seconds,
            "hook_module": hook_module,
            "patch_alpha": float(patch_alpha),
            **captured,
        }
