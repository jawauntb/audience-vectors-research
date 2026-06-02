"""Feasibility report for generation-model patching and steering.

This is intentionally lightweight: it does not load SVD, CogVideoX, or
AnimateDiff checkpoints. Instead it inspects the repo's existing adapter
artifacts, local generated-video features, and dependency/cache state, then
writes a JSON + Markdown report describing which generator patch path is
actionable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = ROOT / "data/reports/generation_patch_feasibility.json"
DEFAULT_MD_OUT = ROOT / "data/reports/generation_patch_feasibility.md"

ALPHA_SETS = {
    "svd_smoke_large": {
        "large_a0": -10.0,
        "large_a1": -5.0,
        "large_a2": 0.0,
        "large_a3": 5.0,
        "large_a4": 10.0,
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def artifact(path: Path) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {"path": rel(path), "exists": exists}
    if exists and path.is_file():
        payload["bytes"] = path.stat().st_size
    return payload


def module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False}
    try:
        module = __import__(name)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    return {"available": True, "version": getattr(module, "__version__", None)}


def torch_status() -> dict[str, Any]:
    status = module_status("torch")
    if not status["available"]:
        return status
    import torch  # noqa: PLC0415

    status.update(
        {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
        }
    )
    return status


def cached_hf_models() -> list[str]:
    hub = Path.home() / ".cache/huggingface/hub"
    if not hub.exists():
        return []
    models = []
    for path in sorted(hub.glob("models--*")):
        name = path.name.replace("models--", "").replace("--", "/")
        if any(
            token in name.lower()
            for token in (
                "svd",
                "stable-video",
                "cogvideo",
                "animatediff",
                "clip-vit-h",
                "t5",
            )
        ):
            models.append(name)
    return models


def to_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def vector_summary(vector: np.ndarray) -> dict[str, Any]:
    return {
        "dim": int(vector.size),
        "norm": float(np.linalg.norm(vector)),
        "finite": bool(np.isfinite(vector).all()),
    }


def load_checkpoint(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        import torch  # noqa: PLC0415

        return torch.load(path, weights_only=False, map_location="cpu"), None
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)


def load_direction_artifacts() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    specs = {
        "svd_clip_h": {
            "path": ROOT / "data/reports/adapter_tribe_to_clip_h.pt",
            "keys": ("v_mem_clip_h_native", "v_mem_clip_h_via_adapter"),
            "preferred_key": "v_mem_clip_h_via_adapter",
            "expected_dim": 1024,
            "space": "CLIP-ViT-H image projection",
        },
        "cogvideox_t5": {
            "path": ROOT / "data/reports/adapter_tribe_to_t5.pt",
            "keys": ("v_mem_t5_native", "v_mem_t5_via_adapter"),
            "preferred_key": "v_mem_t5_native",
            "expected_dim": 4096,
            "space": "CogVideoX T5-XXL prompt embedding",
        },
        "clip_vit_l_image": {
            "path": ROOT / "data/reports/adapter_tribe_to_clip.pt",
            "keys": ("v_mem_clip_native", "v_mem_clip_via_adapter"),
            "preferred_key": "v_mem_clip_via_adapter",
            "expected_dim": 768,
            "space": "CLIP-ViT-L image projection",
        },
    }
    summaries: dict[str, Any] = {}
    vectors: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        ckpt, error = load_checkpoint(spec["path"])
        summary: dict[str, Any] = {
            "artifact": artifact(spec["path"]),
            "space": spec["space"],
            "expected_dim": spec["expected_dim"],
            "preferred_key": spec["preferred_key"],
        }
        if error is not None:
            summary["load_error"] = error
            summaries[name] = summary
            continue
        assert ckpt is not None
        summary["encoder"] = ckpt.get("encoder")
        summary["cos_alignment"] = _optional_float(ckpt.get("cos_alignment"))
        summary["vectors"] = {}
        for key in spec["keys"]:
            if key not in ckpt:
                continue
            vector = to_array(ckpt[key])
            summary["vectors"][key] = vector_summary(vector)
            vectors[f"{name}:{key}"] = vector
        summaries[name] = summary
    return summaries, vectors


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_feature(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    key = "frames" if "frames" in payload.files else "embedding"
    arr = np.asarray(payload[key], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr.reshape(-1)


def load_bmd_scores() -> dict[str, float]:
    path = ROOT / "data/raw/bold_moments/annotations.json"
    if not path.exists():
        return {}
    annotations = json.loads(path.read_text())
    return {
        f"bmd_vid_idx{entry_id}": float(entry["memorability_score"])
        for entry_id, entry in annotations.items()
        if "memorability_score" in entry
    }


def train_tribe_v_mem() -> tuple[np.ndarray | None, dict[str, Any]]:
    feature_dir = ROOT / "data/features/tribe"
    scores_by_video = load_bmd_scores()
    rows: list[tuple[np.ndarray, float]] = []
    for path in sorted(feature_dir.glob("bmd_vid_idx*.npz")):
        video_id = path.stem.split("_seg_")[0]
        score = scores_by_video.get(video_id)
        if score is None:
            continue
        rows.append((load_feature(path), score))
    if not rows:
        return None, {"available": False, "reason": "no scored TRIBE features found"}
    features = np.stack([row[0] for row in rows]).astype(np.float32)
    scores = np.asarray([row[1] for row in rows], dtype=np.float32)
    order = np.argsort(scores)
    n_each = max(3, int(len(scores) * 0.30))
    direction = features[order[-n_each:]].mean(axis=0) - features[order[:n_each]].mean(
        axis=0
    )
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return None, {"available": False, "reason": "zero-norm TRIBE direction"}
    direction = direction / norm
    return direction.astype(np.float32), {
        "available": True,
        "feature_dir": rel(feature_dir),
        "n_features": int(len(features)),
        "dim": int(features.shape[1]),
        "top_bottom_frac": 0.30,
        "norm": float(np.linalg.norm(direction)),
    }


def rank_ordinal(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(np.float64)


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    a = rank_ordinal(np.asarray(values_a, dtype=np.float64))
    b = rank_ordinal(np.asarray(values_b, dtype=np.float64))
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    rho = float(np.corrcoef(a, b)[0, 1])
    return rho if math.isfinite(rho) else None


def load_manifest_alpha_map(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    if not path.exists():
        return {}, {}
    rows = json.loads(path.read_text())
    alpha_by_label = {}
    group_by_label = {}
    for row in rows:
        label = str(row["label"])
        alpha_by_label[label] = float(row["alpha"])
        group_by_label[label] = str(row.get("clip") or row.get("prompt") or "all")
    return alpha_by_label, group_by_label


def projection_sweep(
    *,
    name: str,
    generated_dir: Path,
    features_dir: Path,
    alpha_by_label: dict[str, float],
    group_by_label: dict[str, str],
    v_mem: np.ndarray | None,
) -> dict[str, Any]:
    labels = sorted(alpha_by_label)
    rows = []
    missing_features = []
    missing_videos = []
    for label in labels:
        video = generated_dir / f"{label}.mp4"
        feature = features_dir / f"{label}.npz"
        if not video.exists():
            missing_videos.append(label)
        if not feature.exists():
            missing_features.append(label)
            continue
        projection = None
        if v_mem is not None:
            projection = float(load_feature(feature) @ v_mem)
        rows.append(
            {
                "label": label,
                "group": group_by_label.get(label, "all"),
                "alpha": alpha_by_label[label],
                "projection": projection,
            }
        )
    summary: dict[str, Any] = {
        "name": name,
        "generated_dir": rel(generated_dir),
        "features_dir": rel(features_dir),
        "n_manifest": len(labels),
        "n_rows": len(rows),
        "missing_features": missing_features,
        "missing_videos": missing_videos,
        "rows": rows,
    }
    if not rows or v_mem is None:
        return summary
    summary.update(score_projection_rows(rows))
    return summary


def score_projection_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    alpha = [float(row["alpha"]) for row in rows]
    proj = [float(row["projection"]) for row in rows if row["projection"] is not None]
    overall = spearman(alpha, proj) if len(alpha) == len(proj) else None
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)
    group_scores = []
    for group, group_rows in sorted(by_group.items()):
        group_rows = [row for row in group_rows if row["projection"] is not None]
        if len(group_rows) < 3:
            continue
        group_rows = sorted(group_rows, key=lambda row: float(row["alpha"]))
        group_alpha = [float(row["alpha"]) for row in group_rows]
        group_proj = [float(row["projection"]) for row in group_rows]
        group_scores.append(
            {
                "group": group,
                "n": len(group_rows),
                "spearman": spearman(group_alpha, group_proj),
                "delta_max_minus_min": float(group_proj[-1] - group_proj[0]),
            }
        )
    numeric_scores = [
        row["spearman"] for row in group_scores if row["spearman"] is not None
    ]
    positive_scores = [rho for rho in numeric_scores if rho > 0]
    return {
        "spearman": overall,
        "per_group": group_scores,
        "mean_group_spearman": float(np.mean(numeric_scores))
        if numeric_scores
        else None,
        "positive_group_pct": float(100 * len(positive_scores) / len(numeric_scores))
        if numeric_scores
        else None,
    }


def run_projection_dry_runs(v_mem: np.ndarray | None) -> list[dict[str, Any]]:
    specs = [
        (
            "svd_smoke",
            ROOT / "data/generated/svd_smoke",
            ROOT / "data/features/tribe_svd_smoke",
            ROOT / "data/generated/svd_smoke/manifest.json",
        ),
        (
            "svd_sweep",
            ROOT / "data/generated/svd_sweep",
            ROOT / "data/features/tribe_svd_sweep",
            ROOT / "data/generated/svd_sweep/manifest.json",
        ),
        (
            "cogvideox_smoke_native",
            ROOT / "data/generated/cogvideox_smoke",
            ROOT / "data/features/tribe_cogvideox_smoke",
            ROOT / "data/generated/cogvideox_smoke/manifest.json",
        ),
        (
            "cogvideox_smoke_adapter",
            ROOT / "data/generated/cogvideox_smoke_adapter",
            ROOT / "data/features/tribe_cogvideox_smoke_adapter",
            ROOT / "data/generated/cogvideox_smoke_adapter/manifest.json",
        ),
    ]
    out = []
    for name, generated_dir, features_dir, manifest in specs:
        alpha_by_label, group_by_label = load_manifest_alpha_map(manifest)
        out.append(
            projection_sweep(
                name=name,
                generated_dir=generated_dir,
                features_dir=features_dir,
                alpha_by_label=alpha_by_label,
                group_by_label=group_by_label,
                v_mem=v_mem,
            )
        )
    labels = ALPHA_SETS["svd_smoke_large"]
    out.append(
        projection_sweep(
            name="svd_smoke_large",
            generated_dir=ROOT / "data/generated/svd_smoke_large",
            features_dir=ROOT / "data/features/tribe_svd_smoke_large",
            alpha_by_label=labels,
            group_by_label={label: "all" for label in labels},
            v_mem=v_mem,
        )
    )
    return out


def embedding_patch_probe(
    *,
    name: str,
    embedding_path: Path,
    vector: np.ndarray | None,
    vector_key: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "embedding_artifact": artifact(embedding_path),
        "vector_key": vector_key,
    }
    if vector is None:
        payload["status"] = "missing_vector"
        return payload
    if not embedding_path.exists():
        payload["status"] = "missing_embeddings"
        return payload
    data = np.load(embedding_path, allow_pickle=False)
    if "embeddings" not in data.files:
        payload["status"] = "missing_embeddings_key"
        return payload
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    sample = embeddings[0].reshape(-1)
    payload["embedding_shape"] = list(embeddings.shape)
    payload["sample_norm"] = float(np.linalg.norm(sample))
    if sample.shape[-1] != vector.shape[-1]:
        payload["status"] = "dim_mismatch"
        payload["vector_dim"] = int(vector.shape[-1])
        payload["embedding_dim"] = int(sample.shape[-1])
        return payload
    v_norm = vector / (np.linalg.norm(vector) + 1e-12)
    checks = {}
    for alpha in (-2.0, 0.0, 2.0):
        patched = sample + alpha * v_norm
        checks[f"alpha_{alpha:+.1f}"] = {
            "patched_norm": float(np.linalg.norm(patched)),
            "shift_norm": float(np.linalg.norm(patched - sample)),
            "finite": bool(np.isfinite(patched).all()),
        }
    payload["status"] = "ok"
    payload["checks"] = checks
    return payload


def build_environment() -> dict[str, Any]:
    modules = {
        name: module_status(name)
        for name in ("diffusers", "accelerate", "transformers", "modal", "imageio")
    }
    return {
        "torch": torch_status(),
        "modules": modules,
        "hf_cached_generation_models": cached_hf_models(),
    }


def patch_points() -> list[dict[str, Any]]:
    return [
        {
            "model": "Stable Video Diffusion XT",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "local_status": "implemented in Modal wrapper; local generation not feasible without diffusers, accelerate, CUDA, and weights",
            "source": "src/audience_vectors/modal_app/functions/svd_generator.py",
            "patch_point": "StableVideoDiffusionPipeline._encode_image output before U-Net cross-attention",
            "activation_shape": "(batch, 1, 1024)",
            "direction": "v_mem_clip_h_via_adapter from adapter_tribe_to_clip_h.pt",
            "operation": "image_embedding = image_embedding + alpha * normalize(v_mem_clip_h)",
            "connection_to_v_mem": "TRIBE top-vs-bottom memorability direction is mapped into CLIP-ViT-H image space by the trained TRIBE-to-CLIP-H adapter.",
            "feasibility": "best immediate path",
        },
        {
            "model": "CogVideoX-5B",
            "repo_id": "THUDM/CogVideoX-5b",
            "local_status": "implemented in Modal wrapper; local generation not feasible without diffusers, accelerate, CUDA, and weights",
            "source": "src/audience_vectors/modal_app/functions/cogvideox_generator.py",
            "patch_point": "CogVideoXPipeline.encode_prompt prompt_embeds before diffusion transformer",
            "activation_shape": "(batch, sequence, 4096)",
            "direction": "v_mem_t5_native or v_mem_t5_via_adapter from adapter_tribe_to_t5.pt",
            "operation": "prompt_embeds = prompt_embeds + alpha * normalize(v_mem_t5)",
            "connection_to_v_mem": "BMD captions were encoded with CogVideoX T5-XXL and paired with TRIBE features; native and adapter-derived T5 directions are available.",
            "feasibility": "actionable but needs broader sweep; current one-prompt smoke is not convincing",
        },
        {
            "model": "AnimateDiff",
            "repo_id": "typical stack: SD 1.5 base plus AnimateDiff motion adapter",
            "local_status": "no repo wrapper, no cached AnimateDiff/SD weights, and no diffusers/accelerate install",
            "source": "not present",
            "patch_point": "for text-to-video, patch CLIP text prompt hidden states after encode_prompt; for image-conditioned variants, patch IP-Adapter/CLIP image conditioning",
            "activation_shape": "SD1.5 text hidden states are usually (batch, sequence, 768); IP-Adapter image embeddings vary by adapter",
            "direction": "not ready; existing CLIP-ViT-L direction is image projection space, not SD CLIP text hidden-state space",
            "operation": "prompt_embeds or image_embeds += alpha * compatible v_mem direction",
            "connection_to_v_mem": "would require either a TRIBE-to-SD-CLIP-text adapter from captions or an image-conditioned AnimateDiff/IP-Adapter path compatible with the existing CLIP image direction.",
            "feasibility": "not immediate",
        },
    ]


def build_report() -> dict[str, Any]:
    directions, vectors = load_direction_artifacts()
    v_mem, v_mem_summary = train_tribe_v_mem()
    projection_runs = run_projection_dry_runs(v_mem)
    embedding_runs = [
        embedding_patch_probe(
            name="svd_clip_h_embedding_patch",
            embedding_path=ROOT / "data/features/clip_image_h_embeddings.npz",
            vector=vectors.get("svd_clip_h:v_mem_clip_h_via_adapter"),
            vector_key="svd_clip_h:v_mem_clip_h_via_adapter",
        ),
        embedding_patch_probe(
            name="cogvideox_t5_native_embedding_patch",
            embedding_path=ROOT / "data/features/t5xxl_captions.npz",
            vector=vectors.get("cogvideox_t5:v_mem_t5_native"),
            vector_key="cogvideox_t5:v_mem_t5_native",
        ),
        embedding_patch_probe(
            name="clip_vit_l_image_embedding_patch",
            embedding_path=ROOT / "data/features/clip_image_embeddings.npz",
            vector=vectors.get("clip_vit_l_image:v_mem_clip_via_adapter"),
            vector_key="clip_vit_l_image:v_mem_clip_via_adapter",
        ),
    ]
    return {
        "conclusion": {
            "best_immediate_path": "SVD conditioning-space steering via CLIP-ViT-H image embeddings.",
            "local_generation_feasible": False,
            "local_dry_run_feasible": True,
            "why_not_local_generation": [
                "CUDA is unavailable locally.",
                "diffusers and accelerate are not installed in the local uv environment.",
                "SVD, CogVideoX, and AnimateDiff generation checkpoints are not cached locally; existing wrappers are designed for Modal GPU volumes.",
            ],
            "next_generation_step": "Run an SVD Modal mini-sweep with the existing SVDGenerator patch, then score outputs with cached TRIBE v_mem projections.",
        },
        "environment": build_environment(),
        "direction_artifacts": directions,
        "tribe_v_mem": v_mem_summary,
        "patch_points": patch_points(),
        "embedding_patch_dry_runs": embedding_runs,
        "generated_video_projection_dry_runs": projection_runs,
    }


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Generation-model patching feasibility",
        "",
        "## Conclusion",
        "",
        f"- Best immediate path: {report['conclusion']['best_immediate_path']}",
        f"- Local generation feasible: {report['conclusion']['local_generation_feasible']}",
        f"- Local dry-run feasible: {report['conclusion']['local_dry_run_feasible']}",
        f"- Next generation step: {report['conclusion']['next_generation_step']}",
        "",
        "Local generation is blocked by:",
    ]
    lines.extend(
        f"- {reason}" for reason in report["conclusion"]["why_not_local_generation"]
    )
    lines.extend(["", "## Patch Points", ""])
    for point in report["patch_points"]:
        lines.extend(
            [
                f"### {point['model']}",
                "",
                f"- Feasibility: {point['feasibility']}",
                f"- Source: `{point['source']}`",
                f"- Patch point: {point['patch_point']}",
                f"- Activation shape: `{point['activation_shape']}`",
                f"- Direction: {point['direction']}",
                f"- Operation: `{point['operation']}`",
                f"- v_mem connection: {point['connection_to_v_mem']}",
                "",
            ]
        )
    lines.extend(["## Local Environment", ""])
    env = report["environment"]
    torch = env["torch"]
    lines.extend(
        [
            f"- torch: {torch.get('version')} (cuda={torch.get('cuda_available')}, mps={torch.get('mps_available')})",
            f"- diffusers: {env['modules']['diffusers'].get('available')}",
            f"- accelerate: {env['modules']['accelerate'].get('available')}",
            f"- cached generation-related HF models: {', '.join(env['hf_cached_generation_models']) or 'none'}",
            "",
            "## Direction Artifacts",
            "",
            "| Artifact | Preferred vector | Dim | Norm | Cos alignment |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, summary in report["direction_artifacts"].items():
        key = summary.get("preferred_key")
        vector = summary.get("vectors", {}).get(key, {})
        lines.append(
            f"| {name} | `{key}` | {vector.get('dim', 'n/a')} | "
            f"{fmt_float(vector.get('norm'), 4)} | {fmt_float(summary.get('cos_alignment'), 4)} |"
        )
    lines.extend(["", "## Embedding Patch Dry Runs", ""])
    lines.extend(["| Probe | Status | Embedding shape |", "|---|---:|---:|"])
    for probe in report["embedding_patch_dry_runs"]:
        shape = probe.get("embedding_shape", "n/a")
        lines.append(f"| {probe['name']} | {probe.get('status')} | `{shape}` |")
    lines.extend(["", "## Cached Generated-Video Dry Runs", ""])
    lines.extend(
        [
            "| Run | Rows | Spearman(alpha, v_mem projection) | Mean group rho | Positive groups | Missing features |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report["generated_video_projection_dry_runs"]:
        lines.append(
            f"| {run['name']} | {run['n_rows']}/{run['n_manifest']} | "
            f"{fmt_float(run.get('spearman'))} | {fmt_float(run.get('mean_group_spearman'))} | "
            f"{fmt_float(run.get('positive_group_pct'), 1)}% | {len(run.get('missing_features', []))} |"
        )
    lines.extend(
        [
            "",
            "## Concrete Recommendation",
            "",
            "Use SVD first. The repo already has the exact conditioning patch, a compatible 1024-dim CLIP-H v_mem direction, cached generated videos, and cached TRIBE features for dry-run scoring. CogVideoX is also patched, but the available one-prompt smoke is weak and should be treated as exploratory until a broader prompt/seed sweep is run. AnimateDiff needs new model plumbing and a compatible text- or image-conditioning direction before it is a serious follow-up.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    report = build_report()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {rel(args.json_out)}")
    print(f"wrote {rel(args.md_out)}")


if __name__ == "__main__":
    main()
