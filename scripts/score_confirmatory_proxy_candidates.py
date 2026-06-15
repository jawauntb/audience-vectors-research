"""Score confirmatory Seedance candidates with TRIBE, V-JEPA, and CLIP.

The input manifest freezes exact MP4 byte targets. This script attaches proxy
scores to those exact bytes using bounded parallel Modal calls for TRIBE/V-JEPA
and local CLIP prompt-video alignment. It is resumable through local feature
caches and writes a scored manifest plus a markdown selection audit.

This is still proxy selection, not human memorability evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from audience_vectors.services.tribe_service import TribeService, TribeValidationError
from audience_vectors.services.vjepa_service import VjepaService, VjepaValidationError

EXPERIMENT_DIR = Path(
    "research_program/neurips_memorability_selector/experiments/"
    "content_pocket_confirmatory_recognition_20260615"
)
DEFAULT_INPUT_MANIFEST = (
    EXPERIMENT_DIR / "seedance_candidate_proxy_scoring_manifest_improved_v1_20260615.json"
)
DEFAULT_OUT_JSON = (
    EXPERIMENT_DIR / "seedance_candidate_proxy_scores_improved_v1_20260615.json"
)
DEFAULT_OUT_MD = (
    EXPERIMENT_DIR / "seedance_candidate_proxy_scores_improved_v1_20260615.md"
)
DEFAULT_FEATURE_DIR = Path(
    "data/features/content_pocket_confirmatory_recognition_20260615/"
    "proxy_scores_improved_v1"
)
DEFAULT_TRIBE_VMEM = Path(
    "/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz"
)
DEFAULT_VJEPA_DIRECTION = Path(
    "/Users/jawaun/isc_mod/data/models/vectors/"
    "facebook__vjepa2-vitl-fpc64-256__vjepa_mean_pool__bmd_memorability_n1026.npz"
)

POSITIVE_ROLES = {"accepted_positive_anchor", "boundary_positive"}
NEGATIVE_ROLES = {"previous_hard_negative"}

FloatArray = NDArray[np.float32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--tribe-vmem", type=Path, default=DEFAULT_TRIBE_VMEM)
    parser.add_argument("--vjepa-direction", type=Path, default=DEFAULT_VJEPA_DIRECTION)
    parser.add_argument("--tribe-app-name", default=None)
    parser.add_argument("--vjepa-app-name", default=None)
    parser.add_argument("--tribe-concurrency", type=int, default=3)
    parser.add_argument("--vjepa-concurrency", type=int, default=4)
    parser.add_argument("--tribe-timeout", type=float, default=600.0)
    parser.add_argument("--vjepa-timeout", type=float, default=420.0)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-frames", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-tribe", action="store_true")
    parser.add_argument("--skip-vjepa", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_direction(path: Path) -> FloatArray:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=False)
    if "direction" not in payload:
        raise KeyError(f"{path} has no 'direction' array")
    direction = np.asarray(payload["direction"], dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError(f"zero-norm direction in {path}")
    return direction / norm


def npz_vector(path: Path, key: str) -> FloatArray | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        payload = np.load(path, allow_pickle=False)
        return np.asarray(payload[key], dtype=np.float32).reshape(-1)
    except (KeyError, OSError, ValueError):
        return None


def zscore_by_family(
    rows: list[dict[str, Any]],
    key: str,
    out_key: str,
) -> None:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None and math.isfinite(float(row[key])):
            by_family[str(row["family_id"])].append(row)
    for family_rows in by_family.values():
        values = np.asarray([float(row[key]) for row in family_rows], dtype=np.float32)
        std = float(values.std())
        if std < 1e-8:
            z = np.zeros_like(values)
        else:
            z = (values - float(values.mean())) / std
        for row, value in zip(family_rows, z):
            row[out_key] = float(value)


def candidate_cache_path(feature_dir: Path, subdir: str, job_id: str) -> Path:
    return feature_dir / subdir / f"{job_id}.npz"


def local_path(row: dict[str, Any]) -> Path:
    path = Path(str(row["source_absolute_path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_frames(path: Path, n_frames: int) -> list[Image.Image]:
    all_frames = [np.asarray(frame) for frame in iio.imiter(path)]
    if not all_frames:
        raise ValueError(f"no frames found in {path}")
    indices = np.linspace(0, len(all_frames) - 1, num=n_frames, dtype=int)
    return [Image.fromarray(all_frames[int(index)]).convert("RGB") for index in indices]


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalized_np(tensor: torch.Tensor) -> FloatArray:
    arr = tensor.detach().float().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


async def score_tribe_one(
    *,
    row: dict[str, Any],
    service: TribeService,
    semaphore: asyncio.Semaphore,
    out_path: Path,
    direction: FloatArray,
    timeout: float,
    force: bool,
) -> dict[str, Any]:
    job_id = str(row["job_id"])
    cached = npz_vector(out_path, "mean_feature")
    if cached is not None and not force:
        return {
            "job_id": job_id,
            "status": "cached",
            "feature_path": str(out_path),
            "tribe_bmd_projection": float(np.dot(cached, direction)),
        }
    video_path = local_path(row)
    async with semaphore:
        try:
            result = await asyncio.wait_for(
                service.predict_video_bytes(
                    video_path.read_bytes(),
                    suffix=video_path.suffix or ".mp4",
                ),
                timeout=timeout,
            )
        except TribeValidationError as exc:
            return {"job_id": job_id, "status": "rejected", "error": str(exc)}
        except TimeoutError as exc:
            return {"job_id": job_id, "status": "timeout", "error": repr(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"job_id": job_id, "status": "failed", "error": repr(exc)}
    if result is None or getattr(result, "frames", None) is None:
        return {"job_id": job_id, "status": "failed", "error": "empty TRIBE result"}
    frames = np.asarray(result.frames, dtype=np.float32)
    mean_feature = frames.mean(axis=0).reshape(-1).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mean_feature=mean_feature,
        frames_shape=np.asarray(frames.shape, dtype=np.int32),
        duration_seconds=np.asarray(float(result.duration_seconds), dtype=np.float32),
        job_id=np.asarray(job_id),
        source_absolute_path=np.asarray(str(video_path)),
    )
    return {
        "job_id": job_id,
        "status": "written",
        "feature_path": str(out_path),
        "tribe_bmd_projection": float(np.dot(mean_feature, direction)),
        "tribe_frames_shape": list(frames.shape),
        "tribe_duration_seconds": float(result.duration_seconds),
    }


async def score_vjepa_one(
    *,
    row: dict[str, Any],
    service: VjepaService,
    semaphore: asyncio.Semaphore,
    out_path: Path,
    direction: FloatArray,
    timeout: float,
    force: bool,
) -> dict[str, Any]:
    job_id = str(row["job_id"])
    cached = npz_vector(out_path, "embedding")
    if cached is not None and not force:
        return {
            "job_id": job_id,
            "status": "cached",
            "feature_path": str(out_path),
            "vjepa_bmd_projection": float(np.dot(cached, direction)),
        }
    video_path = local_path(row)
    async with semaphore:
        try:
            result = await asyncio.wait_for(
                service.predict_video_bytes(
                    video_path.read_bytes(),
                    suffix=video_path.suffix or ".mp4",
                ),
                timeout=timeout,
            )
        except VjepaValidationError as exc:
            return {"job_id": job_id, "status": "rejected", "error": str(exc)}
        except TimeoutError as exc:
            return {"job_id": job_id, "status": "timeout", "error": repr(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"job_id": job_id, "status": "failed", "error": repr(exc)}
    if result is None:
        return {"job_id": job_id, "status": "failed", "error": "empty V-JEPA result"}
    if hasattr(result, "embedding"):
        embedding = np.asarray(result.embedding, dtype=np.float32).reshape(-1)
        duration = float(result.duration_seconds)
        n_frames = int(result.n_frames)
    else:
        embedding = np.asarray(result["embedding"], dtype=np.float32).reshape(-1)
        duration = float(result["duration_seconds"])
        n_frames = int(result["n_frames"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embedding=embedding,
        duration_seconds=np.asarray(duration, dtype=np.float32),
        n_frames=np.asarray(n_frames, dtype=np.int32),
        job_id=np.asarray(job_id),
        source_absolute_path=np.asarray(str(video_path)),
    )
    return {
        "job_id": job_id,
        "status": "written",
        "feature_path": str(out_path),
        "vjepa_bmd_projection": float(np.dot(embedding, direction)),
        "vjepa_duration_seconds": duration,
        "vjepa_n_frames": n_frames,
    }


async def score_tribe(
    rows: list[dict[str, Any]],
    *,
    app_name: str | None,
    feature_dir: Path,
    direction: FloatArray,
    concurrency: int,
    timeout: float,
    force: bool,
) -> list[dict[str, Any]]:
    service = TribeService(app_name=app_name)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        score_tribe_one(
            row=row,
            service=service,
            semaphore=semaphore,
            out_path=candidate_cache_path(feature_dir, "tribe", str(row["job_id"])),
            direction=direction,
            timeout=timeout,
            force=force,
        )
        for row in rows
    ]
    return await asyncio.gather(*tasks)


async def score_vjepa(
    rows: list[dict[str, Any]],
    *,
    app_name: str | None,
    feature_dir: Path,
    direction: FloatArray,
    concurrency: int,
    timeout: float,
    force: bool,
) -> list[dict[str, Any]]:
    service = VjepaService(app_name=app_name)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        score_vjepa_one(
            row=row,
            service=service,
            semaphore=semaphore,
            out_path=candidate_cache_path(feature_dir, "vjepa", str(row["job_id"])),
            direction=direction,
            timeout=timeout,
            force=force,
        )
        for row in rows
    ]
    return await asyncio.gather(*tasks)


class ClipPromptVideoScorer:
    def __init__(self, model_id: str) -> None:
        self.device = choose_device()
        print(f"[clip] loading {model_id} on {self.device}", flush=True)
        self.processor: Any = CLIPProcessor.from_pretrained(model_id)
        model: Any = CLIPModel.from_pretrained(model_id)
        self.model: Any = model.to(self.device).eval()
        self.text_cache: dict[str, FloatArray] = {}

    def text_embedding(self, text: str) -> FloatArray:
        if text in self.text_cache:
            return self.text_cache[text]
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        out = normalized_np(embedding)[0]
        self.text_cache[text] = out
        return out

    def video_embedding(self, path: Path, n_frames: int) -> tuple[FloatArray, int]:
        frames = load_frames(path, n_frames=n_frames)
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            frame_embeddings = self.model.get_image_features(**inputs)
        frame_np = normalized_np(frame_embeddings)
        mean_embedding = frame_np.mean(axis=0)
        mean_embedding /= max(float(np.linalg.norm(mean_embedding)), 1e-12)
        return mean_embedding.astype(np.float32), len(frames)


def score_clip_one(
    *,
    row: dict[str, Any],
    scorer: ClipPromptVideoScorer,
    out_path: Path,
    n_frames: int,
    force: bool,
) -> dict[str, Any]:
    job_id = str(row["job_id"])
    cached = npz_vector(out_path, "video_embedding")
    if cached is not None and not force:
        cached_payload = np.load(out_path, allow_pickle=False)
        sampled_frames = int(np.asarray(cached_payload["sampled_frames"]).reshape(-1)[0])
        if sampled_frames == n_frames:
            text_embedding = scorer.text_embedding(str(row["prompt"]))
            return {
                "job_id": job_id,
                "status": "cached",
                "feature_path": str(out_path),
                "clip_prompt_video_alignment": float(np.dot(cached, text_embedding)),
            }
    video_path = local_path(row)
    try:
        video_embedding, sampled_frames = scorer.video_embedding(video_path, n_frames)
        text_embedding = scorer.text_embedding(str(row["prompt"]))
    except Exception as exc:  # noqa: BLE001
        return {"job_id": job_id, "status": "failed", "error": repr(exc)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        video_embedding=video_embedding,
        sampled_frames=np.asarray(sampled_frames, dtype=np.int32),
        job_id=np.asarray(job_id),
        source_absolute_path=np.asarray(str(video_path)),
        clip_model=np.asarray(scorer.model.config.name_or_path),
    )
    return {
        "job_id": job_id,
        "status": "written",
        "feature_path": str(out_path),
        "clip_prompt_video_alignment": float(np.dot(video_embedding, text_embedding)),
        "clip_sampled_frames": sampled_frames,
    }


def score_clip(
    rows: list[dict[str, Any]],
    *,
    feature_dir: Path,
    model_id: str,
    n_frames: int,
    force: bool,
) -> list[dict[str, Any]]:
    scorer = ClipPromptVideoScorer(model_id)
    out = []
    for index, row in enumerate(rows, start=1):
        print(f"[clip] {index}/{len(rows)} {row['job_id']}", flush=True)
        out.append(
            score_clip_one(
                row=row,
                scorer=scorer,
                out_path=candidate_cache_path(feature_dir, "clip_video", str(row["job_id"])),
                n_frames=n_frames,
                force=force,
            )
        )
    return out


def attach_model_results(
    rows: list[dict[str, Any]],
    *,
    tribe_results: list[dict[str, Any]] | None,
    vjepa_results: list[dict[str, Any]] | None,
    clip_results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    by_job = {str(row["job_id"]): row for row in rows}
    model_status: dict[str, dict[str, int]] = {}

    def attach(results: list[dict[str, Any]] | None, model: str) -> None:
        if results is None:
            model_status[model] = {"skipped": len(rows)}
            return
        counts = Counter(str(result["status"]) for result in results)
        model_status[model] = dict(sorted(counts.items()))
        for result in results:
            row = by_job[str(result["job_id"])]
            scores = row["proxy_scores"]
            if model == "tribe":
                scores["tribe_status"] = result["status"]
                scores["tribe_feature_path"] = result.get("feature_path")
                scores["tribe_bmd_projection"] = result.get("tribe_bmd_projection")
                if result.get("error"):
                    scores["tribe_error"] = result["error"]
            elif model == "vjepa":
                scores["vjepa_status"] = result["status"]
                scores["vjepa_feature_path"] = result.get("feature_path")
                scores["vjepa_bmd_projection"] = result.get("vjepa_bmd_projection")
                if result.get("error"):
                    scores["vjepa_error"] = result["error"]
            elif model == "clip":
                scores["clip_status"] = result["status"]
                scores["clip_video_feature_path"] = result.get("feature_path")
                scores["clip_prompt_video_alignment"] = result.get(
                    "clip_prompt_video_alignment"
                )
                # Back-compat slot from the intake manifest. These text-to-video
                # candidates have no seed image, so CLIP prompt-video alignment is
                # the preservation proxy.
                scores["clip_seed_video_preservation"] = result.get(
                    "clip_prompt_video_alignment"
                )
                if result.get("error"):
                    scores["clip_error"] = result["error"]

    attach(tribe_results, "tribe")
    attach(vjepa_results, "vjepa")
    attach(clip_results, "clip")
    return model_status


def compute_vjepa_centroid_margins(rows: list[dict[str, Any]], feature_dir: Path) -> None:
    embeddings: dict[str, FloatArray] = {}
    for row in rows:
        path = candidate_cache_path(feature_dir, "vjepa", str(row["job_id"]))
        embedding = npz_vector(path, "embedding")
        if embedding is not None:
            norm = max(float(np.linalg.norm(embedding)), 1e-12)
            embeddings[str(row["job_id"])] = embedding / norm

    positives = [
        embeddings[str(row["job_id"])]
        for row in rows
        if row.get("prior_role") in POSITIVE_ROLES and str(row["job_id"]) in embeddings
    ]
    negatives = [
        embeddings[str(row["job_id"])]
        for row in rows
        if row.get("prior_role") in NEGATIVE_ROLES and str(row["job_id"]) in embeddings
    ]
    if not positives or not negatives:
        return
    positive_centroid = np.mean(np.stack(positives), axis=0)
    positive_centroid /= max(float(np.linalg.norm(positive_centroid)), 1e-12)
    negative_centroid = np.mean(np.stack(negatives), axis=0)
    negative_centroid /= max(float(np.linalg.norm(negative_centroid)), 1e-12)
    for row in rows:
        embedding = embeddings.get(str(row["job_id"]))
        if embedding is None:
            continue
        margin = float(np.dot(embedding, positive_centroid) - np.dot(embedding, negative_centroid))
        row["proxy_scores"]["vjepa_centroid_margin"] = margin


def compute_composite_and_roles(rows: list[dict[str, Any]]) -> None:
    for metric, out_key in [
        ("tribe_bmd_projection", "tribe_bmd_z_family"),
        ("vjepa_centroid_margin", "vjepa_margin_z_family"),
        ("clip_seed_video_preservation", "clip_preservation_z_family"),
        ("visual_quality_proxy", "visual_quality_z_family"),
    ]:
        for row in rows:
            value = row["proxy_scores"].get(metric) if metric != "visual_quality_proxy" else row.get(metric)
            row[out_key] = value
        zscore_by_family(rows, out_key, f"{out_key}_z")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scores = row["proxy_scores"]
        required = [
            scores.get("tribe_bmd_projection"),
            scores.get("vjepa_centroid_margin"),
            scores.get("clip_seed_video_preservation"),
        ]
        if all(value is not None and math.isfinite(float(value)) for value in required):
            composite = (
                float(row.get("tribe_bmd_z_family_z") or 0.0)
                + 0.5 * float(row.get("vjepa_margin_z_family_z") or 0.0)
                + 0.25 * float(row.get("clip_preservation_z_family_z") or 0.0)
                + 0.10 * float(row.get("visual_quality_z_family_z") or 0.0)
            )
            scores["composite_proxy_score"] = composite
            by_family[str(row["family_id"])].append(row)
        else:
            scores["composite_proxy_score"] = None

    for family_rows in by_family.values():
        ranked = sorted(
            family_rows,
            key=lambda row: (
                float(row["proxy_scores"]["composite_proxy_score"]),
                float(row.get("visual_quality_proxy") or -999),
            ),
            reverse=True,
        )
        if len(ranked) < 2:
            continue
        ranked[0]["proxy_scores"]["selection_role"] = "selector_top_proxy"
        # Prefer a lower composite clip with no hard flags and close visual quality.
        top_quality = float(ranked[0].get("visual_quality_proxy") or 0.0)
        controls = sorted(
            ranked[1:],
            key=lambda row: (
                float(row["proxy_scores"]["composite_proxy_score"]),
                abs(float(row.get("visual_quality_proxy") or 0.0) - top_quality),
            ),
        )
        controls[0]["proxy_scores"]["selection_role"] = "quality_matched_control_proxy"
        for row in ranked[1:]:
            if row["proxy_scores"].get("selection_role") is None:
                row["proxy_scores"]["selection_role"] = "withheld_candidate"


def summarize(rows: list[dict[str, Any]], model_status: dict[str, dict[str, int]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    family_summary = []
    for family_id, family_rows in sorted(by_family.items()):
        scored = [
            row
            for row in family_rows
            if row["proxy_scores"].get("composite_proxy_score") is not None
        ]
        roles = Counter(str(row["proxy_scores"].get("selection_role")) for row in scored)
        best = max(
            scored,
            key=lambda row: float(row["proxy_scores"]["composite_proxy_score"]),
            default=None,
        )
        control = next(
            (
                row
                for row in scored
                if row["proxy_scores"].get("selection_role")
                == "quality_matched_control_proxy"
            ),
            None,
        )
        family_summary.append(
            {
                "family_id": family_id,
                "n_candidates": len(family_rows),
                "n_complete_proxy_scores": len(scored),
                "selection_status": (
                    "roles_frozen_proxy_only"
                    if roles.get("selector_top_proxy") == 1
                    and roles.get("quality_matched_control_proxy") == 1
                    else "blocked_incomplete_proxy_scores"
                ),
                "selector_top_proxy": best["job_id"] if best else None,
                "quality_matched_control_proxy": control["job_id"] if control else None,
                "roles": dict(sorted(roles.items())),
            }
        )
    return {
        "n_candidates": len(rows),
        "n_complete_proxy_scores": sum(
            1 for row in rows if row["proxy_scores"].get("composite_proxy_score") is not None
        ),
        "model_status": model_status,
        "family_summary": family_summary,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Improved-v1 Seedance Proxy Scores",
        "",
        f"Date: `{report['created_at_utc']}`",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
        f"- Candidates: `{summary['n_candidates']}`",
        f"- Complete proxy scores: `{summary['n_complete_proxy_scores']}`",
        f"- Feature cache dir: `{report['feature_dir']}`",
        "",
        "## Model Status",
        "",
        "| model | status counts |",
        "|---|---|",
    ]
    for model, counts in summary["model_status"].items():
        lines.append(f"| `{model}` | `{json.dumps(counts, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## Family Selection Audit",
            "",
            "| family | complete | status | selector_top_proxy | quality_matched_control_proxy |",
            "|---|---:|---|---|---|",
        ]
    )
    for family in summary["family_summary"]:
        lines.append(
            "| `{family_id}` | {complete} | `{status}` | `{top}` | `{control}` |".format(
                family_id=family["family_id"],
                complete=family["n_complete_proxy_scores"],
                status=family["selection_status"],
                top=family["selector_top_proxy"],
                control=family["quality_matched_control_proxy"],
            )
        )
    lines.extend(
        [
            "",
            "## Composite Rule",
            "",
            "`z_family(TRIBE/BMD) + 0.5*z_family(V-JEPA centroid margin) + 0.25*z_family(CLIP prompt-video alignment) + 0.10*z_family(visual quality)`",
            "",
            "## Claim Boundary",
            "",
            "These are compute-proxy selection scores for exact generated MP4 bytes. They are not human memorability evidence and do not replace the preregistered delayed-recognition study.",
            "",
        ]
    )
    return "\n".join(lines)


async def main_async() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    rows = [dict(row) for row in manifest["candidates"]]
    if args.limit is not None:
        rows = rows[: args.limit]
    for row in rows:
        row["proxy_scores"] = dict(row.get("proxy_scores") or {})

    tribe_direction = load_direction(args.tribe_vmem)
    vjepa_direction = load_direction(args.vjepa_direction)

    print(
        json.dumps(
            {
                "n_candidates": len(rows),
                "tribe_concurrency": args.tribe_concurrency,
                "vjepa_concurrency": args.vjepa_concurrency,
                "clip_frames": args.clip_frames,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    tribe_task = None
    vjepa_task = None
    clip_task = None
    if not args.skip_tribe:
        tribe_task = asyncio.create_task(
            score_tribe(
                rows,
                app_name=args.tribe_app_name,
                feature_dir=args.feature_dir,
                direction=tribe_direction,
                concurrency=args.tribe_concurrency,
                timeout=args.tribe_timeout,
                force=args.force,
            )
        )
    if not args.skip_vjepa:
        vjepa_task = asyncio.create_task(
            score_vjepa(
                rows,
                app_name=args.vjepa_app_name,
                feature_dir=args.feature_dir,
                direction=vjepa_direction,
                concurrency=args.vjepa_concurrency,
                timeout=args.vjepa_timeout,
                force=args.force,
            )
        )
    if not args.skip_clip:
        clip_task = asyncio.create_task(
            asyncio.to_thread(
                score_clip,
                rows,
                feature_dir=args.feature_dir,
                model_id=args.clip_model_id,
                n_frames=args.clip_frames,
                force=args.force,
            )
        )

    tribe_results = await tribe_task if tribe_task is not None else None
    vjepa_results = await vjepa_task if vjepa_task is not None else None
    clip_results = await clip_task if clip_task is not None else None

    model_status = attach_model_results(
        rows,
        tribe_results=tribe_results,
        vjepa_results=vjepa_results,
        clip_results=clip_results,
    )
    compute_vjepa_centroid_margins(rows, args.feature_dir)
    compute_composite_and_roles(rows)
    summary = summarize(rows, model_status)
    status = (
        "proxy_scores_complete_roles_frozen_proxy_only"
        if summary["n_complete_proxy_scores"] == len(rows)
        and all(
            family["selection_status"] == "roles_frozen_proxy_only"
            for family in summary["family_summary"]
        )
        else "proxy_scores_incomplete"
    )
    report = {
        "schema_version": "confirmatory_seedance_proxy_scores.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "source_manifest": str(args.manifest),
        "feature_dir": str(args.feature_dir),
        "tribe_vmem": str(args.tribe_vmem),
        "vjepa_direction": str(args.vjepa_direction),
        "clip_model_id": None if args.skip_clip else args.clip_model_id,
        "summary": summary,
        "rows": sorted(rows, key=lambda row: (str(row["family_id"]), int(row["candidate_index"]))),
    }
    write_json(args.out_json, report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "n_complete_proxy_scores": summary["n_complete_proxy_scores"],
                "output_json": str(args.out_json),
                "output_md": str(args.out_md),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "proxy_scores_complete_roles_frozen_proxy_only" else 1


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
