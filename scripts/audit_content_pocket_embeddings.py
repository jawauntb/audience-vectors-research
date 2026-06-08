"""Audit CLIP/V-JEPA-style embedding structure for SVD content pockets.

The lightweight visual descriptor audit did not explain the stable positive
content pockets strongly enough to become a verifier. This script performs the
next queued audit: encode the exact pocket-regime generated videos with CLIP,
then test whether embedding-space centroid margins or leakage-aware classifiers
separate positive pockets from hard negative controls without using TRIBE score
as an input feature.

V-JEPA is treated as an optional feature family. If V-JEPA embeddings are not
provided, the report records that absence instead of pretending the verifier ran.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from audience_vectors.bo_prompt_manifests import build_replay_seed_pool

POSITIVE_POCKETS = {
    "fresh24_orange_flowers",
    "fresh24_hanging_clothes",
    "fresh24_blue_jellyfish",
    "fresh24_old_car",
}
NEGATIVE_CONTROLS = {
    "fresh24_aerial_beach",
    "fresh24_city_street",
    "fresh24_storm_beach",
}

DEFAULT_REPLAY_REPORT = Path(
    "data/reports/"
    "bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_"
    "20260608.json"
)
DEFAULT_OUT_JSON = (
    Path("research_program")
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
    / "content_pocket_embedding_audit_summary_20260608.json"
)
DEFAULT_OUT_MD = (
    Path("research_program")
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
    / "content_pocket_embedding_audit_result_20260608.md"
)
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class Candidate:
    """One task-level candidate with mean embeddings across replicates."""

    task_id: str
    pocket: str
    label: str
    seed_idx: int
    prompt: str
    mean_score: float
    min_score: float
    max_score: float
    n_replicates: int
    seed_embedding: np.ndarray
    video_embedding: np.ndarray
    vjepa_embedding: np.ndarray | None
    text_embedding: np.ndarray | None


def logical_path(path: Path) -> str:
    """Return a stable repo/data-lake-looking path for reports."""
    parts = path.resolve().parts
    for anchor in ("data", "research_program", "scripts", "src", "tests"):
        if anchor in parts:
            return str(Path(*parts[parts.index(anchor) :]))
    return str(path)


def repo_root_for_report(report_path: Path) -> Path:
    """Infer the worktree root from a data/reports report path."""
    if report_path.parent.name == "reports" and report_path.parent.parent.name == "data":
        return report_path.parent.parent.parent
    return Path.cwd()


def normalize_vector(values: np.ndarray) -> np.ndarray:
    """L2-normalize one vector."""
    arr = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def normalize_matrix(values: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a matrix."""
    arr = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for normalized or unnormalized vectors."""
    aa = normalize_vector(a)
    bb = normalize_vector(b)
    return float(np.dot(aa, bb))


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    """Mean-pool and normalize embeddings."""
    if not vectors:
        raise ValueError("cannot average empty embedding list")
    return normalize_vector(np.mean(np.stack(vectors), axis=0))


def load_npz_embedding(path: Path) -> np.ndarray:
    """Load a normalized feature vector from a feature `.npz` file."""
    data = np.load(path, allow_pickle=False)
    if "embedding" in data:
        values = data["embedding"]
    elif "features" in data:
        values = data["features"]
    else:
        raise KeyError(f"{path} has no `embedding` or `features` array")
    return normalize_vector(np.asarray(values, dtype=np.float32).reshape(-1))


def choose_device() -> Any:
    """Choose a torch device without importing torch at module import time."""
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def frames_from_mp4(path: Path, *, n_frames: int) -> list[Image.Image]:
    """Sample deterministic frames from an MP4."""
    frames = np.asarray(iio.imread(path))
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"expected nonempty video frame stack, got {frames.shape}")
    indices = np.linspace(0, frames.shape[0] - 1, min(n_frames, frames.shape[0]))
    return [
        Image.fromarray(frames[int(round(idx))]).convert("RGB")
        for idx in indices
    ]


class ClipEmbedder:
    """Small CLIP embedding wrapper for seed images, video frames, and prompts."""

    def __init__(self, model_id: str) -> None:
        import torch  # noqa: PLC0415
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        self.torch = torch
        self.device = choose_device()
        self.model_id = model_id
        print(f"[embedding-audit] loading {model_id} on {self.device}", flush=True)
        self.processor: Any = CLIPProcessor.from_pretrained(model_id)
        model: Any = CLIPModel.from_pretrained(model_id)
        self.model: Any = model.to(self.device).eval()
        self.image_cache: dict[Path, np.ndarray] = {}
        self.video_cache: dict[Path, np.ndarray] = {}
        self.text_cache: dict[str, np.ndarray] = {}

    def image_embedding(self, path: Path) -> np.ndarray:
        if path in self.image_cache:
            return self.image_cache[path]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        out = normalize_matrix(features.detach().float().cpu().numpy())[0]
        self.image_cache[path] = out
        return out

    def video_embedding(self, path: Path, *, n_frames: int) -> np.ndarray:
        if path in self.video_cache:
            return self.video_cache[path]
        frames = frames_from_mp4(path, n_frames=n_frames)
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        frame_embeddings = normalize_matrix(features.detach().float().cpu().numpy())
        out = mean_embedding([frame for frame in frame_embeddings])
        self.video_cache[path] = out
        return out

    def text_embedding(self, text: str) -> np.ndarray:
        if text in self.text_cache:
            return self.text_cache[text]
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        out = normalize_matrix(features.detach().float().cpu().numpy())[0]
        self.text_cache[text] = out
        return out


def resolve_report_path(path: Path) -> Path:
    """Resolve a report path with a clear local-data-lake error."""
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{path} does not exist. Pass --replay-report pointing at the local "
        "data-lake copy of the pocket-regime audit report."
    )


def load_report(path: Path) -> dict[str, Any]:
    """Load a replay report JSON."""
    return json.loads(path.read_text())


def rows_by_task(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group scored replay rows by task id."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report["rows"]:
        if row.get("replay_tribe_score") is None:
            continue
        trial = row["trial"]
        grouped[str(trial["task_id"])].append(row)
    return dict(grouped)


def build_candidates(
    *,
    report: dict[str, Any],
    report_path: Path,
    seed_root: Path,
    embedder: ClipEmbedder,
    max_video_frames: int,
    include_text: bool,
    vjepa_features_dir: Path | None,
) -> list[Candidate]:
    """Build task-level candidates and embeddings from a pocket replay report."""
    seed_pool = build_replay_seed_pool(
        seed_root,
        pool_size=int(report["replay_seed_pool_size"]),
    )
    seed_by_slot = {slot.slot: slot for slot in seed_pool}
    video_root = repo_root_for_report(report_path)
    candidates: list[Candidate] = []

    for task_id, group in sorted(rows_by_task(report).items()):
        first = group[0]
        trial = first["trial"]
        seed_idx = int(trial["seed_idx"])
        seed_slot = seed_by_slot[seed_idx]
        pocket = seed_slot.bmd_name
        if pocket not in POSITIVE_POCKETS and pocket not in NEGATIVE_CONTROLS:
            continue

        video_embeddings: list[np.ndarray] = []
        vjepa_embeddings: list[np.ndarray] = []
        scores: list[float] = []
        for row in group:
            local_video_path = row.get("local_video_path")
            if not local_video_path:
                continue
            video_path = video_root / str(local_video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"missing generated video: {video_path}")
            video_embeddings.append(
                embedder.video_embedding(video_path, n_frames=max_video_frames)
            )
            if vjepa_features_dir is not None:
                vjepa_path = vjepa_features_dir / f"{video_path.stem}.npz"
                if vjepa_path.exists():
                    vjepa_embeddings.append(load_npz_embedding(vjepa_path))
            scores.append(float(row["replay_tribe_score"]))

        if not video_embeddings:
            continue
        prompt = str(trial.get("prompt") or seed_slot.prompt)
        candidates.append(
            Candidate(
                task_id=task_id,
                pocket=pocket,
                label="positive" if pocket in POSITIVE_POCKETS else "negative_control",
                seed_idx=seed_idx,
                prompt=prompt,
                mean_score=float(np.mean(scores)),
                min_score=float(np.min(scores)),
                max_score=float(np.max(scores)),
                n_replicates=len(scores),
                seed_embedding=embedder.image_embedding(seed_slot.image_path),
                video_embedding=mean_embedding(video_embeddings),
                vjepa_embedding=(
                    mean_embedding(vjepa_embeddings)
                    if len(vjepa_embeddings) == len(video_embeddings)
                    else None
                ),
                text_embedding=embedder.text_embedding(prompt) if include_text else None,
            )
        )
    return candidates


def candidate_records(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Serialize candidate-level scalar summaries."""
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        record: dict[str, Any] = {
            "task_id": candidate.task_id,
            "pocket": candidate.pocket,
            "label": candidate.label,
            "seed_idx": candidate.seed_idx,
            "mean_replay_tribe_score": candidate.mean_score,
            "min_replay_tribe_score": candidate.min_score,
            "max_replay_tribe_score": candidate.max_score,
            "n_replicates": candidate.n_replicates,
            "seed_video_clip_cosine": cosine(
                candidate.seed_embedding,
                candidate.video_embedding,
            ),
            "vjepa_embedding_available": candidate.vjepa_embedding is not None,
        }
        if candidate.text_embedding is not None:
            record["prompt_video_clip_cosine"] = cosine(
                candidate.text_embedding,
                candidate.video_embedding,
            )
            record["prompt_seed_clip_cosine"] = cosine(
                candidate.text_embedding,
                candidate.seed_embedding,
            )
        out.append(record)
    return out


def pocket_summaries(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Aggregate candidates by pocket."""
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.pocket].append(candidate)
    summaries: list[dict[str, Any]] = []
    for pocket, group in sorted(grouped.items()):
        scores = [candidate.mean_score for candidate in group]
        summaries.append(
            {
                "pocket": pocket,
                "label": group[0].label,
                "n_candidates": len(group),
                "mean_score": float(np.mean(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "positive_candidates": int(sum(score > 0 for score in scores)),
            }
        )
    return summaries


def finite_nonconstant(values: list[float]) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)) and np.std(arr) > 1e-12)


def pearson(x: list[float], y: list[float]) -> float | None:
    if not finite_nonconstant(x) or not finite_nonconstant(y):
        return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def cohen_d(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if len(pos) < 2 or len(neg) < 2:
        return None
    pos_var = float(np.var(pos, ddof=1))
    neg_var = float(np.var(neg, ddof=1))
    pooled = math.sqrt(
        ((len(pos) - 1) * pos_var + (len(neg) - 1) * neg_var)
        / (len(pos) + len(neg) - 2)
    )
    if pooled <= 1e-12:
        return None
    return float((np.mean(pos) - np.mean(neg)) / pooled)


def descriptor_metric(
    *,
    name: str,
    family: str,
    values: list[float],
    labels: list[int],
    scores: list[float],
) -> dict[str, Any] | None:
    """Build one scalar descriptor separation metric."""
    if not finite_nonconstant(values):
        return None
    pos = np.asarray([value for value, label in zip(values, labels, strict=True) if label == 1])
    neg = np.asarray([value for value, label in zip(values, labels, strict=True) if label == 0])
    auc = float(roc_auc_score(labels, values))
    signed_d = cohen_d(pos, neg)
    return {
        "feature": name,
        "family": family,
        "positive_mean": float(np.mean(pos)),
        "negative_mean": float(np.mean(neg)),
        "direction": "higher_for_positive" if auc >= 0.5 else "lower_for_positive",
        "roc_auc_raw": auc,
        "separation_auc": max(auc, 1.0 - auc),
        "cohen_d": signed_d,
        "abs_cohen_d": abs(signed_d) if signed_d is not None else None,
        "pearson_with_mean_score": pearson(values, scores),
    }


def pocket_heldout_centroid_margins(
    candidates: list[Candidate],
    *,
    embedding_name: str,
) -> list[float]:
    """Compute positive-minus-negative centroid margins excluding same pocket."""
    values: list[float] = []
    for candidate in candidates:
        pos_vectors: list[np.ndarray] = []
        neg_vectors: list[np.ndarray] = []
        for other in candidates:
            if other.pocket == candidate.pocket:
                continue
            vector = embedding_value(other, embedding_name)
            if vector is None:
                continue
            if other.label == "positive":
                pos_vectors.append(vector)
            else:
                neg_vectors.append(vector)
        if not pos_vectors or not neg_vectors:
            values.append(float("nan"))
            continue
        vector = embedding_value(candidate, embedding_name)
        if vector is None:
            values.append(float("nan"))
            continue
        values.append(
            cosine(vector, mean_embedding(pos_vectors))
            - cosine(vector, mean_embedding(neg_vectors))
        )
    return values


def embedding_value(candidate: Candidate, embedding_name: str) -> np.ndarray | None:
    """Return one named embedding from a candidate."""
    if embedding_name == "seed_embedding":
        return candidate.seed_embedding
    if embedding_name == "video_embedding":
        return candidate.video_embedding
    if embedding_name == "vjepa_embedding":
        return candidate.vjepa_embedding
    raise ValueError(f"unknown embedding: {embedding_name}")


def embedding_candidates(
    candidates: list[Candidate],
    *,
    embedding_name: str,
) -> list[Candidate]:
    """Keep candidates with a concrete embedding for this family."""
    return [
        candidate
        for candidate in candidates
        if embedding_value(candidate, embedding_name) is not None
    ]


def descriptor_metrics(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Compute embedding descriptor separation metrics."""
    labels = [1 if candidate.label == "positive" else 0 for candidate in candidates]
    scores = [candidate.mean_score for candidate in candidates]
    records = candidate_records(candidates)
    metrics: list[dict[str, Any]] = []

    raw_features: dict[tuple[str, str], list[float]] = {
        ("clip_seed_image", "pocket_heldout_centroid_margin"): pocket_heldout_centroid_margins(
            candidates,
            embedding_name="seed_embedding",
        ),
        ("clip_video", "pocket_heldout_centroid_margin"): pocket_heldout_centroid_margins(
            candidates,
            embedding_name="video_embedding",
        ),
        ("clip_seed_video", "seed_video_clip_cosine"): [
            float(row["seed_video_clip_cosine"]) for row in records
        ],
    }
    if all(candidate.vjepa_embedding is not None for candidate in candidates):
        raw_features[("vjepa_video", "pocket_heldout_centroid_margin")] = (
            pocket_heldout_centroid_margins(
                candidates,
                embedding_name="vjepa_embedding",
            )
        )
    if all(candidate.text_embedding is not None for candidate in candidates):
        raw_features[("clip_prompt_video", "prompt_video_clip_cosine")] = [
            float(row["prompt_video_clip_cosine"]) for row in records
        ]
        raw_features[("clip_prompt_seed", "prompt_seed_clip_cosine")] = [
            float(row["prompt_seed_clip_cosine"]) for row in records
        ]

    for (family, name), values in raw_features.items():
        metric = descriptor_metric(
            name=name,
            family=family,
            values=values,
            labels=labels,
            scores=scores,
        )
        if metric is not None:
            metrics.append(metric)

    return sorted(
        metrics,
        key=lambda row: (
            float(row["separation_auc"]),
            float(row["abs_cohen_d"] or 0.0),
        ),
        reverse=True,
    )


def leave_pocket_out_classifier(
    candidates: list[Candidate],
    *,
    embedding_name: str,
    family: str,
) -> dict[str, Any]:
    """Train a leakage-aware classifier by holding out one pocket at a time."""
    pockets = sorted({candidate.pocket for candidate in candidates})
    true_labels: list[int] = []
    probabilities: list[float] = []
    predictions: list[int] = []

    for pocket in pockets:
        train = [candidate for candidate in candidates if candidate.pocket != pocket]
        test = [candidate for candidate in candidates if candidate.pocket == pocket]
        y_train = np.asarray(
            [1 if candidate.label == "positive" else 0 for candidate in train],
            dtype=np.int64,
        )
        if len(set(y_train.tolist())) < 2:
            continue
        train_vectors = [embedding_value(candidate, embedding_name) for candidate in train]
        test_vectors = [embedding_value(candidate, embedding_name) for candidate in test]
        if any(vector is None for vector in train_vectors + test_vectors):
            continue
        concrete_train_vectors = [
            vector for vector in train_vectors if vector is not None
        ]
        concrete_test_vectors = [vector for vector in test_vectors if vector is not None]
        x_train = normalize_matrix(np.stack(concrete_train_vectors))
        x_test = normalize_matrix(np.stack(concrete_test_vectors))
        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=0,
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        probabilities.extend(float(prob) for prob in probs)
        predictions.extend(int(prob >= 0.5) for prob in probs)
        true_labels.extend(1 if candidate.label == "positive" else 0 for candidate in test)

    if len(set(true_labels)) < 2 or not finite_nonconstant(probabilities):
        auc: float | None = None
    else:
        auc = float(roc_auc_score(true_labels, probabilities))
    balanced = (
        float(balanced_accuracy_score(true_labels, predictions))
        if len(set(true_labels)) >= 2
        else None
    )
    return {
        "family": family,
        "embedding": embedding_name,
        "validation": "leave_one_pocket_out",
        "n_predictions": len(probabilities),
        "roc_auc": auc,
        "balanced_accuracy": balanced,
        "positive_probability_mean": float(
            np.mean(
                [
                    probability
                    for probability, label in zip(probabilities, true_labels, strict=True)
                    if label == 1
                ]
            )
        ),
        "negative_probability_mean": float(
            np.mean(
                [
                    probability
                    for probability, label in zip(probabilities, true_labels, strict=True)
                    if label == 0
                ]
            )
        ),
    }


def classifier_results(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Compute leakage-aware classifier summaries."""
    results = [
        leave_pocket_out_classifier(
            candidates,
            embedding_name="seed_embedding",
            family="clip_seed_image",
        ),
        leave_pocket_out_classifier(
            candidates,
            embedding_name="video_embedding",
            family="clip_video",
        ),
    ]
    vjepa_candidates = embedding_candidates(candidates, embedding_name="vjepa_embedding")
    if len(vjepa_candidates) == len(candidates):
        results.append(
            leave_pocket_out_classifier(
                vjepa_candidates,
                embedding_name="vjepa_embedding",
                family="vjepa_video",
            )
        )
    return results


def gate_summary(
    descriptors: list[dict[str, Any]],
    classifiers: list[dict[str, Any]],
    *,
    min_auc: float,
    min_abs_d: float,
    min_classifier_auc: float,
    min_balanced_accuracy: float,
) -> dict[str, Any]:
    """Decide whether embedding-level explanation clears the audit gate."""
    passing_descriptors = [
        row
        for row in descriptors
        if float(row["separation_auc"]) >= min_auc
        and float(row["abs_cohen_d"] or 0.0) >= min_abs_d
    ]
    passing_classifiers = [
        row
        for row in classifiers
        if row["roc_auc"] is not None
        and row["balanced_accuracy"] is not None
        and float(row["roc_auc"]) >= min_classifier_auc
        and float(row["balanced_accuracy"]) >= min_balanced_accuracy
    ]
    best_descriptor = passing_descriptors[0] if passing_descriptors else (
        descriptors[0] if descriptors else None
    )
    classifier_pool = passing_classifiers if passing_classifiers else classifiers
    best_classifier = sorted(
        classifier_pool,
        key=lambda row: (
            float(row["roc_auc"] or -1.0),
            float(row["balanced_accuracy"] or -1.0),
        ),
        reverse=True,
    )[0] if classifier_pool else None
    return {
        "accepted": bool(passing_descriptors or passing_classifiers),
        "descriptor_rule": f"separation_auc >= {min_auc:.2f} and abs_cohen_d >= {min_abs_d:.2f}",
        "classifier_rule": (
            f"leave-one-pocket-out roc_auc >= {min_classifier_auc:.2f} "
            f"and balanced_accuracy >= {min_balanced_accuracy:.2f}"
        ),
        "n_passing_descriptors": len(passing_descriptors),
        "n_passing_classifiers": len(passing_classifiers),
        "best_descriptor": best_descriptor,
        "best_classifier": best_classifier,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_descriptor_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {feature} | {direction} | {pos} | {neg} | {auc} | {d} | {corr} |".format(
                family=row["family"],
                feature=row["feature"],
                direction=row["direction"],
                pos=fmt(row["positive_mean"]),
                neg=fmt(row["negative_mean"]),
                auc=fmt(row["separation_auc"]),
                d=fmt(row["abs_cohen_d"]),
                corr=fmt(row["pearson_with_mean_score"]),
            )
        )
    return "\n".join(lines)


def render_classifier_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| family | embedding | validation | predictions | AUC | balanced accuracy | pos prob mean | neg prob mean |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {embedding} | {validation} | {n} | {auc} | {bal} | {pos} | {neg} |".format(
                family=row["family"],
                embedding=row["embedding"],
                validation=row["validation"],
                n=row["n_predictions"],
                auc=fmt(row["roc_auc"]),
                bal=fmt(row["balanced_accuracy"]),
                pos=fmt(row["positive_probability_mean"]),
                neg=fmt(row["negative_probability_mean"]),
            )
        )
    return "\n".join(lines)


def render_pocket_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| pocket | label | candidates | mean | min | max | positive candidates |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: float(item["mean_score"]), reverse=True):
        lines.append(
            "| {pocket} | {label} | {n} | {mean} | {min_score} | {max_score} | {pos} |".format(
                pocket=row["pocket"],
                label=row["label"],
                n=row["n_candidates"],
                mean=fmt(row["mean_score"]),
                min_score=fmt(row["min_score"]),
                max_score=fmt(row["max_score"]),
                pos=row["positive_candidates"],
            )
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the human-facing result note."""
    gate = summary["gate"]
    lines = [
        "# Content-Pocket Embedding Audit Result - 2026-06-08",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: do CLIP/V-JEPA-style embeddings explain the stable positive "
        "content pockets after lightweight visual descriptors failed?",
        "",
        "Current regime:",
        "",
        "- Artifact types: restored seed images, generated SVD replay videos, "
        "TRIBE replay scores, CLIP seed/video/text embeddings, optional V-JEPA "
        "embeddings, pocket labels, and leakage-aware verifier outputs.",
        "- Operations: encode seed images and generated-video frame samples with "
        "CLIP, aggregate stochastic replicates to task-level embeddings, compute "
        "pocket-held-out centroid margins, and train leave-one-pocket-out "
        "classifiers.",
        "- Gates/verifiers: embedding descriptors cannot use TRIBE score as an "
        "input. Acceptance requires either the descriptor or classifier rule below.",
        "",
        "Action class: search inside the current compute-proxy regime. It becomes "
        "discovery-relevant only if an embedding descriptor becomes an accepted "
        "verifier for content-pocket consolidation.",
        "",
        "## Inputs",
        "",
        f"- Replay report: `{summary['source_replay_report']}`",
        f"- Seed root: `{summary['seed_root']}`",
        f"- CLIP model: `{summary['clip_model_id']}`",
        f"- V-JEPA status: {summary['vjepa_status']}",
        f"- Candidates: {summary['n_candidates']} task-level candidates from "
        f"{summary['n_rows']} scored replicate rows.",
        "",
        "## Score By Pocket",
        "",
        render_pocket_table(summary["score_by_pocket"]),
        "",
        "## Embedding Descriptor Separators",
        "",
        render_descriptor_table(summary["descriptor_metrics"]),
        "",
        "## Leakage-Aware Classifiers",
        "",
        render_classifier_table(summary["classifier_results"]),
        "",
        "## Gate",
        "",
        f"Descriptor rule: {gate['descriptor_rule']}.",
        "",
        f"Classifier rule: {gate['classifier_rule']}.",
        "",
        f"Gate result: **{'accepted' if gate['accepted'] else 'not accepted'}**.",
        "",
    ]
    if gate["best_descriptor"]:
        best = gate["best_descriptor"]
        lines.extend(
            [
                "Best descriptor:",
                "",
                f"- family: `{best['family']}`",
                f"- feature: `{best['feature']}`",
                f"- separation AUC: {fmt(best['separation_auc'])}",
                f"- absolute Cohen d: {fmt(best['abs_cohen_d'])}",
                f"- correlation with mean TRIBE score: {fmt(best['pearson_with_mean_score'])}",
                "",
            ]
        )
    if gate["best_classifier"]:
        best = gate["best_classifier"]
        lines.extend(
            [
                "Best classifier:",
                "",
                f"- family: `{best['family']}`",
                f"- validation: `{best['validation']}`",
                f"- ROC AUC: {fmt(best['roc_auc'])}",
                f"- balanced accuracy: {fmt(best['balanced_accuracy'])}",
                "",
            ]
        )

    if gate["accepted"]:
        best_family = (
            str(gate["best_descriptor"]["family"])
            if gate["best_descriptor"]
            else "embedding"
        )
        interpretation = (
            "The embedding audit clears the verifier gate. The best accepted "
            f"family is {best_family}, and the stable positive pockets are not "
            "merely opaque score islands: their seed/video embeddings contain "
            "enough structure to distinguish them from hard negative controls "
            "under leakage-aware evaluation. This is still compute-proxy "
            "evidence, not human memorability, but it gives the next "
            "replication or validation packet a real descriptor to track."
        )
        next_move = (
            "Use the accepted embedding descriptor as a covariate and stopping "
            "rule in the next replication or human/BMD validation packet. If a "
            "specific family fails here, keep that caveat explicit rather than "
            "promoting it through the broader accepted-gate result."
        )
    else:
        interpretation = (
            "The CLIP embedding audit does not clear the verifier gate. The "
            "content pockets remain stable under TRIBE, but the tested "
            "embedding structure does not currently explain them strongly "
            "enough. C-017 should remain a black-box compute-proxy finding "
            "until V-JEPA, BMD, human evidence, or another accepted descriptor "
            "explains it."
        )
        next_move = (
            "Use V-JEPA video embeddings or a human/BMD-grounded gate before "
            "spending more budget on blind stochastic replication."
        )

    lines.extend(["## Interpretation", "", interpretation, "", "## Next Move", "", next_move, ""])
    return "\n".join(lines)


def build_summary(
    *,
    replay_report_path: Path,
    seed_root: Path | None,
    clip_model_id: str,
    max_video_frames: int,
    include_text: bool,
    vjepa_features_dir: Path | None,
    min_auc: float,
    min_abs_d: float,
    min_classifier_auc: float,
    min_balanced_accuracy: float,
) -> dict[str, Any]:
    report_path = resolve_report_path(replay_report_path)
    report = load_report(report_path)
    resolved_seed_root = seed_root or Path(str(report["seed_root"]))
    embedder = ClipEmbedder(clip_model_id)
    candidates = build_candidates(
        report=report,
        report_path=report_path,
        seed_root=resolved_seed_root,
        embedder=embedder,
        max_video_frames=max_video_frames,
        include_text=include_text,
        vjepa_features_dir=vjepa_features_dir,
    )
    descriptors = descriptor_metrics(candidates)
    classifiers = classifier_results(candidates)
    gate = gate_summary(
        descriptors,
        classifiers,
        min_auc=min_auc,
        min_abs_d=min_abs_d,
        min_classifier_auc=min_classifier_auc,
        min_balanced_accuracy=min_balanced_accuracy,
    )
    n_vjepa_candidates = sum(candidate.vjepa_embedding is not None for candidate in candidates)
    if vjepa_features_dir is None:
        vjepa_status = "not run; exact pocket-replay feature dir not provided"
    elif n_vjepa_candidates == len(candidates):
        vjepa_status = (
            f"integrated for all {n_vjepa_candidates}/{len(candidates)} candidates "
            f"from `{logical_path(vjepa_features_dir)}`"
        )
    else:
        vjepa_status = (
            f"withheld; exact feature coverage {n_vjepa_candidates}/{len(candidates)} "
            f"from `{logical_path(vjepa_features_dir)}`"
        )
    return {
        "schema_version": 1,
        "kind": "content_pocket_embedding_audit",
        "source_replay_report": logical_path(report_path),
        "seed_root": logical_path(resolved_seed_root),
        "clip_model_id": clip_model_id,
        "max_video_frames": max_video_frames,
        "include_text": include_text,
        "vjepa_status": vjepa_status,
        "n_vjepa_candidates": n_vjepa_candidates,
        "positive_pockets": sorted(POSITIVE_POCKETS),
        "negative_controls": sorted(NEGATIVE_CONTROLS),
        "n_rows": int(sum(candidate.n_replicates for candidate in candidates)),
        "n_candidates": len(candidates),
        "score_by_pocket": pocket_summaries(candidates),
        "candidate_records": candidate_records(candidates),
        "descriptor_metrics": descriptors,
        "classifier_results": classifiers,
        "gate": gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", type=Path, default=DEFAULT_REPLAY_REPORT)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--clip-model-id", default=DEFAULT_CLIP_MODEL_ID)
    parser.add_argument("--max-video-frames", type=int, default=4)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--vjepa-features-dir", type=Path)
    parser.add_argument("--min-auc", type=float, default=0.85)
    parser.add_argument("--min-abs-d", type=float, default=1.0)
    parser.add_argument("--min-classifier-auc", type=float, default=0.85)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.75)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        replay_report_path=args.replay_report,
        seed_root=args.seed_root,
        clip_model_id=args.clip_model_id,
        max_video_frames=args.max_video_frames,
        include_text=args.include_text,
        vjepa_features_dir=args.vjepa_features_dir,
        min_auc=args.min_auc,
        min_abs_d=args.min_abs_d,
        min_classifier_auc=args.min_classifier_auc,
        min_balanced_accuracy=args.min_balanced_accuracy,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    args.out_md.write_text(render_markdown(summary))
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "n_rows": summary["n_rows"],
                "n_candidates": summary["n_candidates"],
                "gate": summary["gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
