"""Build a proxy-only neural-response-guided generation dry run.

This script does not generate videos, mutate the frozen content-pocket task
set, or claim validation. It asks a narrower feasibility question: if existing
proxy signals were used only for candidate selection or reranking, which side of
the frozen pairwise tasks would each proxy choose?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/future_work"
)
INPUT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_TASKS = INPUT_DIR / "content_pocket_validation_pairwise_tasks_20260608.json"
DEFAULT_DESCRIPTOR_SUMMARY = (
    INPUT_DIR / "descriptor_conditioned_replication_embedding_summary_20260608.json"
)
DEFAULT_BOUNDARY_SUMMARY = (
    INPUT_DIR / "boundary_pocket_audit_embedding_summary_20260608.json"
)
DEFAULT_OUT_JSON = (
    ARTIFACT_DIR / "neural_response_guided_generation_dry_run_20260608.json"
)
DEFAULT_OUT_MD = (
    ARTIFACT_DIR / "neural_response_guided_generation_dry_run_20260608.md"
)

CONTENT_POLICY = "content_pocket_candidate"
CONTROL_POLICY = "hard_negative_control"
SOURCE_POOLS = (
    "descriptor_conditioned_replication",
    "boundary_pocket_audit",
)
VIDEO_TASK_RE = re.compile(
    r"^bo_replay_\d+_(?P<task_id>sobol_prompt_search_\d+_slot\d+)_rep\d+$"
)
COMPOSITE_WEIGHTS = {
    "tribe_bmd_projection": 1.0,
    "vjepa_centroid_margin": 0.5,
    "clip_seed_video_preservation": 0.25,
}


@dataclass
class SideRecord:
    """One side of one frozen pairwise task."""

    pairwise_task_id: str
    side: str
    analysis_tier: str
    target_side: str
    policy: str
    pocket: str
    video_label: str
    logical_video_path: str
    source_pool: str
    generation_task_id: str
    tribe_score: float | None
    clip_seed_video_cosine: float | None
    mp4_found: bool
    vjepa_feature_found: bool
    visual_first_status: str | None
    visual_gate_passed: bool | None
    vjepa_embedding: np.ndarray | None
    vjepa_centroid_margin: float | None = None

    @property
    def key(self) -> str:
        return f"{self.pairwise_task_id}:{self.side}"

    @property
    def selects_target(self) -> bool:
        return self.side == self.target_side


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write an indented JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def logical_path(path: Path) -> str:
    """Return a stable repo/data path without local worktree prefixes."""
    parts = path.resolve().parts
    for anchor in ("research_program", "data", "scripts", "tests"):
        if anchor in parts:
            return str(Path(*parts[parts.index(anchor) :]))
    return path.name


def infer_source_pool(path: str) -> str:
    """Map a generated-video path to the source candidate pool."""
    if "bo_descriptor_conditioned_replication" in path:
        return "descriptor_conditioned_replication"
    if "bo_boundary_pocket_audit" in path:
        return "boundary_pocket_audit"
    return "unknown"


def parse_generation_task_id(video_label: str) -> str:
    """Extract the Sobol task id from a replay video label."""
    match = VIDEO_TASK_RE.fullmatch(video_label)
    if match is None:
        raise ValueError(f"cannot parse task id from video_label={video_label!r}")
    return match.group("task_id")


def normalize_vector(values: np.ndarray) -> np.ndarray:
    """L2-normalize one vector."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for normalized or unnormalized vectors."""
    return float(np.dot(normalize_vector(a), normalize_vector(b)))


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    """Mean-pool and normalize embeddings."""
    if not vectors:
        raise ValueError("cannot average empty embedding list")
    return normalize_vector(np.mean(np.stack(vectors), axis=0))


def load_npz_embedding(path: Path) -> np.ndarray:
    """Load a normalized feature vector from an `.npz` feature file."""
    data = np.load(path, allow_pickle=False)
    if "embedding" in data:
        values = data["embedding"]
    elif "features" in data:
        values = data["features"]
    else:
        raise KeyError(f"{path} has no `embedding` or `features` array")
    return normalize_vector(np.asarray(values, dtype=np.float32))


def parse_mapping(items: list[str]) -> dict[str, Path]:
    """Parse repeated `source_pool=path` CLI options."""
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected source_pool=path, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = Path(value).expanduser()
    return out


def default_embedding_summaries() -> dict[str, Path]:
    """Committed embedding summaries used by the dry run."""
    return {
        "descriptor_conditioned_replication": DEFAULT_DESCRIPTOR_SUMMARY,
        "boundary_pocket_audit": DEFAULT_BOUNDARY_SUMMARY,
    }


def load_embedding_lookup(paths: dict[str, Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index committed embedding-summary candidate records by source and task."""
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for source_pool, path in paths.items():
        if not path.exists():
            continue
        for record in load_json(path).get("candidate_records", []):
            lookup[(source_pool, str(record["task_id"]))] = record
    return lookup


def load_replay_status_lookup(
    paths: dict[str, Path],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index visual gate statuses by source pool and video label."""
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for source_pool, path in paths.items():
        if not path.exists():
            continue
        for row in load_json(path).get("rows", []):
            label = str(row.get("label") or Path(str(row.get("local_video_path"))).stem)
            gate = row.get("visual_artifact_gate") or {}
            lookup[(source_pool, label)] = {
                "visual_first_status": row.get("visual_first_status"),
                "visual_gate_passed": gate.get("passes_visual_gate"),
            }
    return lookup


def resolve_logical_path(logical: str, roots: list[Path]) -> Path | None:
    """Resolve a repo-relative data path against candidate local roots."""
    for root in roots:
        candidate = root / logical
        if candidate.exists():
            return candidate
    return None


def side_tribe_score(task: dict[str, Any], side: dict[str, Any]) -> float | None:
    """Return the replay TRIBE/BMD projection score for a task side."""
    metadata = task["metadata"]
    if side["policy"] == CONTENT_POLICY:
        return float(metadata["positive_replay_tribe_score"])
    if side["policy"] == CONTROL_POLICY:
        return float(metadata["control_replay_tribe_score"])
    return None


def side_record(
    *,
    task: dict[str, Any],
    side_name: str,
    data_roots: list[Path],
    embedding_lookup: dict[tuple[str, str], dict[str, Any]],
    replay_status_lookup: dict[tuple[str, str], dict[str, Any]],
    vjepa_feature_dirs: dict[str, Path],
) -> SideRecord:
    """Build one side record with all locally available proxy inputs."""
    raw_side = task[side_name]
    logical_video = str(raw_side["path"])
    source_pool = infer_source_pool(logical_video)
    video_label = str(raw_side["video_label"])
    generation_task_id = parse_generation_task_id(video_label)
    embedding_record = embedding_lookup.get((source_pool, generation_task_id), {})
    vjepa_path = vjepa_feature_dirs.get(source_pool, Path()) / f"{video_label}.npz"
    vjepa_embedding = load_npz_embedding(vjepa_path) if vjepa_path.exists() else None
    status = replay_status_lookup.get((source_pool, video_label), {})
    return SideRecord(
        pairwise_task_id=str(task["task_id"]),
        side=side_name,
        analysis_tier=str(task["metadata"]["analysis_tier"]),
        target_side=str(task["metadata"]["target_side"]),
        policy=str(raw_side["policy"]),
        pocket=str(raw_side["label"]),
        video_label=video_label,
        logical_video_path=logical_video,
        source_pool=source_pool,
        generation_task_id=generation_task_id,
        tribe_score=side_tribe_score(task, raw_side),
        clip_seed_video_cosine=maybe_float(
            embedding_record.get("seed_video_clip_cosine")
        ),
        mp4_found=resolve_logical_path(logical_video, data_roots) is not None,
        vjepa_feature_found=vjepa_embedding is not None,
        visual_first_status=maybe_str(status.get("visual_first_status")),
        visual_gate_passed=maybe_bool(status.get("visual_gate_passed")),
        vjepa_embedding=vjepa_embedding,
    )


def maybe_float(value: Any) -> float | None:
    """Cast finite scalar values to float."""
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def maybe_str(value: Any) -> str | None:
    """Cast optional values to strings."""
    if value is None:
        return None
    return str(value)


def maybe_bool(value: Any) -> bool | None:
    """Cast optional values to bool."""
    if value is None:
        return None
    return bool(value)


def build_side_records(
    *,
    task_doc: dict[str, Any],
    data_roots: list[Path],
    embedding_lookup: dict[tuple[str, str], dict[str, Any]],
    replay_status_lookup: dict[tuple[str, str], dict[str, Any]],
    vjepa_feature_dirs: dict[str, Path],
) -> list[SideRecord]:
    """Build side records for every frozen pairwise task side."""
    records: list[SideRecord] = []
    for task in task_doc["tasks"]:
        for side_name in ("left", "right"):
            records.append(
                side_record(
                    task=task,
                    side_name=side_name,
                    data_roots=data_roots,
                    embedding_lookup=embedding_lookup,
                    replay_status_lookup=replay_status_lookup,
                    vjepa_feature_dirs=vjepa_feature_dirs,
                )
            )
    return records


def assign_vjepa_margins(records: list[SideRecord]) -> None:
    """Assign leave-pocket-out positive-minus-control V-JEPA centroid margins."""
    by_tier: dict[str, list[SideRecord]] = defaultdict(list)
    for record in records:
        by_tier[record.analysis_tier].append(record)
    for group in by_tier.values():
        for record in group:
            if record.vjepa_embedding is None:
                continue
            pos = centroid_pool(group, record=record, policy=CONTENT_POLICY)
            neg = centroid_pool(group, record=record, policy=CONTROL_POLICY)
            if not pos or not neg:
                continue
            record.vjepa_centroid_margin = cosine(
                record.vjepa_embedding,
                mean_embedding(pos),
            ) - cosine(record.vjepa_embedding, mean_embedding(neg))


def centroid_pool(
    group: list[SideRecord],
    *,
    record: SideRecord,
    policy: str,
) -> list[np.ndarray]:
    """Return vectors for centroid construction, excluding the same pocket."""
    vectors: list[np.ndarray] = []
    for other in group:
        if other.policy != policy or other.pocket == record.pocket:
            continue
        if other.vjepa_embedding is not None:
            vectors.append(other.vjepa_embedding)
    return vectors


def finite_values(
    records: list[SideRecord],
    getter: str,
) -> list[tuple[str, str, float]]:
    """Collect finite side values grouped by tier."""
    values: list[tuple[str, str, float]] = []
    for record in records:
        value = getattr(record, getter)
        if value is not None and math.isfinite(float(value)):
            values.append((record.analysis_tier, record.key, float(value)))
    return values


def zscores_by_tier(records: list[SideRecord], getter: str) -> dict[str, float]:
    """Compute within-tier z-scores for one side-record scalar."""
    by_tier: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for tier, key, value in finite_values(records, getter):
        by_tier[tier].append((key, value))
    out: dict[str, float] = {}
    for values in by_tier.values():
        arr = np.asarray([value for _, value in values], dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        for key, value in values:
            out[key] = 0.0 if std <= 1e-12 else float((value - mean) / std)
    return out


def proxy_zscores(records: list[SideRecord]) -> dict[str, dict[str, float]]:
    """Build z-score maps used by the composite proxy."""
    return {
        "tribe_bmd_projection": zscores_by_tier(records, "tribe_score"),
        "vjepa_centroid_margin": zscores_by_tier(records, "vjepa_centroid_margin"),
        "clip_seed_video_preservation": zscores_by_tier(
            records,
            "clip_seed_video_cosine",
        ),
    }


def side_score(
    record: SideRecord,
    metric: str,
    zscores: dict[str, dict[str, float]],
) -> float | None:
    """Return the raw or composite score for one side and proxy metric."""
    if metric == "tribe_bmd_projection":
        return record.tribe_score
    if metric == "vjepa_centroid_margin":
        return record.vjepa_centroid_margin
    if metric == "clip_seed_video_preservation":
        return record.clip_seed_video_cosine
    if metric != "composite_proxy_score":
        raise ValueError(f"unknown metric: {metric}")

    total = 0.0
    for component, weight in COMPOSITE_WEIGHTS.items():
        value = zscores[component].get(record.key)
        if value is None:
            return None
        total += weight * value
    return total


def winner_for_scores(left: float | None, right: float | None) -> str | None:
    """Choose the higher-scoring side, returning `tie` for exact ties."""
    if left is None or right is None:
        return None
    if abs(left - right) <= 1e-12:
        return "tie"
    return "left" if left > right else "right"


def pairwise_decisions(
    records: list[SideRecord],
) -> list[dict[str, Any]]:
    """Score each pairwise task under each available proxy."""
    by_task: dict[str, dict[str, SideRecord]] = defaultdict(dict)
    for record in records:
        by_task[record.pairwise_task_id][record.side] = record
    zscores = proxy_zscores(records)
    metrics = [
        "tribe_bmd_projection",
        "vjepa_centroid_margin",
        "clip_seed_video_preservation",
        "composite_proxy_score",
    ]
    decisions: list[dict[str, Any]] = []
    for task_id, sides in sorted(by_task.items()):
        left = sides["left"]
        right = sides["right"]
        metric_scores = {
            metric: metric_decision(metric, left=left, right=right, zscores=zscores)
            for metric in metrics
        }
        decisions.append(
            {
                "task_id": task_id,
                "analysis_tier": left.analysis_tier,
                "target_side": left.target_side,
                "target_pocket": target_record(left, right).pocket,
                "control_pocket": control_record(left, right).pocket,
                "left": side_public_record(left),
                "right": side_public_record(right),
                "proxy_decisions": metric_scores,
            }
        )
    return decisions


def metric_decision(
    metric: str,
    *,
    left: SideRecord,
    right: SideRecord,
    zscores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build one metric decision record."""
    left_score = side_score(left, metric, zscores)
    right_score = side_score(right, metric, zscores)
    winner = winner_for_scores(left_score, right_score)
    return {
        "left_score": round_or_none(left_score),
        "right_score": round_or_none(right_score),
        "winner_side": winner,
        "selects_target": winner == left.target_side if winner not in (None, "tie") else None,
    }


def target_record(left: SideRecord, right: SideRecord) -> SideRecord:
    """Return the target content-pocket side."""
    return left if left.selects_target else right


def control_record(left: SideRecord, right: SideRecord) -> SideRecord:
    """Return the hard-negative control side."""
    return right if left.selects_target else left


def side_public_record(record: SideRecord) -> dict[str, Any]:
    """Serialize one side record without raw embeddings."""
    return {
        "side": record.side,
        "policy": record.policy,
        "pocket": record.pocket,
        "video_label": record.video_label,
        "path": record.logical_video_path,
        "source_pool": record.source_pool,
        "generation_task_id": record.generation_task_id,
        "mp4_found": record.mp4_found,
        "vjepa_feature_found": record.vjepa_feature_found,
        "visual_first_status": record.visual_first_status,
        "visual_gate_passed": record.visual_gate_passed,
        "tribe_bmd_projection": round_or_none(record.tribe_score),
        "vjepa_centroid_margin": round_or_none(record.vjepa_centroid_margin),
        "clip_seed_video_preservation": round_or_none(
            record.clip_seed_video_cosine
        ),
    }


def round_or_none(value: float | None) -> float | None:
    """Round finite floats for stable artifacts."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 6)


def aggregate_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pairwise proxy decisions."""
    metrics = list(decisions[0]["proxy_decisions"]) if decisions else []
    by_metric = {
        metric: aggregate_metric(decisions, metric=metric)
        for metric in metrics
    }
    return {
        "by_metric": by_metric,
        "agreement_matrix": agreement_matrix(decisions, metrics=metrics),
        "disagreement_examples": disagreement_examples(decisions),
    }


def aggregate_metric(decisions: list[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    """Summarize one metric across all tasks and tiers."""
    usable = [
        decision
        for decision in decisions
        if decision["proxy_decisions"][metric]["selects_target"] is not None
    ]
    selected = [
        decision
        for decision in usable
        if decision["proxy_decisions"][metric]["selects_target"] is True
    ]
    return {
        "n_decisions": len(usable),
        "selects_content_pocket_target": len(selected),
        "target_selection_rate": rate(len(selected), len(usable)),
        "by_tier": {
            tier: aggregate_metric_for_tier(usable, metric=metric, tier=tier)
            for tier in sorted({decision["analysis_tier"] for decision in usable})
        },
    }


def aggregate_metric_for_tier(
    decisions: list[dict[str, Any]],
    *,
    metric: str,
    tier: str,
) -> dict[str, Any]:
    """Summarize one metric within one analysis tier."""
    tiered = [decision for decision in decisions if decision["analysis_tier"] == tier]
    selected = [
        decision
        for decision in tiered
        if decision["proxy_decisions"][metric]["selects_target"] is True
    ]
    return {
        "n_decisions": len(tiered),
        "selects_content_pocket_target": len(selected),
        "target_selection_rate": rate(len(selected), len(tiered)),
    }


def rate(numerator: int, denominator: int) -> float | None:
    """Compute a rounded rate."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def agreement_matrix(
    decisions: list[dict[str, Any]],
    *,
    metrics: list[str],
) -> dict[str, dict[str, int]]:
    """Count same-side proxy agreement for each metric pair."""
    matrix: dict[str, dict[str, int]] = {}
    for metric in metrics:
        matrix[metric] = {}
        for other in metrics:
            matrix[metric][other] = agreement_count(decisions, metric, other)
    return matrix


def agreement_count(decisions: list[dict[str, Any]], metric: str, other: str) -> int:
    """Count tasks where two metrics choose the same non-tie side."""
    count = 0
    for decision in decisions:
        first = decision["proxy_decisions"][metric]["winner_side"]
        second = decision["proxy_decisions"][other]["winner_side"]
        if first not in (None, "tie") and first == second:
            count += 1
    return count


def disagreement_examples(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact examples where proxies disagree."""
    examples: list[dict[str, Any]] = []
    for decision in decisions:
        winners = {
            metric: result["winner_side"]
            for metric, result in decision["proxy_decisions"].items()
            if result["winner_side"] not in (None, "tie")
        }
        if len(set(winners.values())) <= 1:
            continue
        examples.append(
            {
                "task_id": decision["task_id"],
                "analysis_tier": decision["analysis_tier"],
                "target_pocket": decision["target_pocket"],
                "control_pocket": decision["control_pocket"],
                "winner_sides": winners,
            }
        )
    return examples[:8]


def data_resolution(
    task_doc: dict[str, Any],
    records: list[SideRecord],
) -> dict[str, Any]:
    """Summarize available local inputs for the dry run."""
    unique_paths = {record.logical_video_path for record in records}
    found_unique_paths = {
        record.logical_video_path for record in records if record.mp4_found
    }
    statuses = Counter(
        record.visual_first_status or "unknown"
        for record in records
    )
    return {
        "n_tasks": int(task_doc["n_tasks"]),
        "n_side_observations": len(records),
        "n_unique_mp4_paths": len(unique_paths),
        "mp4_found_side_observations": sum(record.mp4_found for record in records),
        "mp4_found_unique_paths": len(found_unique_paths),
        "vjepa_feature_found_side_observations": sum(
            record.vjepa_feature_found for record in records
        ),
        "clip_seed_video_found_side_observations": sum(
            record.clip_seed_video_cosine is not None for record in records
        ),
        "visual_first_status_counts": dict(sorted(statuses.items())),
        "source_pool_counts": dict(
            sorted(Counter(record.source_pool for record in records).items())
        ),
    }


def reward_components() -> list[dict[str, Any]]:
    """Describe proxy reward components without overclaiming."""
    return [
        {
            "name": "tribe_bmd_projection",
            "locally_available_now": True,
            "differentiability": "black-box scalar unless the full TRIBE/BMD scorer is wired into a differentiable generator loop",
            "required_inputs": ["generated MP4", "TRIBE/BMD replay scoring artifact"],
            "role_in_dry_run": "higher raw replay projection selects the side",
            "overclaiming_guard": "proxy for the model projection only; not human memorability",
        },
        {
            "name": "vjepa_centroid_margin",
            "locally_available_now": True,
            "differentiability": "black-box embedding scorer for current use; differentiable only inside a separately supported model pipeline",
            "required_inputs": ["generated MP4", "exact V-JEPA `.npz` features"],
            "role_in_dry_run": "positive-minus-control centroid margin within each analysis tier",
            "overclaiming_guard": "representation agreement signal only; not a validated reward model",
        },
        {
            "name": "clip_seed_video_preservation",
            "locally_available_now": True,
            "differentiability": "black-box embedding similarity in this repo context",
            "required_inputs": ["generated MP4", "seed image", "committed CLIP diagnostic summary"],
            "role_in_dry_run": "higher seed-video CLIP cosine selects the side as a preservation guardrail",
            "overclaiming_guard": "preservation/semantic guardrail; can disagree with TRIBE and content-pocket labels",
        },
        {
            "name": "composite_proxy_score",
            "locally_available_now": True,
            "differentiability": "inherits black-box status of component scorers",
            "formula": "z(TRIBE) + 0.5*z(V-JEPA centroid margin) + 0.25*z(CLIP seed-video preservation), z-scored within tier",
            "required_inputs": ["all component scores above"],
            "role_in_dry_run": "illustrates whether a simple composite would rerank existing candidates differently",
            "overclaiming_guard": "candidate-selection heuristic only; no validation claim",
        },
    ]


def feasible_loops() -> list[dict[str, Any]]:
    """Map candidate guidance loops to current repo feasibility."""
    return [
        {
            "loop": "candidate_reranking",
            "status": "feasible_now",
            "reason": "existing MP4s can be scored by TRIBE, V-JEPA, and committed CLIP summaries without generating new pixels",
        },
        {
            "loop": "evolutionary_selection_over_generated_candidates",
            "status": "feasible_now_with_existing_or_new_generated_batches",
            "reason": "selection can operate on candidate pools after generation, while keeping proxy-only language",
        },
        {
            "loop": "prompt_search",
            "status": "blocked_for_current_svd_runner",
            "reason": "do not treat prompt rewrites as pixel-affecting unless the generator is prompt-conditioned in that runner",
        },
        {
            "loop": "latent_or_guidance_optimization",
            "status": "blocked_until_generator_exposes_pixel_affecting_controls",
            "reason": "requires a generator path where latent, noise, guidance, or conditioning changes actually affect MP4 pixels",
        },
    ]


def stop_rules() -> list[str]:
    """Safety gates for this feasibility lane."""
    return [
        "Stop if the task is framed as proving human memorability or replacing human/BMD validation.",
        "Stop if the frozen 24-task content-pocket stimulus set would be edited by a proxy experiment.",
        "Stop if prompt text is varied in a runner where prompt text does not affect generated pixels.",
        "Stop if V-JEPA, CLIP, saliency, or TRIBE outputs are described as validated human reward models.",
        "Stop before launch if MP4s, V-JEPA features, or visual-gate statuses are missing for the intended candidate pool.",
    ]


def launch_blockers(resolution: dict[str, Any]) -> list[str]:
    """Convert missing local artifacts into concrete launch blockers."""
    blockers: list[str] = []
    if resolution["mp4_found_side_observations"] != resolution["n_side_observations"]:
        blockers.append("Resolve all referenced MP4 paths before a scored reranking run.")
    if (
        resolution["vjepa_feature_found_side_observations"]
        != resolution["n_side_observations"]
    ):
        blockers.append("Extract exact V-JEPA features for every candidate side.")
    if (
        resolution["clip_seed_video_found_side_observations"]
        != resolution["n_side_observations"]
    ):
        blockers.append("Provide CLIP preservation summaries for every candidate side.")
    if resolution["visual_first_status_counts"].get("unknown"):
        blockers.append("Load replay visual-gate reports for every candidate side.")
    if not blockers:
        blockers.append("No local artifact blocker for this dry run; scientific overclaiming gates still apply.")
    return blockers


def build_payload(
    *,
    tasks_path: Path,
    task_doc: dict[str, Any],
    records: list[SideRecord],
    decisions: list[dict[str, Any]],
    embedding_summary_paths: dict[str, Path],
    replay_report_paths: dict[str, Path],
    vjepa_feature_dirs: dict[str, Path],
) -> dict[str, Any]:
    """Build the machine-readable dry-run payload."""
    resolution = data_resolution(task_doc, records)
    return {
        "schema_version": "neural_response_guided_generation_dry_run.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "lane": "parallel_satellite_feasibility_track",
            "no_generation": True,
            "proxy_only": True,
            "modifies_frozen_validation_set": False,
            "human_memorability_validation_claim": False,
        },
        "source_inputs": {
            "pairwise_tasks": logical_path(tasks_path),
            "embedding_summaries": {
                key: logical_path(path)
                for key, path in sorted(embedding_summary_paths.items())
            },
            "replay_reports": {
                key: logical_path(path)
                for key, path in sorted(replay_report_paths.items())
            },
            "vjepa_feature_dirs": {
                key: logical_path(path)
                for key, path in sorted(vjepa_feature_dirs.items())
            },
        },
        "data_resolution": resolution,
        "reward_components": reward_components(),
        "feasible_loops": feasible_loops(),
        "pairwise_proxy_results": decisions,
        "aggregate": aggregate_decisions(decisions),
        "smallest_safe_next_spike": [
            "Run this proxy-only scorer on a larger generated candidate batch with held-out naming and no human-validation language.",
            "Review disagreement cases where CLIP preservation picks controls but TRIBE/V-JEPA pick content-pocket candidates.",
            "Only after proxy behavior is documented, decide whether a new candidate-generation batch is worth human or measured-BMD evaluation.",
        ],
        "stop_rules": stop_rules(),
        "launch_blockers": launch_blockers(resolution),
        "relation_to_content_pocket_validation": (
            "The frozen 24-task set is used here only as an existing target for a "
            "no-generation proxy-agreement dry run. It remains separate from the "
            "manual MP4 screening, hosted-video, forced-choice, and measured-BMD "
            "validation lane."
        ),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    """Build the human-readable Markdown report."""
    aggregate = payload["aggregate"]["by_metric"]
    resolution = payload["data_resolution"]
    visual_status_counts = json.dumps(
        resolution["visual_first_status_counts"],
        sort_keys=True,
    )
    lines = [
        "# Neural-Response-Guided Generation Dry-Run Spike",
        "",
        "Date: 2026-06-08",
        "",
        "## Scope",
        "",
        "This is a no-generation, proxy-only feasibility dry run. It does not change the frozen content-pocket validation set and does not claim that TRIBE, V-JEPA, CLIP, saliency, or any composite score validates human memorability.",
        "",
        "## Local Signal Availability",
        "",
        f"- Frozen pairwise tasks: {resolution['n_tasks']} tasks, {resolution['n_side_observations']} side observations, {resolution['n_unique_mp4_paths']} unique MP4 paths.",
        f"- MP4s resolved locally: {resolution['mp4_found_unique_paths']} / {resolution['n_unique_mp4_paths']} unique paths.",
        f"- V-JEPA features resolved: {resolution['vjepa_feature_found_side_observations']} / {resolution['n_side_observations']} side observations.",
        f"- CLIP seed-video preservation scores resolved: {resolution['clip_seed_video_found_side_observations']} / {resolution['n_side_observations']} side observations.",
        f"- Visual first-frame statuses: {visual_status_counts}.",
        "",
        "## Pairwise Proxy Outcomes",
        "",
        "| Proxy | Decisions | Selects content-pocket target | Rate | Primary rate | Exploratory rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric, row in aggregate.items():
        primary = row["by_tier"].get("primary", {})
        exploratory = row["by_tier"].get("exploratory_boundary", {})
        lines.append(
            "| "
            f"{metric} | {row['n_decisions']} | "
            f"{row['selects_content_pocket_target']} | "
            f"{format_rate(row['target_selection_rate'])} | "
            f"{format_rate(primary.get('target_selection_rate'))} | "
            f"{format_rate(exploratory.get('target_selection_rate'))} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: these are proxy agreement counts over existing clips, not validation. CLIP seed-video preservation is intentionally listed as a guardrail because it can select hard negatives when those clips preserve the seed image more strongly.",
            "",
            "## Disagreement Examples",
            "",
        ]
    )
    examples = payload["aggregate"]["disagreement_examples"]
    if not examples:
        lines.append("- No proxy disagreement examples were found.")
    for example in examples:
        winner_sides = json.dumps(example["winner_sides"], sort_keys=True)
        lines.append(
            "- "
            f"{example['task_id']} ({example['analysis_tier']}): "
            f"{example['target_pocket']} vs {example['control_pocket']}; "
            f"winner sides {winner_sides}."
        )
    lines.extend(
        [
            "",
            "## Feasible Loops",
            "",
        ]
    )
    for loop in payload["feasible_loops"]:
        lines.append(f"- {loop['loop']}: {loop['status']}. {loop['reason']}")
    lines.extend(
        [
            "",
            "## Smallest Safe Next Spike",
            "",
            *[f"- {item}" for item in payload["smallest_safe_next_spike"]],
            "",
            "## Stop Rules",
            "",
            *[f"- {item}" for item in payload["stop_rules"]],
            "",
            "## Launch Blockers",
            "",
            *[f"- {item}" for item in payload["launch_blockers"]],
            "",
            "## Relation To Content-Pocket Validation",
            "",
            payload["relation_to_content_pocket_validation"],
            "",
        ]
    )
    return "\n".join(lines)


def format_rate(value: Any) -> str:
    """Format a nullable rate as a compact percentage."""
    if value is None:
        return "n/a"
    return f"{100 * float(value):.1f}%"


def build_dry_run(
    *,
    tasks_path: Path,
    data_roots: list[Path],
    embedding_summary_paths: dict[str, Path],
    replay_report_paths: dict[str, Path],
    vjepa_feature_dirs: dict[str, Path],
) -> tuple[dict[str, Any], str]:
    """Run the no-generation proxy dry run and return JSON plus Markdown."""
    task_doc = load_json(tasks_path)
    records = build_side_records(
        task_doc=task_doc,
        data_roots=data_roots,
        embedding_lookup=load_embedding_lookup(embedding_summary_paths),
        replay_status_lookup=load_replay_status_lookup(replay_report_paths),
        vjepa_feature_dirs=vjepa_feature_dirs,
    )
    assign_vjepa_margins(records)
    decisions = pairwise_decisions(records)
    payload = build_payload(
        tasks_path=tasks_path,
        task_doc=task_doc,
        records=records,
        decisions=decisions,
        embedding_summary_paths=embedding_summary_paths,
        replay_report_paths=replay_report_paths,
        vjepa_feature_dirs=vjepa_feature_dirs,
    )
    return payload, build_markdown(payload)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=[Path(".")],
        help="Repo root used to resolve task MP4 paths; repeatable.",
    )
    parser.add_argument(
        "--embedding-summary",
        action="append",
        default=[],
        help="source_pool=path for committed embedding summaries.",
    )
    parser.add_argument(
        "--replay-report",
        action="append",
        default=[],
        help="source_pool=path for replay visual-gate reports.",
    )
    parser.add_argument(
        "--vjepa-feature-dir",
        action="append",
        default=[],
        help="source_pool=path for exact V-JEPA feature `.npz` directories.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    embedding_paths = default_embedding_summaries()
    embedding_paths.update(parse_mapping(args.embedding_summary))
    payload, markdown = build_dry_run(
        tasks_path=args.tasks,
        data_roots=[path.expanduser() for path in args.data_root],
        embedding_summary_paths=embedding_paths,
        replay_report_paths=parse_mapping(args.replay_report),
        vjepa_feature_dirs=parse_mapping(args.vjepa_feature_dir),
    )
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
