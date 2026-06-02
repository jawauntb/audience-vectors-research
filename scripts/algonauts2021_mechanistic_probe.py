"""Algonauts 2021 scaffold for open-model brain-encoding interventions.

Why this exists
---------------
TRIBE is useful but opaque. Algonauts 2021 gives us a smaller, open baseline:
AlexNet video activations -> linear fMRI encoding models. That makes it a good
place to run reviewer-preferred interventions on model features rather than only
on TRIBE's output embedding.

This script does three things:

1. ``check``: verify local readiness and write a status report.
2. ``validate``: fit an encoding model from precomputed Algonauts/AlexNet PCA
   features to a subject/ROI and report validation correlation.
3. ``component-ablation``: remove individual PCA components and report which
   ones matter most for fMRI encoding.
4. ``direction-ablation``: given per-video scalar scores, learn a contrastive
   feature direction on the encoding-train split, ablate it, and measure the
   drop in fMRI encoding performance.

The Algonauts data itself is form-gated, so the default check mode is designed
to be useful before the dataset is present.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALGONAUTS_CHALLENGE_URL = "https://algonautsproject.com/2021/challenge.html"
ALGONAUTS_DEVKIT_URL = (
    "https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/" "Algonauts2021_devkit"
)

DEFAULT_DATA_ROOT = Path("data/raw/algonauts2021")
DEFAULT_DEVKIT_DIR = Path("data/external/Algonauts2021_devkit")
DEFAULT_REPORT_JSON = Path("data/reports/algonauts2021_status.json")
DEFAULT_REPORT_MD = Path("data/reports/algonauts2021_status.md")
MINI_ROIS = ("V1", "V2", "V3", "V4", "LOC", "EBA", "FFA", "STS", "PPA")


def _exists(path: Path) -> bool:
    return path.exists()


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _git_sha(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _pca_dir(activation_dir: Path) -> Path:
    if activation_dir.name == "pca_100":
        return activation_dir
    return activation_dir / "pca_100"


def _track_for_roi(roi: str) -> str:
    return "full_track" if roi == "WB" else "mini_track"


def _pickle_load_latin1(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh, encoding="latin1")


def _load_fmri(fmri_dir: Path, subject: str, roi: str) -> np.ndarray:
    roi_path = fmri_dir / _track_for_roi(roi) / subject / f"{roi}.pkl"
    if not roi_path.exists():
        raise FileNotFoundError(f"missing fMRI file: {roi_path}")

    payload = _pickle_load_latin1(roi_path)
    if not isinstance(payload, dict) or "train" not in payload:
        raise ValueError(f"unexpected Algonauts fMRI payload in {roi_path}")

    train = np.asarray(payload["train"], dtype=np.float32)
    if train.ndim == 3:
        return np.asarray(train.mean(axis=1), dtype=np.float32)
    if train.ndim == 2:
        return train
    raise ValueError(f"unexpected fMRI train shape for {roi_path}: {train.shape}")


def _load_train_activations(activation_dir: Path, layer: str) -> np.ndarray:
    path = _pca_dir(activation_dir) / f"train_{layer}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing activation file: {path}")
    activations = np.load(path)
    if activations.ndim != 2:
        raise ValueError(f"expected 2D activations in {path}, got {activations.shape}")
    return np.asarray(activations, dtype=np.float32)


def _vectorized_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    numerator = (x_centered * y_centered).sum(axis=0)
    denominator = np.sqrt((x_centered**2).sum(axis=0) * (y_centered**2).sum(axis=0))
    return numerator / np.maximum(denominator, 1e-8)


def _fit_predict_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    *,
    alpha: float,
    batch_size: int,
) -> np.ndarray:
    pred = np.zeros((val_x.shape[0], train_y.shape[1]), dtype=np.float32)
    for start in range(0, train_y.shape[1], batch_size):
        end = min(start + batch_size, train_y.shape[1])
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(train_x, train_y[:, start:end])
        pred[:, start:end] = np.asarray(model.predict(val_x), dtype=np.float32)
    return pred


def _encoding_eval(
    activations: np.ndarray,
    fmri: np.ndarray,
    *,
    train_n: int,
    alpha: float,
    batch_size: int,
) -> dict[str, Any]:
    if activations.shape[0] < train_n + 1:
        raise ValueError(
            f"need at least {train_n + 1} activation rows, got {activations.shape[0]}"
        )
    if fmri.shape[0] < activations.shape[0]:
        raise ValueError(
            f"fMRI rows {fmri.shape[0]} < activation rows {activations.shape[0]}"
        )

    train_x, val_x, train_y, val_y = _split_scaled(activations, fmri, train_n)
    pred = _fit_predict_ridge(
        train_x,
        train_y,
        val_x,
        alpha=alpha,
        batch_size=batch_size,
    )
    corr = _vectorized_correlation(val_y, pred)
    return _summarize_corr(corr) | {
        "n_train": int(train_x.shape[0]),
        "n_val": int(val_x.shape[0]),
        "feature_dim": int(train_x.shape[1]),
        "n_voxels": int(train_y.shape[1]),
        "alpha": float(alpha),
    }


def _split_scaled(
    activations: np.ndarray,
    fmri: np.ndarray,
    train_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x_raw = activations[:train_n]
    val_x_raw = activations[train_n : fmri.shape[0]]
    train_y = fmri[:train_n]
    val_y = fmri[train_n : fmri.shape[0]]

    scaler = StandardScaler()
    train_x = np.asarray(scaler.fit_transform(train_x_raw), dtype=np.float32)
    val_x = np.asarray(scaler.transform(val_x_raw), dtype=np.float32)
    return train_x, val_x, train_y, val_y


def _summarize_corr(corr: np.ndarray) -> dict[str, Any]:
    finite = corr[np.isfinite(corr)]
    if finite.size == 0:
        return {
            "mean_corr": None,
            "median_corr": None,
            "p10_corr": None,
            "p90_corr": None,
            "n_finite_voxels": 0,
        }
    return {
        "mean_corr": float(finite.mean()),
        "median_corr": float(np.median(finite)),
        "p10_corr": float(np.quantile(finite, 0.10)),
        "p90_corr": float(np.quantile(finite, 0.90)),
        "n_finite_voxels": int(finite.size),
    }


def _fit_contrastive_direction(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    top_frac: float,
) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(round(len(scores) * top_frac)))
    neg = features[order[:n_each]].mean(axis=0)
    pos = features[order[-n_each:]].mean(axis=0)
    direction = pos - neg
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("contrastive direction has near-zero norm")
    return np.asarray(direction / norm, dtype=np.float32)


def _ablate_direction(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return features - np.outer(features @ direction, direction)


def _load_scores(path: Path, *, n_rows: int, index_base: int) -> np.ndarray:
    if path.suffix == ".npy":
        scores = np.asarray(np.load(path), dtype=np.float32)
        if scores.ndim != 1:
            raise ValueError(f"expected 1D score array, got {scores.shape}")
        if scores.shape[0] < n_rows:
            raise ValueError(f"score array has {scores.shape[0]} rows, need {n_rows}")
        return scores[:n_rows]

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            if len(payload) < n_rows:
                raise ValueError(f"score list has {len(payload)} rows, need {n_rows}")
            return np.asarray(payload[:n_rows], dtype=np.float32)
        if isinstance(payload, dict):
            return _scores_from_mapping(payload, n_rows=n_rows, index_base=index_base)
        raise ValueError(f"unsupported JSON score payload: {type(payload)}")

    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        mapping = _csv_score_mapping(rows)
        return _scores_from_mapping(mapping, n_rows=n_rows, index_base=index_base)

    raise ValueError("scores must be .npy, .json, or .csv")


def _csv_score_mapping(rows: list[dict[str, str]]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for row in rows:
        key = row.get("video_id") or row.get("id") or row.get("index") or row.get("row")
        val = row.get("score") or row.get("memorability") or row.get("value")
        if key is None or val is None:
            raise ValueError(
                "CSV scores need columns like video_id/score, id/score, or index/score"
            )
        mapping[str(key)] = float(val)
    return mapping


def _scores_from_mapping(
    mapping: dict[str, Any],
    *,
    n_rows: int,
    index_base: int,
) -> np.ndarray:
    scores = np.full(n_rows, np.nan, dtype=np.float32)
    for raw_key, raw_val in mapping.items():
        index = _video_index(raw_key, index_base=index_base)
        if 0 <= index < n_rows:
            scores[index] = float(raw_val)
    if np.isnan(scores).any():
        missing = int(np.isnan(scores).sum())
        raise ValueError(f"score file is missing {missing} of {n_rows} training rows")
    return scores


def _video_index(raw: str, *, index_base: int) -> int:
    text = str(raw)
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"could not parse numeric video index from {raw!r}")
    return int(match.group(1)) - index_base


def _layout_status(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root
    activation_dir = args.activation_dir or data_root / "alexnet"
    fmri_dir = args.fmri_dir or data_root / "participants_data_v2021"
    video_dir = data_root / "AlgonautsVideos268_All_30fpsmax"
    pca_dir = _pca_dir(activation_dir)

    package_names = (
        "numpy",
        "sklearn",
        "torch",
        "torchvision",
        "cv2",
        "decord",
        "nilearn",
        "nibabel",
    )
    packages = {name: _package_available(name) for name in package_names}
    train_layer_5 = pca_dir / "train_layer_5.npy"
    test_layer_5 = pca_dir / "test_layer_5.npy"
    has_layer5_pca = train_layer_5.exists() and test_layer_5.exists()
    blockers = _blockers(
        devkit_dir=args.devkit_dir,
        video_dir=video_dir,
        fmri_dir=fmri_dir,
        train_layer_5=train_layer_5,
        test_layer_5=test_layer_5,
        packages=packages,
    )

    video_count = len(list(video_dir.glob("*"))) if video_dir.exists() else 0
    fmri_subjects = _subject_dirs(fmri_dir / "mini_track")
    status = {
        "challenge_url": ALGONAUTS_CHALLENGE_URL,
        "devkit_url": ALGONAUTS_DEVKIT_URL,
        "devkit_dir": str(args.devkit_dir),
        "devkit_exists": _exists(args.devkit_dir),
        "devkit_sha": _git_sha(args.devkit_dir),
        "data_root": str(data_root),
        "video_dir": str(video_dir),
        "video_dir_exists": video_dir.exists(),
        "video_file_count": video_count,
        "fmri_dir": str(fmri_dir),
        "fmri_dir_exists": fmri_dir.exists(),
        "fmri_subjects_detected": fmri_subjects,
        "activation_dir": str(activation_dir),
        "pca_dir": str(pca_dir),
        "train_layer_5_exists": train_layer_5.exists(),
        "test_layer_5_exists": test_layer_5.exists(),
        "ready_for_feature_probe": has_layer5_pca,
        "packages": packages,
        "ready_for_feature_extraction": bool(
            has_layer5_pca
            or (
                args.devkit_dir.exists()
                and video_dir.exists()
                and packages["torchvision"]
                and packages["cv2"]
                and packages["decord"]
            )
        ),
        "ready_for_encoding_validation": bool(
            fmri_dir.exists() and train_layer_5.exists() and packages["sklearn"]
        ),
        "blockers": blockers,
        "next_commands": _next_commands(
            data_root=data_root, devkit_dir=args.devkit_dir
        ),
    }
    return status


def _subject_dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("sub")
    )


def _blockers(
    *,
    devkit_dir: Path,
    video_dir: Path,
    fmri_dir: Path,
    train_layer_5: Path,
    test_layer_5: Path,
    packages: dict[str, bool],
) -> list[str]:
    blockers = []
    if not devkit_dir.exists():
        blockers.append("Official Algonauts2021_devkit is not cloned.")
    if not video_dir.exists():
        blockers.append(
            "Video directory is missing; download data via the official form."
        )
    if not fmri_dir.exists():
        blockers.append(
            "Legacy Algonauts participants_data_v2021 is missing. The old form is "
            "offline; use BOLD Moments/OpenNeuro fMRI for measured-brain analyses."
        )
    if not train_layer_5.exists() or not test_layer_5.exists():
        blockers.append(
            "AlexNet PCA features are missing; run the devkit feature extractor."
        )
    if (not train_layer_5.exists() or not test_layer_5.exists()) and (
        not packages["torchvision"] or not packages["cv2"] or not packages["decord"]
    ):
        blockers.append(
            "Local video feature extraction dependencies are incomplete "
            "(torchvision/cv2/decord). Validation can still run if PCA features exist."
        )
    return blockers


def _next_commands(*, data_root: Path, devkit_dir: Path) -> dict[str, str]:
    video_dir = data_root / "AlgonautsVideos268_All_30fpsmax"
    activation_dir = data_root / "alexnet"
    return {
        "feature_extraction": (
            "uv run python scripts/extract_boldmoments_alexnet_layer5.py "
            f"--video-dir {video_dir} --save-dir {activation_dir}"
        ),
        "official_devkit_feature_extraction": (
            "uv run python "
            f"{devkit_dir}/feature_extraction/generate_features_alexnet.py "
            f"-vdir {video_dir} -sdir {activation_dir}"
        ),
        "validation": (
            "uv run python scripts/algonauts2021_mechanistic_probe.py validate "
            "--subject sub04 --roi EBA --layer layer_5"
        ),
        "component_ablation": (
            "uv run python scripts/algonauts2021_mechanistic_probe.py "
            "component-ablation --subject sub04 --roi EBA --layer layer_5"
        ),
        "direction_ablation": (
            "uv run python scripts/algonauts2021_mechanistic_probe.py "
            "direction-ablation --subject sub04 --roi EBA --layer layer_5 "
            "--scores path/to/algonauts_video_scores.csv"
        ),
    }


def _write_reports(payload: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2))
    md_out.write_text(_markdown_report(payload))


def _markdown_report(payload: dict[str, Any]) -> str:
    blockers = payload.get("blockers") or []
    commands = payload.get("next_commands") or {}
    package_lines = [
        f"- `{name}`: {'ok' if ok else 'missing'}"
        for name, ok in (payload.get("packages") or {}).items()
    ]
    blocker_lines = [f"- {item}" for item in blockers] or ["- none"]
    command_lines = [f"- `{name}`: `{cmd}`" for name, cmd in commands.items()]
    return "\n".join(
        [
            "# Algonauts 2021 Open-Model Probe Status",
            "",
            f"- Challenge: {payload.get('challenge_url')}",
            f"- Devkit: {payload.get('devkit_url')}",
            f"- Devkit SHA: `{payload.get('devkit_sha') or 'not detected'}`",
            f"- Data root: `{payload.get('data_root')}`",
            f"- Videos detected: {payload.get('video_file_count', 0)}",
            f"- fMRI subjects detected: {', '.join(payload.get('fmri_subjects_detected') or []) or 'none'}",
            f"- Ready for feature extraction: {payload.get('ready_for_feature_extraction')}",
            f"- Ready for feature probe: {payload.get('ready_for_feature_probe')}",
            f"- Ready for encoding validation: {payload.get('ready_for_encoding_validation')}",
            "",
            "## Packages",
            *package_lines,
            "",
            "## Blockers",
            *blocker_lines,
            "",
            "## Next Commands",
            *command_lines,
            "",
        ]
    )


def cmd_check(args: argparse.Namespace) -> None:
    payload = _layout_status(args)
    _write_reports(payload, args.out_json, args.out_md)
    print(f"[algonauts] wrote {args.out_json}")
    print(f"[algonauts] wrote {args.out_md}")
    if payload["blockers"]:
        print("[algonauts] blockers:")
        for blocker in payload["blockers"]:
            print(f"  - {blocker}")
    else:
        print("[algonauts] ready")


def cmd_validate(args: argparse.Namespace) -> None:
    activation_dir = args.activation_dir or args.data_root / "alexnet"
    fmri_dir = args.fmri_dir or args.data_root / "participants_data_v2021"
    activations = _load_train_activations(activation_dir, args.layer)
    fmri = _load_fmri(fmri_dir, args.subject, args.roi)
    result = _encoding_eval(
        activations,
        fmri,
        train_n=args.train_n,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )
    result |= {
        "mode": "validate",
        "subject": args.subject,
        "roi": args.roi,
        "layer": args.layer,
        "activation_dir": str(activation_dir),
        "fmri_dir": str(fmri_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[algonauts] mean validation r = {_format_float(result['mean_corr'])}")
    print(f"[algonauts] wrote {args.output}")


def cmd_component_ablation(args: argparse.Namespace) -> None:
    activation_dir = args.activation_dir or args.data_root / "alexnet"
    fmri_dir = args.fmri_dir or args.data_root / "participants_data_v2021"
    activations = _load_train_activations(activation_dir, args.layer)
    fmri = _load_fmri(fmri_dir, args.subject, args.roi)
    train_x, val_x, train_y, val_y = _split_scaled(activations, fmri, args.train_n)

    baseline_pred = _fit_predict_ridge(
        train_x,
        train_y,
        val_x,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )
    baseline = _summarize_corr(_vectorized_correlation(val_y, baseline_pred))
    baseline_mean = _require_float(baseline["mean_corr"], "baseline mean_corr")

    n_components = min(args.components, train_x.shape[1])
    component_results = []
    for component in range(n_components):
        train_masked = train_x.copy()
        val_masked = val_x.copy()
        train_masked[:, component] = 0.0
        val_masked[:, component] = 0.0
        pred = _fit_predict_ridge(
            train_masked,
            train_y,
            val_masked,
            alpha=args.alpha,
            batch_size=args.batch_size,
        )
        summary = _summarize_corr(_vectorized_correlation(val_y, pred))
        mean_corr = _require_float(
            summary["mean_corr"], f"component {component} mean_corr"
        )
        component_results.append(
            {
                "component": component,
                "mean_corr": mean_corr,
                "delta_vs_baseline": mean_corr - baseline_mean,
                **summary,
            }
        )

    component_results.sort(key=lambda row: row["delta_vs_baseline"])
    result = {
        "mode": "component-ablation",
        "subject": args.subject,
        "roi": args.roi,
        "layer": args.layer,
        "activation_dir": str(activation_dir),
        "fmri_dir": str(fmri_dir),
        "n_components_tested": n_components,
        "baseline": baseline,
        "components_ranked_by_encoding_drop": component_results,
        "interpretation": (
            "More negative delta_vs_baseline means the PCA component is more important "
            "for this open AlexNet->fMRI encoding model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[algonauts] baseline mean r = {baseline_mean:+.4f}")
    print(f"[algonauts] tested {n_components} components")
    print(f"[algonauts] wrote {args.output}")


def cmd_direction_ablation(args: argparse.Namespace) -> None:
    activation_dir = args.activation_dir or args.data_root / "alexnet"
    fmri_dir = args.fmri_dir or args.data_root / "participants_data_v2021"
    activations = _load_train_activations(activation_dir, args.layer)
    fmri = _load_fmri(fmri_dir, args.subject, args.roi)
    scores = _load_scores(
        args.scores, n_rows=fmri.shape[0], index_base=args.score_index_base
    )

    base = _encoding_eval(
        activations,
        fmri,
        train_n=args.train_n,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )
    scaler = StandardScaler()
    train_x = np.asarray(
        scaler.fit_transform(activations[: args.train_n]), dtype=np.float32
    )
    full_x = np.asarray(
        scaler.transform(activations[: fmri.shape[0]]), dtype=np.float32
    )
    direction = _fit_contrastive_direction(
        train_x,
        scores[: args.train_n],
        top_frac=args.top_frac,
    )
    ablated = _ablate_direction(full_x, direction)
    ablated_result = _encoding_eval(
        ablated,
        fmri,
        train_n=args.train_n,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )

    mean_base = _require_float(base["mean_corr"], "baseline mean_corr")
    mean_ablated = _require_float(ablated_result["mean_corr"], "ablated mean_corr")
    result = {
        "mode": "direction-ablation",
        "subject": args.subject,
        "roi": args.roi,
        "layer": args.layer,
        "scores": str(args.scores),
        "top_frac": float(args.top_frac),
        "baseline": base,
        "ablated": ablated_result,
        "mean_corr_delta": mean_ablated - mean_base,
        "direction_norm": 1.0,
        "interpretation": (
            "Negative delta means the supplied score direction carries information "
            "used by the open AlexNet->fMRI encoding model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[algonauts] baseline mean r = {mean_base:+.4f}")
    print(f"[algonauts] ablated  mean r = {mean_ablated:+.4f}")
    print(f"[algonauts] delta          = {mean_ablated - mean_base:+.4f}")
    print(f"[algonauts] wrote {args.output}")


def _require_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is missing")
    return float(value)


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=cmd_check)
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="write local readiness report")
    _add_common_paths(check)
    check.add_argument("--out-json", type=Path, default=DEFAULT_REPORT_JSON)
    check.add_argument("--out-md", type=Path, default=DEFAULT_REPORT_MD)
    check.set_defaults(func=cmd_check)

    validate = sub.add_parser("validate", help="fit AlexNet-PCA -> fMRI encoding")
    _add_common_paths(validate)
    _add_encoding_args(validate)
    validate.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/algonauts2021_validation.json"),
    )
    validate.set_defaults(func=cmd_validate)

    component = sub.add_parser(
        "component-ablation",
        help="ablate individual AlexNet-PCA components and rank encoding drops",
    )
    _add_common_paths(component)
    _add_encoding_args(component)
    component.add_argument("--components", type=int, default=20)
    component.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/algonauts2021_component_ablation.json"),
    )
    component.set_defaults(func=cmd_component_ablation)

    ablate = sub.add_parser(
        "direction-ablation",
        help="ablate a supplied score direction from AlexNet-PCA features",
    )
    _add_common_paths(ablate)
    _add_encoding_args(ablate)
    ablate.add_argument("--scores", type=Path, required=True)
    ablate.add_argument("--score-index-base", type=int, default=1)
    ablate.add_argument("--top-frac", type=float, default=0.30)
    ablate.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/algonauts2021_direction_ablation.json"),
    )
    ablate.set_defaults(func=cmd_direction_ablation)
    return parser


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--devkit-dir", type=Path, default=DEFAULT_DEVKIT_DIR)
    parser.add_argument("--activation-dir", type=Path, default=None)
    parser.add_argument("--fmri-dir", type=Path, default=None)


def _add_encoding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", default="sub04")
    parser.add_argument("--roi", default="EBA", choices=MINI_ROIS + ("WB",))
    parser.add_argument("--layer", default="layer_5")
    parser.add_argument("--train-n", type=int, default=900)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2000)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
