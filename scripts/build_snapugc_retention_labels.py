"""Build canonical SnapUGC/VQualA retention labels from official CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--dataset", default="SnapUGC")
    parser.add_argument("--metadata-id-column", default="Id")
    parser.add_argument("--score-id-column", default="Id")
    parser.add_argument("--ecr-column", default="ECR")
    parser.add_argument("--title-column", default="Title")
    parser.add_argument("--description-column", default="Description")
    parser.add_argument("--download-link-column", default="Download_link")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help=(
            "Optional local directory containing videos named by sample id. If a "
            "video is found, video_path uses the local path; otherwise it falls "
            "back to Download_link unless --require-local-video is set."
        ),
    )
    parser.add_argument(
        "--media-path-template",
        default=None,
        help=(
            "Optional template for video_path. Supports {sample_id}, {id}, "
            "{download_link}, {title}, and {description}; useful for Modal paths."
        ),
    )
    parser.add_argument("--require-local-video", action="store_true")
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument(
        "--allow-prediction-score-file",
        action="store_true",
        help=(
            "Allow score CSV filenames that look like predictions or baseline "
            "submissions. Without this flag, such files are blocked to avoid "
            "accidentally using model predictions as ground truth."
        ),
    )
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-distinct-ecr", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_snapugc_retention_labels(
        metadata_csv=args.metadata_csv,
        scores_csv=args.scores_csv,
        output_csv=args.output_csv,
        dataset=args.dataset,
        metadata_id_column=args.metadata_id_column,
        score_id_column=args.score_id_column,
        ecr_column=args.ecr_column,
        title_column=args.title_column,
        description_column=args.description_column,
        download_link_column=args.download_link_column,
        video_root=args.video_root,
        media_path_template=args.media_path_template,
        require_local_video=args.require_local_video,
        allow_missing_labels=args.allow_missing_labels,
        allow_prediction_score_file=args.allow_prediction_score_file,
        min_samples=args.min_samples,
        min_distinct_ecr=args.min_distinct_ecr,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_snapugc_builder_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ready_for_retention_audit"]:
        raise SystemExit(1)


def build_snapugc_retention_labels(
    *,
    metadata_csv: Path,
    scores_csv: Path,
    output_csv: Path,
    dataset: str = "SnapUGC",
    metadata_id_column: str = "Id",
    score_id_column: str = "Id",
    ecr_column: str = "ECR",
    title_column: str = "Title",
    description_column: str = "Description",
    download_link_column: str = "Download_link",
    video_root: Path | None = None,
    media_path_template: str | None = None,
    require_local_video: bool = False,
    allow_missing_labels: bool = False,
    allow_prediction_score_file: bool = False,
    min_samples: int = 30,
    min_distinct_ecr: int = 3,
) -> dict[str, Any]:
    metadata_rows = read_csv_rows(metadata_csv)
    score_rows = read_csv_rows(scores_csv)
    score_index = index_scores(
        score_rows,
        scores_csv=scores_csv,
        score_id_column=score_id_column,
        ecr_column=ecr_column,
    )
    output_rows: list[dict[str, str]] = []
    missing_labels: list[str] = []
    missing_media: list[str] = []
    invalid_ecr: list[str] = []
    duplicate_metadata_ids = duplicate_ids(metadata_rows, metadata_id_column)

    for metadata_index, row in enumerate(metadata_rows):
        sample_id = required_cell(row, metadata_id_column, metadata_csv)
        score_row = score_index["by_id"].get(sample_id)
        if score_row is None:
            missing_labels.append(sample_id)
            if not allow_missing_labels:
                continue
        raw_ecr = score_row.get(ecr_column) if score_row is not None else None
        ecr = finite_float(raw_ecr)
        if ecr is None:
            invalid_ecr.append(sample_id)
            continue
        media_path = resolve_media_path(
            row,
            sample_id=sample_id,
            video_root=video_root,
            media_path_template=media_path_template,
            require_local_video=require_local_video,
            download_link_column=download_link_column,
            title_column=title_column,
            description_column=description_column,
        )
        if media_path is None:
            missing_media.append(sample_id)
            continue
        output_rows.append(
            {
                "sample_id": sample_id,
                "video_path": media_path,
                "ecr": format_float(ecr),
                "title": row.get(title_column, ""),
                "description": row.get(description_column, ""),
                "download_link": row.get(download_link_column, ""),
                "metadata_row_index": str(metadata_index),
                "score_row_index": (
                    str(score_row.get("__row_index__", ""))
                    if score_row is not None
                    else ""
                ),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_output_csv(output_csv, output_rows)
    ecr_values = [float(row["ecr"]) for row in output_rows]
    blocking_reasons = snapugc_builder_blocking_reasons(
        n_output_rows=len(output_rows),
        n_missing_labels=len(missing_labels),
        n_invalid_ecr=len(invalid_ecr),
        n_missing_media=len(missing_media),
        duplicate_metadata_ids=duplicate_metadata_ids,
        duplicate_score_ids=score_index["duplicate_ids"],
        allow_missing_labels=allow_missing_labels,
        prediction_score_file_detected=prediction_score_file_detected(scores_csv),
        allow_prediction_score_file=allow_prediction_score_file,
        min_samples=min_samples,
        min_distinct_ecr=min_distinct_ecr,
        ecr_values=ecr_values,
    )
    return {
        "schema_version": 1,
        "experiment": "snapugc_retention_label_builder",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "metadata_csv": str(metadata_csv),
        "scores_csv": str(scores_csv),
        "output_csv": str(output_csv),
        "metadata_csv_sha256": sha256(metadata_csv.read_bytes()).hexdigest(),
        "scores_csv_sha256": sha256(scores_csv.read_bytes()).hexdigest(),
        "output_csv_sha256": sha256(output_csv.read_bytes()).hexdigest(),
        "columns": {
            "metadata_id_column": metadata_id_column,
            "score_id_column": score_id_column,
            "ecr_column": ecr_column,
            "title_column": title_column,
            "description_column": description_column,
            "download_link_column": download_link_column,
        },
        "video_root": str(video_root) if video_root is not None else None,
        "media_path_template": media_path_template,
        "require_local_video": require_local_video,
        "allow_missing_labels": allow_missing_labels,
        "allow_prediction_score_file": allow_prediction_score_file,
        "prediction_score_file_detected": prediction_score_file_detected(scores_csv),
        "n_metadata_rows": len(metadata_rows),
        "n_score_rows": len(score_rows),
        "n_output_rows": len(output_rows),
        "n_missing_labels": len(missing_labels),
        "missing_label_ids": missing_labels[:20],
        "n_invalid_ecr": len(invalid_ecr),
        "invalid_ecr_ids": invalid_ecr[:20],
        "n_missing_media": len(missing_media),
        "missing_media_ids": missing_media[:20],
        "n_duplicate_metadata_ids": len(duplicate_metadata_ids),
        "duplicate_metadata_ids": duplicate_metadata_ids[:20],
        "n_duplicate_score_ids": len(score_index["duplicate_ids"]),
        "duplicate_score_ids": score_index["duplicate_ids"][:20],
        "ecr_summary": summarize_values(ecr_values),
        "min_samples": min_samples,
        "min_distinct_ecr": min_distinct_ecr,
        "ready_for_retention_audit": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "recommended_next_commands": recommended_next_commands(output_csv),
        "claim_boundary": (
            "This builder only normalizes official SnapUGC-style metadata and "
            "ECR labels into the local canonical CSV. It does not verify label "
            "access rights, download videos, run TRIBE, or validate H2."
        ),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return [dict(row) for row in reader]


def index_scores(
    rows: list[dict[str, str]],
    *,
    scores_csv: Path,
    score_id_column: str,
    ecr_column: str,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, str]] = {}
    seen: Counter[str] = Counter()
    for index, row in enumerate(rows):
        sample_id = required_cell(row, score_id_column, scores_csv)
        required_cell(row, ecr_column, scores_csv)
        row = dict(row)
        row["__row_index__"] = str(index)
        seen[sample_id] += 1
        if sample_id not in by_id:
            by_id[sample_id] = row
    duplicate_score_ids = sorted(sample_id for sample_id, count in seen.items() if count > 1)
    return {"by_id": by_id, "duplicate_ids": duplicate_score_ids}


def resolve_media_path(
    row: dict[str, str],
    *,
    sample_id: str,
    video_root: Path | None,
    media_path_template: str | None,
    require_local_video: bool,
    download_link_column: str,
    title_column: str,
    description_column: str,
) -> str | None:
    download_link = row.get(download_link_column, "").strip()
    if media_path_template:
        return media_path_template.format(
            sample_id=sample_id,
            id=sample_id,
            download_link=download_link,
            title=row.get(title_column, ""),
            description=row.get(description_column, ""),
        )
    if video_root is not None:
        local_video = find_local_video(video_root, sample_id)
        if local_video is not None:
            return str(local_video)
        if require_local_video:
            return None
    return download_link or None


def find_local_video(video_root: Path, sample_id: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = video_root / f"{sample_id}{suffix}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def snapugc_builder_blocking_reasons(
    *,
    n_output_rows: int,
    n_missing_labels: int,
    n_invalid_ecr: int,
    n_missing_media: int,
    duplicate_metadata_ids: list[str],
    duplicate_score_ids: list[str],
    allow_missing_labels: bool,
    prediction_score_file_detected: bool,
    allow_prediction_score_file: bool,
    min_samples: int,
    min_distinct_ecr: int,
    ecr_values: list[float],
) -> list[str]:
    reasons: list[str] = []
    if prediction_score_file_detected and not allow_prediction_score_file:
        reasons.append(
            "scores CSV filename looks like baseline/prediction output, not "
            "ground-truth ECR labels"
        )
    if n_output_rows < min_samples:
        reasons.append(f"output row count {n_output_rows} is below minimum {min_samples}")
    if n_missing_labels and not allow_missing_labels:
        reasons.append(f"{n_missing_labels} metadata rows are missing ECR labels")
    if n_invalid_ecr:
        reasons.append(f"{n_invalid_ecr} rows have non-finite ECR")
    if n_missing_media:
        reasons.append(f"{n_missing_media} rows are missing usable media paths")
    if duplicate_metadata_ids:
        reasons.append(f"{len(duplicate_metadata_ids)} duplicate metadata ids found")
    if duplicate_score_ids:
        reasons.append(f"{len(duplicate_score_ids)} duplicate score ids found")
    if len(set(ecr_values)) < min_distinct_ecr:
        reasons.append(
            f"distinct ECR count {len(set(ecr_values))} is below minimum "
            f"{min_distinct_ecr}"
        )
    return reasons


def write_output_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "video_path",
        "ecr",
        "title",
        "description",
        "download_link",
        "metadata_row_index",
        "score_row_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_snapugc_builder_markdown(report: dict[str, Any]) -> str:
    summary = report["ecr_summary"]
    lines = [
        "# SnapUGC Retention Label Builder",
        "",
        "## Verdict",
        "",
        f"- Ready for retention audit: {report['ready_for_retention_audit']}",
        f"- Dataset: `{report['dataset']}`",
        f"- Metadata CSV: `{report['metadata_csv']}`",
        f"- Scores CSV: `{report['scores_csv']}`",
        f"- Output CSV: `{report['output_csv']}`",
        f"- Metadata rows: {report['n_metadata_rows']}",
        f"- Score rows: {report['n_score_rows']}",
        f"- Output rows: {report['n_output_rows']}",
        f"- Missing labels: {report['n_missing_labels']}",
        f"- Missing media: {report['n_missing_media']}",
        f"- ECR distinct values: {summary['n_distinct']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reasons"]
    lines.extend(f"- {reason}" for reason in blockers) if blockers else lines.append(
        "- none"
    )
    lines.extend(["", "## Recommended Next Commands", ""])
    lines.extend(f"```bash\n{command}\n```" for command in report["recommended_next_commands"])
    return "\n".join(lines) + "\n"


def recommended_next_commands(output_csv: Path) -> list[str]:
    label_audit = (
        "research_program/dopamine_detox_attention_capture/results/"
        "snapugc_retention_label_audit.json"
    )
    baseline_audit = (
        "research_program/dopamine_detox_attention_capture/results/"
        "snapugc_retention_baseline_audit.json"
    )
    return [
        (
            "uv run python scripts/audit_attention_capture_retention_labels.py "
            f"--labels-csv {output_csv} --dataset SnapUGC "
            "--sample-id-column sample_id --ground-truth-column ecr "
            "--media-path-column video_path --ground-truth-name ecr "
            f"--output-json {label_audit} "
            "--output-md research_program/dopamine_detox_attention_capture/"
            "results/snapugc_retention_label_audit.md"
        ),
        (
            "uv run python scripts/audit_attention_capture_retention_baselines.py "
            f"--labels-csv {output_csv} --label-audit {label_audit} "
            "--dataset SnapUGC --sample-id-column sample_id "
            "--ground-truth-column ecr --media-path-column video_path "
            "--ground-truth-name ecr "
            f"--output-json {baseline_audit} "
            "--output-md research_program/dopamine_detox_attention_capture/"
            "results/snapugc_retention_baseline_audit.md"
        ),
    ]


def duplicate_ids(rows: list[dict[str, str]], id_column: str) -> list[str]:
    counts = Counter(row.get(id_column, "").strip() for row in rows)
    return sorted(sample_id for sample_id, count in counts.items() if sample_id and count > 1)


def prediction_score_file_detected(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(token in lower_name for token in ("baseline", "prediction", "submission"))


def required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or value.strip() == "":
        raise ValueError(f"{path} has a row missing required column {column!r}")
    return value.strip()


def finite_float(value: object) -> float | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "n_distinct": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "n": len(values),
        "n_distinct": len(set(values)),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()
