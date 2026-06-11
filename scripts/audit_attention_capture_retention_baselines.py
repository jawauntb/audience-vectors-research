"""Run cheap SnapUGC/VQualA retention-label baseline checks before TRIBE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from audience_vectors.attention_capture import spearman_rho

ID_COLUMN_HINTS = ("sample_id", "video_id", "video", "id")
GROUND_TRUTH_HINTS = ("ecr", "completion", "retention", "engagement")
MEDIA_PATH_HINTS = ("media_path", "video_path", "path", "url", "video_url", "filepath")
TEXT_WORD_RE = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, default=None)
    parser.add_argument("--dataset", default="SnapUGC")
    parser.add_argument("--sample-id-column", default=None)
    parser.add_argument("--ground-truth-column", default=None)
    parser.add_argument("--media-path-column", default=None)
    parser.add_argument("--ground-truth-name", default=None)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-distinct-ground-truth", type=int, default=3)
    parser.add_argument("--min-feature-coverage", type=float, default=0.5)
    parser.add_argument("--min-feature-distinct", type=int, default=3)
    parser.add_argument("--min-baseline-abs-rho", type=float, default=0.10)
    parser.add_argument("--max-control-abs-rho", type=float, default=0.30)
    parser.add_argument("--max-reported-features", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_retention_baselines(
        labels_csv=args.labels_csv,
        dataset=args.dataset,
        sample_id_column=args.sample_id_column,
        ground_truth_column=args.ground_truth_column,
        media_path_column=args.media_path_column,
        ground_truth_name=args.ground_truth_name,
        label_audit=args.label_audit,
        min_samples=args.min_samples,
        min_distinct_ground_truth=args.min_distinct_ground_truth,
        min_feature_coverage=args.min_feature_coverage,
        min_feature_distinct=args.min_feature_distinct,
        min_baseline_abs_rho=args.min_baseline_abs_rho,
        max_control_abs_rho=args.max_control_abs_rho,
        max_reported_features=args.max_reported_features,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_retention_baseline_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ready_for_modal_slice"]:
        raise SystemExit(1)


def audit_retention_baselines(
    *,
    labels_csv: Path,
    dataset: str = "SnapUGC",
    sample_id_column: str | None = None,
    ground_truth_column: str | None = None,
    media_path_column: str | None = None,
    ground_truth_name: str | None = None,
    label_audit: Path | None = None,
    min_samples: int = 30,
    min_distinct_ground_truth: int = 3,
    min_feature_coverage: float = 0.5,
    min_feature_distinct: int = 3,
    min_baseline_abs_rho: float = 0.10,
    max_control_abs_rho: float = 0.30,
    max_reported_features: int = 50,
) -> dict[str, Any]:
    rows, header = read_csv_rows(labels_csv)
    resolved_sample_id = sample_id_column or choose_column(header, ID_COLUMN_HINTS)
    resolved_ground_truth = ground_truth_column or choose_column(
        header,
        GROUND_TRUTH_HINTS,
    )
    resolved_media_path = media_path_column or choose_column(header, MEDIA_PATH_HINTS)
    parsed = parse_label_rows(
        rows,
        sample_id_column=resolved_sample_id,
        ground_truth_column=resolved_ground_truth,
    )
    finite_ground_truth = {
        row["row_index"]: float(row["ground_truth"])
        for row in parsed
        if row["ground_truth"] is not None
    }
    valid_sample_ids = [
        str(row["sample_id"]) for row in parsed if row["sample_id"] is not None
    ]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(valid_sample_ids).items() if count > 1
    )
    label_audit_summary = validate_retention_label_audit(
        label_audit=label_audit,
        labels_csv=labels_csv,
        dataset=dataset,
        sample_id_column=resolved_sample_id,
        ground_truth_column=resolved_ground_truth,
    )
    blocking_reasons = retention_baseline_blocking_reasons(
        n_rows=len(rows),
        n_finite_ground_truth=len(finite_ground_truth),
        n_distinct_ground_truth=len(set(finite_ground_truth.values())),
        sample_id_column=resolved_sample_id,
        ground_truth_column=resolved_ground_truth,
        missing_sample_ids=[row for row in parsed if row["sample_id"] is None],
        duplicate_ids=duplicate_ids,
        min_samples=min_samples,
        min_distinct_ground_truth=min_distinct_ground_truth,
    )
    blocking_reasons.extend(label_audit_summary["blocking_reasons"])

    excluded_columns = {
        column
        for column in (resolved_sample_id, resolved_ground_truth, resolved_media_path)
        if column
    }
    feature_metrics = discover_feature_metrics(
        rows=rows,
        header=header,
        finite_ground_truth=finite_ground_truth,
        excluded_columns=excluded_columns,
        min_samples=min_samples,
        min_feature_coverage=min_feature_coverage,
        min_feature_distinct=min_feature_distinct,
    )
    feature_metrics = sorted(
        feature_metrics,
        key=lambda item: (float(item["abs_rho"]), int(item["n"])),
        reverse=True,
    )
    controls = negative_control_metrics(
        parsed=parsed,
        finite_ground_truth=finite_ground_truth,
    )
    control_warnings = [
        (
            f"negative control {control['feature']} has |rho|="
            f"{control['abs_rho']:.4f}, above {max_control_abs_rho:.4f}"
        )
        for control in controls
        if control["abs_rho"] >= max_control_abs_rho
    ]
    best_feature = feature_metrics[0] if feature_metrics else None
    baseline_signal_detected = bool(
        best_feature and best_feature["abs_rho"] >= min_baseline_abs_rho
    )
    warnings = retention_baseline_warnings(
        feature_metrics=feature_metrics,
        baseline_signal_detected=baseline_signal_detected,
        control_warnings=control_warnings,
        min_baseline_abs_rho=min_baseline_abs_rho,
    )
    ready_for_modal_slice = not blocking_reasons and not control_warnings
    return {
        "schema_version": 1,
        "experiment": "attention_capture_retention_baseline_audit",
        "dataset": dataset,
        "labels_csv": str(labels_csv),
        "labels_csv_sha256": sha256(labels_csv.read_bytes()).hexdigest(),
        "label_audit": label_audit_summary,
        "columns": header,
        "sample_id_column": resolved_sample_id,
        "ground_truth_column": resolved_ground_truth,
        "ground_truth_name": ground_truth_name or resolved_ground_truth,
        "media_path_column": resolved_media_path,
        "n_rows": len(rows),
        "n_finite_ground_truth": len(finite_ground_truth),
        "n_distinct_ground_truth": len(set(finite_ground_truth.values())),
        "n_duplicate_sample_ids": len(duplicate_ids),
        "duplicate_sample_ids": duplicate_ids[:20],
        "min_samples": min_samples,
        "min_distinct_ground_truth": min_distinct_ground_truth,
        "min_baseline_abs_rho": min_baseline_abs_rho,
        "max_control_abs_rho": max_control_abs_rho,
        "n_feature_metrics": len(feature_metrics),
        "best_feature": best_feature,
        "baseline_signal_detected": baseline_signal_detected,
        "negative_controls": controls,
        "ready_for_modal_slice": ready_for_modal_slice,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "feature_metrics": feature_metrics[:max_reported_features],
        "next_actions": next_actions(
            ready_for_modal_slice=ready_for_modal_slice,
            baseline_signal_detected=baseline_signal_detected,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
        ),
        "claim_boundary": (
            "This diagnostic uses cheap CSV metadata only. It can catch broken "
            "labels, leakage, or obvious metadata signal before Modal TRIBE "
            "extraction, but it does not validate the neural capture_score."
        ),
    }


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return [dict(row) for row in reader], list(reader.fieldnames)


def parse_label_rows(
    rows: list[dict[str, str]],
    *,
    sample_id_column: str | None,
    ground_truth_column: str | None,
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = cell(row, sample_id_column)
        ground_truth = finite_float(cell(row, ground_truth_column))
        parsed.append(
            {
                "row_index": idx,
                "sample_id": sample_id,
                "ground_truth": ground_truth,
            }
        )
    return parsed


def discover_feature_metrics(
    *,
    rows: list[dict[str, str]],
    header: list[str],
    finite_ground_truth: dict[int, float],
    excluded_columns: set[str],
    min_samples: int,
    min_feature_coverage: float,
    min_feature_distinct: int,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    n_valid_ground_truth = len(finite_ground_truth)
    for column in header:
        if column in excluded_columns:
            continue
        numeric_pairs = numeric_feature_pairs(
            rows=rows,
            column=column,
            finite_ground_truth=finite_ground_truth,
        )
        if feature_pair_ready(
            numeric_pairs,
            n_valid_ground_truth=n_valid_ground_truth,
            min_samples=min_samples,
            min_feature_coverage=min_feature_coverage,
            min_feature_distinct=min_feature_distinct,
        ):
            metric = feature_metric(
                feature=column,
                source_column=column,
                kind="numeric",
                pairs=numeric_pairs,
            )
            if metric is not None:
                metrics.append(metric)
            continue

        for kind, pairs in text_feature_pairs(
            rows=rows,
            column=column,
            finite_ground_truth=finite_ground_truth,
        ).items():
            if not feature_pair_ready(
                pairs,
                n_valid_ground_truth=n_valid_ground_truth,
                min_samples=min_samples,
                min_feature_coverage=min_feature_coverage,
                min_feature_distinct=min_feature_distinct,
            ):
                continue
            metric = feature_metric(
                feature=f"{column}_{kind}",
                source_column=column,
                kind=kind,
                pairs=pairs,
            )
            if metric is not None:
                metrics.append(metric)
    return metrics


def numeric_feature_pairs(
    *,
    rows: list[dict[str, str]],
    column: str,
    finite_ground_truth: dict[int, float],
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for idx, row in enumerate(rows):
        ground_truth = finite_ground_truth.get(idx)
        value = finite_float(cell(row, column))
        if ground_truth is None or value is None:
            continue
        pairs.append((value, ground_truth))
    return pairs


def text_feature_pairs(
    *,
    rows: list[dict[str, str]],
    column: str,
    finite_ground_truth: dict[int, float],
) -> dict[str, list[tuple[float, float]]]:
    char_pairs: list[tuple[float, float]] = []
    word_pairs: list[tuple[float, float]] = []
    for idx, row in enumerate(rows):
        ground_truth = finite_ground_truth.get(idx)
        value = cell(row, column)
        if ground_truth is None or value is None:
            continue
        char_pairs.append((float(len(value)), ground_truth))
        word_pairs.append((float(len(TEXT_WORD_RE.findall(value))), ground_truth))
    return {"char_count": char_pairs, "word_count": word_pairs}


def feature_pair_ready(
    pairs: list[tuple[float, float]],
    *,
    n_valid_ground_truth: int,
    min_samples: int,
    min_feature_coverage: float,
    min_feature_distinct: int,
) -> bool:
    if len(pairs) < min_samples:
        return False
    if n_valid_ground_truth and len(pairs) / n_valid_ground_truth < min_feature_coverage:
        return False
    return len({feature for feature, _ground_truth in pairs}) >= min_feature_distinct


def feature_metric(
    *,
    feature: str,
    source_column: str,
    kind: str,
    pairs: list[tuple[float, float]],
) -> dict[str, Any] | None:
    feature_values = [feature_value for feature_value, _ground_truth in pairs]
    ground_truth_values = [ground_truth for _feature_value, ground_truth in pairs]
    rho = spearman_rho(feature_values, ground_truth_values)
    if rho is None:
        return None
    return {
        "feature": feature,
        "source_column": source_column,
        "kind": kind,
        "n": len(pairs),
        "rho": rho,
        "abs_rho": abs(rho),
        "feature_n_distinct": len(set(feature_values)),
    }


def negative_control_metrics(
    *,
    parsed: list[dict[str, Any]],
    finite_ground_truth: dict[int, float],
) -> list[dict[str, Any]]:
    row_index_pairs: list[tuple[float, float]] = []
    hash_pairs: list[tuple[float, float]] = []
    for row in parsed:
        idx = int(row["row_index"])
        ground_truth = finite_ground_truth.get(idx)
        sample_id = row.get("sample_id")
        if ground_truth is None:
            continue
        row_index_pairs.append((float(idx), ground_truth))
        if isinstance(sample_id, str) and sample_id:
            hash_pairs.append((stable_uniform(sample_id), ground_truth))
    controls = [
        feature_metric(
            feature="row_index",
            source_column="__row_index__",
            kind="negative_control",
            pairs=row_index_pairs,
        ),
        feature_metric(
            feature="sample_id_hash_uniform",
            source_column="sample_id",
            kind="negative_control",
            pairs=hash_pairs,
        ),
    ]
    return [control for control in controls if control is not None]


def validate_retention_label_audit(
    *,
    label_audit: Path | None,
    labels_csv: Path,
    dataset: str,
    sample_id_column: str | None,
    ground_truth_column: str | None,
) -> dict[str, Any]:
    if label_audit is None:
        return {
            "path": None,
            "sha256": None,
            "experiment": None,
            "ready_for_manifest_alignment": None,
            "labels_csv_relation": None,
            "ground_truth_column": None,
            "ground_truth_name": None,
            "n_rows": None,
            "blocking_reasons": [],
        }

    payload = json.loads(label_audit.read_text(encoding="utf-8"))
    reasons: list[str] = []
    experiment = payload.get("experiment")
    if experiment != "attention_capture_retention_label_audit":
        reasons.append(
            "label audit experiment is not attention_capture_retention_label_audit"
        )
    if payload.get("ready_for_manifest_alignment") is not True:
        upstream_reasons = payload.get("blocking_reasons") or []
        reason_text = "; ".join(str(reason) for reason in upstream_reasons)
        reasons.append(f"label audit is not ready: {reason_text or 'unknown reason'}")

    relation = label_csv_relation(
        payload.get("labels_csv"),
        labels_csv,
        sample_id_column=sample_id_column,
    )
    if payload.get("labels_csv") and relation == "mismatch":
        reasons.append(
            "label audit labels_csv is neither the baseline labels_csv nor "
            "an exact row superset"
        )

    audit_dataset = str(payload.get("dataset") or "unknown")
    if dataset != "unknown" and audit_dataset not in ("unknown", dataset):
        reasons.append("label audit dataset differs from baseline dataset")

    audited_ground_truth = payload.get("ground_truth_name") or payload.get(
        "ground_truth_column"
    )
    if (
        isinstance(audited_ground_truth, str)
        and ground_truth_column
        and audited_ground_truth.lower() != ground_truth_column.lower()
    ):
        reasons.append("label audit ground truth differs from baseline ground truth")

    return {
        "path": str(label_audit),
        "sha256": sha256(label_audit.read_bytes()).hexdigest(),
        "experiment": experiment,
        "ready_for_manifest_alignment": payload.get("ready_for_manifest_alignment"),
        "labels_csv_relation": relation,
        "ground_truth_column": payload.get("ground_truth_column"),
        "ground_truth_name": payload.get("ground_truth_name"),
        "n_rows": payload.get("n_rows"),
        "blocking_reasons": reasons,
    }


def retention_baseline_blocking_reasons(
    *,
    n_rows: int,
    n_finite_ground_truth: int,
    n_distinct_ground_truth: int,
    sample_id_column: str | None,
    ground_truth_column: str | None,
    missing_sample_ids: list[dict[str, Any]],
    duplicate_ids: list[str],
    min_samples: int,
    min_distinct_ground_truth: int,
) -> list[str]:
    reasons: list[str] = []
    if sample_id_column is None:
        reasons.append("no sample-id column found")
    if ground_truth_column is None:
        reasons.append("no retention ground-truth column found")
    if n_rows < min_samples:
        reasons.append(f"row count {n_rows} is below minimum {min_samples}")
    if missing_sample_ids:
        reasons.append(f"{len(missing_sample_ids)} rows are missing sample ids")
    if duplicate_ids:
        reasons.append(f"{len(duplicate_ids)} duplicate sample ids found")
    if n_finite_ground_truth < min_samples:
        reasons.append(
            f"finite ground-truth count {n_finite_ground_truth} is below "
            f"minimum {min_samples}"
        )
    if n_distinct_ground_truth < min_distinct_ground_truth:
        reasons.append(
            "distinct finite ground-truth count "
            f"{n_distinct_ground_truth} is below minimum "
            f"{min_distinct_ground_truth}"
        )
    return reasons


def retention_baseline_warnings(
    *,
    feature_metrics: list[dict[str, Any]],
    baseline_signal_detected: bool,
    control_warnings: list[str],
    min_baseline_abs_rho: float,
) -> list[str]:
    warnings = list(control_warnings)
    if not feature_metrics:
        warnings.append("no cheap metadata feature had enough coverage/variance")
    elif not baseline_signal_detected:
        warnings.append(
            "no cheap metadata baseline reached "
            f"|rho| >= {min_baseline_abs_rho:.4f}; this is not a label failure, "
            "but TRIBE results should be interpreted against a weak cheap baseline"
        )
    return warnings


def next_actions(
    *,
    ready_for_modal_slice: bool,
    baseline_signal_detected: bool,
    blocking_reasons: list[str],
    warnings: list[str],
) -> list[str]:
    actions: list[str] = []
    if blocking_reasons:
        actions.append(
            "Fix retention-label mechanics before running Modal TRIBE extraction."
        )
    if any("negative control" in warning for warning in warnings):
        actions.append(
            "Inspect row ordering, sample-id construction, and label leakage before "
            "using these labels for claims."
        )
    if ready_for_modal_slice:
        actions.append(
            "Run a small full-mode Modal TRIBE slice with the same audited labels."
        )
    if ready_for_modal_slice and not baseline_signal_detected:
        actions.append(
            "Record the cheap-baseline null result as a control when interpreting "
            "TRIBE correlations."
        )
    return dedupe(actions)


def render_retention_baseline_markdown(report: dict[str, Any]) -> str:
    best = report["best_feature"] or {}
    lines = [
        "# Retention Baseline Audit",
        "",
        "## Verdict",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Labels CSV: `{report['labels_csv']}`",
        f"- Ready for Modal slice: {report['ready_for_modal_slice']}",
        f"- Baseline signal detected: {report['baseline_signal_detected']}",
        f"- Feature metrics tested: {report['n_feature_metrics']}",
        f"- Best cheap feature: `{best.get('feature', 'n/a')}`",
        f"- Best |rho|: {format_optional_float(best.get('abs_rho'))}",
        f"- Negative-control max |rho|: {format_optional_float(max_control_abs_rho(report))}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reasons"]
    lines.extend(f"- {reason}" for reason in blockers) if blockers else lines.append(
        "- none"
    )
    lines.extend(["", "## Warnings", ""])
    warnings = report["warnings"]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append(
        "- none"
    )
    lines.extend(["", "## Next Actions", ""])
    actions = report["next_actions"]
    lines.extend(f"- {action}" for action in actions) if actions else lines.append(
        "- none"
    )
    lines.extend(
        [
            "",
            "## Top Cheap Features",
            "",
            "| feature | kind | n | rho | |rho| | distinct |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    feature_metrics = report["feature_metrics"][:10]
    if not feature_metrics:
        lines.append("| none | n/a | 0 | n/a | n/a | 0 |")
    for metric in feature_metrics:
        lines.append(
            f"| {table_cell(metric['feature'])} | {metric['kind']} | "
            f"{metric['n']} | {metric['rho']:.4f} | {metric['abs_rho']:.4f} | "
            f"{metric['feature_n_distinct']} |"
        )
    lines.extend(
        [
            "",
            "## Negative Controls",
            "",
            "| control | n | rho | |rho| |",
            "|---|---:|---:|---:|",
        ]
    )
    controls = report["negative_controls"]
    if not controls:
        lines.append("| none | 0 | n/a | n/a |")
    for control in controls:
        lines.append(
            f"| {table_cell(control['feature'])} | {control['n']} | "
            f"{control['rho']:.4f} | {control['abs_rho']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def choose_column(header: list[str], hints: tuple[str, ...]) -> str | None:
    matches = matching_columns(header, hints)
    return matches[0] if matches else None


def matching_columns(header: list[str], hints: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for column in header:
        normalized = normalize(column)
        if normalized in hints or any(hint in normalized for hint in hints):
            matches.append(column)
    return matches


def cell(row: dict[str, str], column: str | None) -> str | None:
    if column is None:
        return None
    value = row.get(column)
    if value is None:
        return None
    value = value.strip()
    return value or None


def finite_float(value: object) -> float | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def label_csv_relation(
    audit_labels_csv: object,
    labels_csv: Path,
    *,
    sample_id_column: str | None,
) -> str | None:
    if not isinstance(audit_labels_csv, str) or not audit_labels_csv:
        return None
    if sample_id_column is None:
        return "mismatch"
    audit_path = Path(audit_labels_csv).expanduser().resolve()
    labels_path = labels_csv.expanduser().resolve()
    if audit_path == labels_path:
        return "same"
    if not audit_path.exists() or not labels_path.exists():
        return "mismatch"

    audit_rows = read_csv_rows(audit_path)[0]
    subset_rows = read_csv_rows(labels_path)[0]
    by_sample_id = {
        row.get(sample_id_column): row for row in audit_rows if row.get(sample_id_column)
    }
    for row in subset_rows:
        sample_id = row.get(sample_id_column)
        audit_row = by_sample_id.get(sample_id)
        if audit_row is None:
            return "mismatch"
        for key, value in row.items():
            if audit_row.get(key) != value:
                return "mismatch"
    return "subset"


def stable_uniform(value: str) -> float:
    digest = sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:13], 16) / float(16**13)


def max_control_abs_rho(report: dict[str, Any]) -> float | None:
    values = [float(control["abs_rho"]) for control in report["negative_controls"]]
    return max(values) if values else None


def format_optional_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


def table_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


if __name__ == "__main__":
    main()
