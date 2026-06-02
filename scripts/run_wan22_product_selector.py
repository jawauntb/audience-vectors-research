"""Run the Wan2.2 audience-vector product selector on generated clips.

This is the one-command glue for the product-style workflow after clips have
already been generated:

1. Score base/single and best-of-N clips with the TRIBE/BMD direction when
   reports are missing and --score-missing is set.
2. Score best-of-N candidates with CLIP preservation guardrails.
3. Summarize the selector policy: base vs single vs raw best-of-N vs gated
   best-of-N, with optional QC contact sheets.

It intentionally does not launch Wan generation. Generation is still expensive
and model-specific; this script is the deterministic selector/re-scoring layer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelectorPreset:
    single_generated_dir: Path
    bon_generated_dir: Path
    seed_root: Path
    single_report: Path
    bon_report: Path
    composite_report: Path
    out_json: Path
    out_md: Path
    qc_prefix: Path
    feature_prefix: str


PRESETS: dict[str, SelectorPreset] = {
    "pref-weighted-r16-s300": SelectorPreset(
        single_generated_dir=Path(
            "data/generated/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12/"
            "wan22_tribe_proxy_pref_weighted_r16_s300_wan22_lora_eval_fresh_picsum_24_eval_24x2_s12_m1p0"
        ),
        bon_generated_dir=Path(
            "data/generated/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0/"
            "wan22_tribe_proxy_pref_weighted_r16_s300_wan22_lora_eval_fresh_picsum_24_bon_24x4_s12_m1p0"
        ),
        seed_root=Path("data/training/wan22_lora_eval_fresh_picsum_24"),
        single_report=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12_results.json"
        ),
        bon_report=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_results.json"
        ),
        composite_report=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_composite_gate008.json"
        ),
        out_json=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_product_selector.json"
        ),
        out_md=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_product_selector.md"
        ),
        qc_prefix=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_qc"
        ),
        feature_prefix="tribe_wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300",
    ),
    "legacy-r16-s150": SelectorPreset(
        single_generated_dir=Path(
            "data/generated/wan22_lora_eval_fresh_picsum_24_r16_s150_s12/"
            "wan22_tribe_proxy_i2v_low_r16_s150_wan22_lora_eval_fresh_picsum_24_eval_24x2_s12_m1p0"
        ),
        bon_generated_dir=Path(
            "data/generated/wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0/"
            "wan22_tribe_proxy_i2v_low_r16_s150_wan22_lora_eval_fresh_picsum_24_bon_24x4_s12_m1p0"
        ),
        seed_root=Path("data/training/wan22_lora_eval_fresh_picsum_24"),
        single_report=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_r16_s150_s12_results.json"
        ),
        bon_report=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0_results.json"
        ),
        composite_report=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0_composite_gate008.json"
        ),
        out_json=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0_product_selector.json"
        ),
        out_md=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0_product_selector.md"
        ),
        qc_prefix=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0_qc"
        ),
        feature_prefix="tribe_wan22_lora_eval_fresh_picsum_24_r16_s150",
    ),
}


def choose_path(override: Path | None, preset_path: Path) -> Path:
    return override if override is not None else preset_path


def run_command(args: list[str], *, dry_run: bool) -> None:
    print("[selector-run] " + " ".join(args), flush=True)
    if dry_run:
        return
    subprocess.run(args, check=True)


def score_report_if_missing(
    *,
    report: Path,
    generated_dir: Path,
    feature_dir: Path,
    score_missing: bool,
    skip_upload: bool,
    concurrency: int,
    dry_run: bool,
) -> None:
    if report.exists():
        print(f"[selector-run] using existing report: {report}", flush=True)
        return
    if not score_missing:
        raise FileNotFoundError(
            f"{report} is missing. Pass --score-missing to run TRIBE scoring."
        )
    cmd = [
        sys.executable,
        "scripts/eval_wan22_best_of_n.py",
        "--generated-dir",
        str(generated_dir),
        "--feature-dir",
        str(feature_dir),
        "--report-path",
        str(report),
        "--concurrency",
        str(concurrency),
    ]
    if skip_upload:
        cmd.append("--skip-upload")
    run_command(cmd, dry_run=dry_run)


def run_composite(
    *,
    generated_dir: Path,
    tribe_report: Path,
    seed_root: Path,
    composite_report: Path,
    composite_md: Path,
    image_weight: float,
    prompt_weight: float,
    max_seed_drop: float | None,
    max_prompt_drop: float | None,
    force: bool,
    dry_run: bool,
) -> None:
    if composite_report.exists() and not force:
        print(f"[selector-run] using existing composite: {composite_report}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/score_wan22_composite_preservation.py",
        "--generated-dir",
        str(generated_dir),
        "--tribe-report",
        str(tribe_report),
        "--seed-root",
        str(seed_root),
        "--out-json",
        str(composite_report),
        "--out-md",
        str(composite_md),
        "--image-weight",
        str(image_weight),
        "--prompt-weight",
        str(prompt_weight),
    ]
    if max_seed_drop is not None:
        cmd.extend(["--max-seed-cosine-drop-from-best", str(max_seed_drop)])
    if max_prompt_drop is not None:
        cmd.extend(["--max-prompt-cosine-drop-from-best", str(max_prompt_drop)])
    run_command(cmd, dry_run=dry_run)


def run_summary(
    *,
    single_report: Path,
    bon_report: Path,
    composite_report: Path,
    out_json: Path,
    out_md: Path,
    single_generated_dir: Path,
    bon_generated_dir: Path,
    qc_prefix: Path | None,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/summarize_wan22_product_selector.py",
        "--single-report",
        str(single_report),
        "--bon-report",
        str(bon_report),
        "--composite-report",
        str(composite_report),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    if qc_prefix is not None:
        cmd.extend(
            [
                "--single-generated-dir",
                str(single_generated_dir),
                "--bon-generated-dir",
                str(bon_generated_dir),
                "--qc-prefix",
                str(qc_prefix),
            ]
        )
    run_command(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="pref-weighted-r16-s300")
    parser.add_argument("--single-generated-dir", type=Path)
    parser.add_argument("--bon-generated-dir", type=Path)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--single-report", type=Path)
    parser.add_argument("--bon-report", type=Path)
    parser.add_argument("--composite-report", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--qc-prefix", type=Path)
    parser.add_argument("--no-qc", action="store_true")
    parser.add_argument("--score-missing", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--force-composite", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--image-weight", type=float, default=0.75)
    parser.add_argument("--prompt-weight", type=float, default=0.25)
    parser.add_argument("--max-seed-drop", type=float, default=0.08)
    parser.add_argument("--max-prompt-drop", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    single_generated_dir = choose_path(
        args.single_generated_dir, preset.single_generated_dir
    )
    bon_generated_dir = choose_path(args.bon_generated_dir, preset.bon_generated_dir)
    seed_root = choose_path(args.seed_root, preset.seed_root)
    single_report = choose_path(args.single_report, preset.single_report)
    bon_report = choose_path(args.bon_report, preset.bon_report)
    composite_report = choose_path(args.composite_report, preset.composite_report)
    out_json = choose_path(args.out_json, preset.out_json)
    out_md = choose_path(args.out_md, preset.out_md)
    qc_prefix = None if args.no_qc else choose_path(args.qc_prefix, preset.qc_prefix)
    composite_md = composite_report.with_suffix(".md")

    score_report_if_missing(
        report=single_report,
        generated_dir=single_generated_dir,
        feature_dir=Path(f"data/features/{preset.feature_prefix}_single_selector"),
        score_missing=args.score_missing,
        skip_upload=args.skip_upload,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )
    score_report_if_missing(
        report=bon_report,
        generated_dir=bon_generated_dir,
        feature_dir=Path(f"data/features/{preset.feature_prefix}_bon_selector"),
        score_missing=args.score_missing,
        skip_upload=args.skip_upload,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )
    run_composite(
        generated_dir=bon_generated_dir,
        tribe_report=bon_report,
        seed_root=seed_root,
        composite_report=composite_report,
        composite_md=composite_md,
        image_weight=args.image_weight,
        prompt_weight=args.prompt_weight,
        max_seed_drop=args.max_seed_drop,
        max_prompt_drop=args.max_prompt_drop,
        force=args.force_composite,
        dry_run=args.dry_run,
    )
    run_summary(
        single_report=single_report,
        bon_report=bon_report,
        composite_report=composite_report,
        out_json=out_json,
        out_md=out_md,
        single_generated_dir=single_generated_dir,
        bon_generated_dir=bon_generated_dir,
        qc_prefix=qc_prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
