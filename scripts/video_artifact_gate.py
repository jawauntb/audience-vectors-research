#!/usr/bin/env python
"""Flag generated videos whose tail frames collapse visually."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audience_vectors.visual_artifact_gate import (
    ArtifactThresholds,
    summarize_video_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--fail-on-artifacts", action="store_true")
    parser.add_argument("--min-tail-sharpness-ratio", type=float, default=0.35)
    parser.add_argument("--min-tail-contrast-ratio", type=float, default=0.55)
    parser.add_argument("--min-tail-contrast", type=float, default=0.04)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = ArtifactThresholds(
        min_tail_sharpness_ratio=args.min_tail_sharpness_ratio,
        min_tail_contrast_ratio=args.min_tail_contrast_ratio,
        min_tail_contrast=args.min_tail_contrast,
    )
    report = summarize_video_dir(
        args.video_dir,
        pattern=args.pattern,
        samples=args.samples,
        thresholds=thresholds,
    )
    payload = json.dumps(report, indent=2)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(payload, encoding="utf-8")
        print(f"[visual-gate] wrote report to {args.report_path}")
    else:
        print(payload)

    print(
        "[visual-gate] "
        f"{report['n_failed']}/{report['n_videos']} videos failed visual gate"
    )
    if args.fail_on_artifacts and not report["passes_visual_gate"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
