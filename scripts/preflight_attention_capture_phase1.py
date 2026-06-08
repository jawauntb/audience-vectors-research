"""Preflight a Phase 1 attention-capture manifest before scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audience_vectors.attention_capture import (
    preflight_phase1_manifest,
    render_preflight_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--roi-masks", type=Path, default=None)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-distinct-ground-truth", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = preflight_phase1_manifest(
        args.manifest,
        roi_masks_path=args.roi_masks,
        min_samples=args.min_samples,
        min_distinct_ground_truth=args.min_distinct_ground_truth,
        epsilon=args.epsilon,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_preflight_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["mechanical_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
