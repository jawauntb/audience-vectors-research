#!/usr/bin/env python3
"""F1: Replicate best-of-N baseline (Condition C1) from Brown §6.7.

Scores the pre-generated SVD variants in audience_vectors_share using
the TRIBE scorer and reports lift statistics.

Can run immediately with mock TRIBE (--mock) to verify the pipeline,
or with the real TRIBE scorer once v_mem.npy is available.

Usage:
    # With mock scorer (no GPU needed — for pipeline validation):
    uv run python scripts/run_baseline.py --mock

    # With real TRIBE scorer:
    uv run python scripts/run_baseline.py
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path(__file__).parent.parent
        / "../audience_vectors_share/generated/svd_best_of_n",
    )
    parser.add_argument("--prompt", type=str, default="A scene from a naturalistic video clip.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mock", action="store_true", help="Use mock TRIBE (no GPU)")
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_c1.json"))
    args = parser.parse_args()

    gen_dir = args.generated_dir.resolve()
    if not gen_dir.exists():
        console.print(f"[red]Directory not found: {gen_dir}")
        raise SystemExit(1)

    from bo_mem.baselines.best_of_n import BestOfNBaseline
    from bo_mem.config import Config
    from bo_mem.objectives.clip_scorer import CLIPScorer
    from bo_mem.objectives.tribe_scorer import MockTribeBackend, TribeScorer

    cfg = Config()

    if args.mock:
        console.print("[yellow]Using mock TRIBE backend (random scores)")
        tribe = TribeScorer(backend=MockTribeBackend(seed=42))
    else:
        tribe = TribeScorer(vmem_path=cfg.paths.vmem)
        if not tribe.vmem_loaded:
            console.print(f"[red]v_mem not found at {cfg.paths.vmem}. Run with --mock or provide the file.")
            raise SystemExit(1)

    clip = CLIPScorer(device=cfg.device)
    baseline = BestOfNBaseline(tribe_scorer=tribe, clip_scorer=clip, prompt=args.prompt)

    console.rule("[bold]C1 — Best-of-N Baseline")
    results = baseline.run_all_seeds(gen_dir, n_variants=args.n)
    summary = baseline.summarize(results)

    table = Table(title="Best-of-N Results per Seed")
    table.add_column("Seed", style="cyan")
    table.add_column("N", style="white")
    table.add_column("Winner mem", style="green")
    table.add_column("Median mem", style="yellow")
    table.add_column("Lift", style="magenta")

    for r in results:
        table.add_row(
            r.seed_id,
            str(r.n),
            f"{r.winner_mem_score:.3f}",
            f"{r.median_mem_score:.3f}",
            f"{r.lift:+.3f}",
        )

    console.print(table)
    console.print(f"\n[bold]Mean lift: {summary['mean_lift']:+.3f} ± {summary['std_lift']:.3f}")
    console.print(f"(Paper target: +2.07 ± 0.60 for SVD-XT)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_seed": [
            {"seed": r.seed_id, "lift": r.lift, "winner_mem": r.winner_mem_score,
             "median_mem": r.median_mem_score, "n": r.n}
            for r in results
        ]}, f, indent=2)
    console.print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
