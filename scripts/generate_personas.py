"""Generate synthetic viewer archetypes via Anthropic.

Writes `data/labels/personas.parquet`. Idempotent — re-running overwrites
unless you pass `--append`.

Usage:
    uv run python scripts/generate_personas.py --n 20
    uv run python scripts/generate_personas.py --n 50 --append
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audience_vectors.config import get_config
from audience_vectors.labeling import DEFAULT_CLUSTERS, PersonaGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--append", action="store_true", help="Append instead of overwrite.")
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=None,
        help="Override the cluster list (default: 10 starter archetypes).",
    )
    args = parser.parse_args()

    cfg = get_config()
    cfg.paths.ensure()

    if not cfg.api_keys.anthropic:
        print("[fail] ANTHROPIC_API_KEY missing in .env")
        sys.exit(1)

    output_path = args.output or (cfg.paths.labels / "personas.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clusters = tuple(args.clusters) if args.clusters else DEFAULT_CLUSTERS
    print(f"[plan] generating n={args.n} personas across {len(clusters)} clusters "
          f"with {cfg.models.anthropic_judge}")
    gen = PersonaGenerator(
        api_key=cfg.api_keys.anthropic,
        model=cfg.models.anthropic_judge,
    )
    personas = gen.generate(n=args.n, clusters=clusters)
    print(f"[result] generated {len(personas)} personas")

    import polars as pl  # noqa: PLC0415

    rows = [p.model_dump(mode="json") for p in personas]
    new_df = pl.DataFrame(rows)
    if args.append and output_path.exists():
        old_df = pl.read_parquet(output_path)
        new_df = pl.concat([old_df, new_df], how="vertical_relaxed")
    new_df.write_parquet(output_path)
    print(f"[done] wrote {len(new_df)} personas -> {output_path}")

    sample = personas[0]
    print(f"\n--- sample persona ---")
    print(f"id:       {sample.persona_id}")
    print(f"cluster:  {sample.cluster}")
    print(f"story:    {sample.story}")
    top_attn = sorted(sample.attention_weights.items(), key=lambda kv: -kv[1])[:3]
    top_dis = sorted(sample.dislikes.items(), key=lambda kv: -kv[1])[:3]
    print(f"top attn: {', '.join(f'{k}={v:.2f}' for k,v in top_attn)}")
    print(f"top dis:  {', '.join(f'{k}={v:.2f}' for k,v in top_dis)}")


if __name__ == "__main__":
    main()
