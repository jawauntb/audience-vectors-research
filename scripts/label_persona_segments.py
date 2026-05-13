"""Score every segment from every persona's perspective.

Loops persona × segment. Uses the same `GeminiLabeler` with a
persona-conditioned prompt. Output goes to a separate parquet so
non-persona labels stay clean.

Usage:
    uv run python scripts/label_persona_segments.py
    uv run python scripts/label_persona_segments.py --max-segments 5 --max-personas 4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from audience_vectors.config import get_config
from audience_vectors.labeling import GeminiLabeler
from audience_vectors.schemas import Persona, Segment


def _read_segments(p: Path) -> list[Segment]:
    import polars as pl  # noqa: PLC0415

    return [Segment.model_validate(r) for r in pl.read_parquet(p).to_dicts()]


def _read_personas(p: Path) -> list[Persona]:
    import polars as pl  # noqa: PLC0415

    return [Persona.model_validate(r) for r in pl.read_parquet(p).to_dicts()]


def _existing_keys(p: Path) -> set[tuple[str, str]]:
    if not p.exists():
        return set()
    import polars as pl  # noqa: PLC0415

    df = pl.read_parquet(p)
    if "segment_id" not in df.columns or "persona_id" not in df.columns:
        return set()
    return set(zip(df["segment_id"].to_list(), df["persona_id"].to_list()))


async def _run(args: argparse.Namespace) -> int:
    cfg = get_config()
    cfg.paths.ensure()
    segments_path = args.segments or (cfg.paths.training / "segments.parquet")
    personas_path = args.personas or (cfg.paths.labels / "personas.parquet")
    output_path = args.output or (cfg.paths.labels / "synthetic_persona_gemini.parquet")

    if not segments_path.exists():
        print(f"[fail] missing {segments_path}")
        return 1
    if not personas_path.exists():
        print(f"[fail] missing {personas_path} — run scripts/generate_personas.py")
        return 1
    if not cfg.api_keys.google:
        print("[fail] GOOGLE_API_KEY missing")
        return 1

    segments = _read_segments(segments_path)
    personas = _read_personas(personas_path)
    if args.max_segments:
        segments = segments[: args.max_segments]
    if args.max_personas:
        personas = personas[: args.max_personas]

    already = _existing_keys(output_path)
    pairs = [
        (p, s) for p in personas for s in segments
        if (s.sample_id, p.persona_id) not in already
    ]
    print(
        f"[plan] personas={len(personas)} segments={len(segments)} "
        f"pairs_pending={len(pairs)} already={len(already)} "
        f"model={cfg.models.gemini_video}"
    )
    if not pairs:
        print("[done] nothing to do.")
        return 0

    labeler = GeminiLabeler(
        api_key=cfg.api_keys.google,
        model=cfg.models.gemini_video,
        max_concurrency=args.concurrency,
    )

    # Fire all pairs concurrently; the semaphore inside the labeler caps
    # in-flight requests at `max_concurrency`.
    coros = [labeler.label_segment(s, persona=p) for p, s in pairs]
    raw = await asyncio.gather(*coros)
    results = [r for r in raw if r is not None]
    print(f"[result] labeled={len(results)} / pairs={len(pairs)}")

    if not results:
        return 1

    import polars as pl  # noqa: PLC0415

    new_rows = [r.model_dump(mode="json") for r in results]
    new_df = pl.DataFrame(new_rows)
    if output_path.exists():
        old_df = pl.read_parquet(output_path)
        combined = pl.concat([old_df, new_df], how="vertical_relaxed")
    else:
        combined = new_df
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)
    print(f"[done] wrote {len(new_rows)} new labels -> {output_path} (total: {len(combined)})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--personas", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--max-personas", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
