"""Persona-conditioned segment scoring via Claude Haiku 4.5.

Re-scores each segment from each persona's perspective using the Gemini-produced
text description as input. Pure text → text, much cheaper and faster than video.
Prompt caching on the persona system message gives ~90% cost reduction for
the persona × N segments grid.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import polars as pl
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

AXES = [
    "attention", "memorability", "confusion", "emotional_intensity",
    "semantic_surprise", "narrative_progress", "social_salience",
    "audio_salience", "visual_salience", "rewatch_likelihood",
]

SCHEMA = {
    "type": "object",
    "properties": {
        **{axis: {"type": "number"} for axis in AXES},
        "reason": {"type": "string"},
    },
    "required": [*AXES, "reason"],
    "additionalProperties": False,
}


def _persona_system(persona: dict) -> str:
    aw = persona["attention_weights"]
    dl = persona["dislikes"]
    aw_text = "\n".join(f"  - {k}: {v}" for k, v in aw.items())
    dl_text = "\n".join(f"  - {k}: {v}" for k, v in dl.items())
    return (
        f"You are roleplaying as a viewer persona: {persona['persona_id']} "
        f"(cluster: {persona['cluster']}).\n\n"
        f"Story: {persona['story']}\n\n"
        f"Your attention weights (0-1, what catches your eye):\n{aw_text}\n\n"
        f"Your dislikes (0-1, what turns you off):\n{dl_text}\n\n"
        "Given a short video clip description, score it from THIS persona's "
        "perspective on 10 axes (attention, memorability, confusion, "
        "emotional_intensity, semantic_surprise, narrative_progress, "
        "social_salience, audio_salience, visual_salience, rewatch_likelihood). "
        "Each score is 0-1. Also provide a one-sentence reason. "
        "Return JSON only."
    )


async def _score_one(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    model: str,
    system: str,
    segment_id: str,
    segment_reason: str,
) -> dict | None:
    async with sem:
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=512,
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Video segment description:\n{segment_reason}\n\n"
                        "Score this segment from your persona's perspective."
                    ),
                }],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("haiku call failed for %s: %s", segment_id, exc)
            return None

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        return None
    return {
        "segment_id": segment_id,
        "scores": scores,
        "usage": {
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0),
            "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0),
        },
    }


async def _score_persona(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    model: str,
    persona: dict,
    segments: list[dict],
) -> list[dict]:
    system = _persona_system(persona)
    coros = [
        _score_one(client, sem, model, system, s["segment_id"], s["reason"])
        for s in segments
    ]
    results = await asyncio.gather(*coros)
    rows = []
    for r in results:
        if r is None:
            continue
        rows.append({
            "segment_id": r["segment_id"],
            "persona_id": persona["persona_id"],
            "scores": r["scores"],
            "reason": r["scores"].get("reason", ""),
            "source": "synthetic_haiku",
            "prompt_version": "haiku_persona_v1",
            "model_id": model,
        })
    return rows


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = args.model or os.environ.get("ANTHROPIC_MODEL_BULK", "claude-haiku-4-5")

    personas_df = pl.read_parquet("data/labels/personas.parquet")
    personas = personas_df.to_dicts()
    if args.max_personas:
        personas = personas[: args.max_personas]

    seg_df = pl.read_parquet("data/labels/synthetic_gemini.parquet")
    seg_df = seg_df.unique(subset=["segment_id"]).filter(pl.col("reason").is_not_null())
    segments = seg_df.select(["segment_id", "reason"]).to_dicts()
    if args.max_segments:
        segments = segments[: args.max_segments]

    print(f"[haiku] personas={len(personas)} segments={len(segments)} "
          f"calls={len(personas) * len(segments)} concurrency={args.concurrency}")

    client = AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    all_rows: list[dict] = []
    for i, persona in enumerate(personas):
        rows = await _score_persona(client, sem, model, persona, segments)
        all_rows.extend(rows)
        print(f"[haiku] persona {i+1}/{len(personas)} ({persona['persona_id']}): "
              f"{len(rows)}/{len(segments)} succeeded")
        # incremental save so we don't lose progress on crash
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(all_rows).write_parquet(out)

    print(f"[done] wrote {len(all_rows)} rows to {args.output}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="data/labels/synthetic_persona_haiku.parquet")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-personas", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
