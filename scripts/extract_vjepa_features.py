"""Run V-JEPA 2 over segments, persist mean-pooled embeddings.

Prereq:
    modal deploy -m audience_vectors.modal_app.app
    modal run -m audience_vectors.modal_app.functions.vjepa_predictor::populate_vjepa_weights

Usage:
    uv run python scripts/extract_vjepa_features.py --limit 20

The Modal container can't see local file paths, so for BMD segments we
substitute the segment's MiT_url (from BMD annotations) before dispatch.
For segments from other datasets, pass `--remote-url-map` pointing at a
JSON file mapping `video_id -> public_url`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from audience_vectors.config import get_config
from audience_vectors.features import VjepaFeatureExtractor
from audience_vectors.schemas import Segment


def _load_bmd_url_map() -> dict[str, str]:
    bmd_path = Path("./data/raw/bold_moments/annotations.json")
    if not bmd_path.exists():
        return {}
    with bmd_path.open() as fh:
        ann = json.load(fh)
    return {
        f"bmd_vid_idx{eid}": entry.get("MiT_url", "")
        for eid, entry in ann.items()
        if entry.get("MiT_url")
    }


def _substitute_paths(
    segments: list[Segment],
    *,
    local_dir: Path,
    volume_mount: str,
    url_map: dict[str, str],
) -> list[Segment]:
    """Prefer Modal-volume paths (`/bmd-videos/<file>.mp4`) when the local
    file exists; fall back to MiT URL otherwise. Modal containers can't see
    the laptop's filesystem, so we map every local path to the volume mount
    they CAN see.
    """
    out: list[Segment] = []
    n_vol = n_url = n_skip = 0
    for s in segments:
        # Filename pattern from the BMD adapter: bmd_<matrix_name>
        # e.g. video_id="bmd_vid_idx0042" → file="vid_idx0042.mp4"
        matrix_name = s.video_id.removeprefix("bmd_")
        local_file = local_dir / f"{matrix_name}.mp4"
        if local_file.exists():
            # Modal volume layout: uploaded dir 'videos/' is at volume root
            new_path = f"{volume_mount}/videos/{matrix_name}.mp4"
            s = s.model_copy(update={"media_path": new_path})
            n_vol += 1
        else:
            url = url_map.get(s.video_id)
            if url:
                s = s.model_copy(update={"media_path": url})
                n_url += 1
            else:
                n_skip += 1
        out.append(s)
    print(f"[paths] volume={n_vol}  url_fallback={n_url}  no_source={n_skip}")
    return out


async def _run(args: argparse.Namespace) -> int:
    cfg = get_config()
    cfg.paths.ensure()
    segments_path = args.segments or (cfg.paths.training / "segments.parquet")
    if not segments_path.exists():
        print(f"[fail] missing {segments_path}")
        return 1
    output_dir = args.output or (cfg.paths.features / "vjepa")
    output_dir.mkdir(parents=True, exist_ok=True)

    import polars as pl  # noqa: PLC0415

    segments = [Segment.model_validate(r) for r in pl.read_parquet(segments_path).to_dicts()]
    if args.limit:
        segments = segments[: args.limit]

    # Prefer Modal-volume paths for locally-downloaded videos, fall back to
    # MiT URL for clips we couldn't grab. Modal containers can't see laptop
    # paths directly, so we route through the `bmd-videos-v1` volume.
    url_map = _load_bmd_url_map()
    if args.remote_url_map:
        with open(args.remote_url_map) as fh:
            url_map.update(json.load(fh))
    local_dir = Path("./data/raw/bold_moments/videos")
    segments = _substitute_paths(
        segments,
        local_dir=local_dir,
        volume_mount="/bmd-videos",
        url_map=url_map,
    )

    extractor = VjepaFeatureExtractor(
        output_dir=output_dir, max_concurrency=args.concurrency,
    )
    print(f"[plan] segments={len(segments)} output={output_dir} concurrency={args.concurrency}")
    written = await extractor.extract_many(segments)
    print(f"[done] features written / cached for {len(written)}/{len(segments)} segments")
    if written:
        print(f"       sample: {written[0]}")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--remote-url-map", type=Path, default=None,
        help="JSON file mapping video_id -> public URL (for non-BMD datasets).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
