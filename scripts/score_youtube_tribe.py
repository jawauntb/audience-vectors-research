"""Score a YouTube video with the TRIBE memorability direction.

The script keeps the full trail:

1. download/cache YouTube metadata and an MP4 source,
2. cut the source into <=30s TRIBE-compatible windows,
3. upload windows to the Modal `bmd-videos-v1` volume,
4. run TRIBE, and
5. project each segment onto the BMD-trained memorability direction.

Outputs:
    data/external/youtube/<video-id>/tribe_score_report.json
    data/external/youtube/<video-id>/tribe_score_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from audience_vectors.features.tribe_extractor import TribeFeatureExtractor
from audience_vectors.schemas import Segment

LOGGER = logging.getLogger(__name__)
TRIBE_MAX_SECONDS = 30.0
DEFAULT_VIDEO_ID = "nw-2sPa7DAg"


@dataclass(frozen=True)
class SegmentWindow:
    sample_id: str
    local_path: Path
    modal_path: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class ReferenceDirection:
    axis: str
    label_source: str
    direction: np.ndarray
    projections: np.ndarray
    n_segments: int
    n_videos: int
    pos_set_size: int
    neg_set_size: int
    mean: float
    std: float


def run_command(cmd: list[str]) -> None:
    LOGGER.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffprobe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def ensure_youtube_assets(url: str, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"
    source_path = out_dir / "source.mp4"

    if not metadata_path.exists():
        result = subprocess.run(
            ["uvx", "yt-dlp", "--dump-json", "--no-playlist", url],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata_path.write_text(result.stdout)

    if not source_path.exists():
        run_command(
            [
                "uvx",
                "yt-dlp",
                "--no-playlist",
                "-f",
                "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
                "--merge-output-format",
                "mp4",
                "-o",
                str(out_dir / "source.%(ext)s"),
                url,
            ]
        )

    metadata = json.loads(metadata_path.read_text())
    return source_path, metadata


def segment_video(
    *,
    video_path: Path,
    video_id: str,
    out_dir: Path,
    segment_seconds: float,
    overwrite: bool,
) -> list[SegmentWindow]:
    if segment_seconds <= 0 or segment_seconds > TRIBE_MAX_SECONDS:
        raise ValueError(
            f"segment_seconds must be in (0, {TRIBE_MAX_SECONDS}], "
            f"got {segment_seconds}"
        )

    segment_dir = out_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    duration = ffprobe_duration(video_path)
    n_segments = int(math.ceil(duration / segment_seconds))
    windows: list[SegmentWindow] = []

    for idx in range(n_segments):
        start = idx * segment_seconds
        end = min(duration, start + segment_seconds)
        if end - start < 0.5:
            continue
        sample_id = f"youtube_{video_id}_seg_{idx:04d}"
        local_path = segment_dir / f"{sample_id}.mp4"
        if overwrite or not local_path.exists() or local_path.stat().st_size == 0:
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{end - start:.3f}",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    str(local_path),
                ]
            )
        remote_path = f"/youtube/{video_id}/{local_path.name}"
        windows.append(
            SegmentWindow(
                sample_id=sample_id,
                local_path=local_path,
                modal_path=f"/bmd-videos{remote_path}",
                start_s=start,
                end_s=end,
            )
        )
    return windows


def upload_segments(windows: list[SegmentWindow], *, force_upload: bool) -> None:
    import modal  # noqa: PLC0415

    volume = modal.Volume.from_name("bmd-videos-v1", create_if_missing=True)
    with volume.batch_upload(force=force_upload) as batch:
        for window in windows:
            remote_path = window.modal_path.removeprefix("/bmd-videos")
            batch.put_file(window.local_path, remote_path)


async def extract_tribe_features(
    *,
    windows: list[SegmentWindow],
    features_dir: Path,
    concurrency: int,
) -> list[Path]:
    extractor = TribeFeatureExtractor(output_dir=features_dir, max_concurrency=concurrency)
    segments = [
        Segment(
            sample_id=window.sample_id,
            video_id=window.sample_id,
            source_dataset="youtube",
            start_time=window.start_s,
            end_time=window.end_s,
            duration=window.duration_s,
            media_path=window.modal_path,
        )
        for window in windows
    ]
    return await extractor.extract_many(segments)


def load_memorability_scores(path: Path) -> dict[str, float]:
    annotations = json.loads(path.read_text())
    return {
        f"bmd_vid_idx{entry_id}": float(entry["memorability_score"])
        for entry_id, entry in annotations.items()
        if "memorability_score" in entry
    }


def load_feature_vector(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "frames" in payload.files:
        frames = np.asarray(payload["frames"], dtype=np.float32)
        return frames.mean(axis=0) if frames.ndim == 2 else frames
    if "embedding" in payload.files:
        return np.asarray(payload["embedding"], dtype=np.float32)
    raise ValueError(f"{path} has neither frames nor embedding")


def train_reference_direction(
    *,
    features_dir: Path,
    annotations_path: Path,
    top_frac: float,
) -> ReferenceDirection:
    scores_by_video = load_memorability_scores(annotations_path)
    rows: list[tuple[str, str, float, np.ndarray]] = []
    for path in sorted(features_dir.glob("bmd_vid_idx*.npz")):
        video_id = path.stem.split("_seg_")[0]
        score = scores_by_video.get(video_id)
        if score is None:
            continue
        rows.append((path.stem, video_id, score, load_feature_vector(path)))

    if len(rows) < 10:
        raise ValueError(f"not enough BMD TRIBE features in {features_dir}")

    y = np.asarray([row[2] for row in rows], dtype=np.float32)
    x = np.stack([row[3] for row in rows]).astype(np.float32)
    order = np.argsort(y)
    n_each = max(3, int(len(rows) * top_frac))
    neg = order[:n_each]
    pos = order[-n_each:]
    direction = x[pos].mean(axis=0) - x[neg].mean(axis=0)
    direction /= np.linalg.norm(direction) + 1e-12
    projections = x @ direction
    return ReferenceDirection(
        axis="memorability",
        label_source="BMD human memorability_score",
        direction=direction.astype(np.float32),
        projections=projections.astype(np.float32),
        n_segments=len(rows),
        n_videos=len({row[1] for row in rows}),
        pos_set_size=len(pos),
        neg_set_size=len(neg),
        mean=float(projections.mean()),
        std=float(projections.std(ddof=1)),
    )


def load_persona_axis_scores(persona_file: Path, axis: str) -> dict[str, float]:
    df = pl.read_parquet(persona_file)
    scores = df.select("scores").unnest("scores")
    if axis not in scores.columns:
        raise ValueError(f"axis {axis!r} not in persona score columns: {scores.columns}")
    rows = (
        df.with_columns(scores[axis].alias("_score"))
        .select(["segment_id", "_score"])
        .drop_nulls("_score")
        .group_by("segment_id")
        .agg(pl.mean("_score").alias("_score"))
        .to_dicts()
    )
    return {str(row["segment_id"]): float(row["_score"]) for row in rows}


def train_persona_axis_reference(
    *,
    features_dir: Path,
    persona_file: Path,
    axis: str,
    top_frac: float,
) -> ReferenceDirection:
    scores_by_segment = load_persona_axis_scores(persona_file, axis)
    rows: list[tuple[str, float, np.ndarray]] = []
    for path in sorted(features_dir.glob("bmd_vid_idx*.npz")):
        score = scores_by_segment.get(path.stem)
        if score is None:
            continue
        rows.append((path.stem, score, load_feature_vector(path)))

    if len(rows) < 10:
        raise ValueError(
            f"not enough synthetic persona {axis!r} labels with features: {len(rows)}"
        )

    y = np.asarray([row[1] for row in rows], dtype=np.float32)
    x = np.stack([row[2] for row in rows]).astype(np.float32)
    order = np.argsort(y)
    n_each = max(3, int(len(rows) * top_frac))
    neg = order[:n_each]
    pos = order[-n_each:]
    direction = x[pos].mean(axis=0) - x[neg].mean(axis=0)
    direction /= np.linalg.norm(direction) + 1e-12
    projections = x @ direction
    return ReferenceDirection(
        axis=axis,
        label_source=f"mean synthetic persona {axis}",
        direction=direction.astype(np.float32),
        projections=projections.astype(np.float32),
        n_segments=len(rows),
        n_videos=len({row[0].split("_seg_")[0] for row in rows}),
        pos_set_size=len(pos),
        neg_set_size=len(neg),
        mean=float(projections.mean()),
        std=float(projections.std(ddof=1)),
    )


def percentile_rank(reference: np.ndarray, score: float) -> float:
    sorted_ref = np.sort(reference)
    return float(np.searchsorted(sorted_ref, score, side="right") / len(sorted_ref))


def timestamp(seconds: float) -> str:
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{sec:02d}"


def build_report(
    *,
    url: str,
    metadata: dict[str, Any],
    windows: list[SegmentWindow],
    features_dir: Path,
    reference: ReferenceDirection,
    attention_reference: ReferenceDirection | None,
    segment_seconds: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        feature_path = features_dir / f"{window.sample_id}.npz"
        if not feature_path.exists():
            rows.append(
                {
                    "sample_id": window.sample_id,
                    "start_s": window.start_s,
                    "end_s": window.end_s,
                    "status": "missing_features",
                }
            )
            continue
        vec = load_feature_vector(feature_path)
        score = float(vec @ reference.direction)
        z = (score - reference.mean) / reference.std if reference.std else 0.0
        attention: dict[str, Any] = {}
        if attention_reference is not None:
            att_score = float(vec @ attention_reference.direction)
            att_z = (
                (att_score - attention_reference.mean) / attention_reference.std
                if attention_reference.std
                else 0.0
            )
            attention = {
                "synthetic_attention_projection": att_score,
                "synthetic_attention_reference_z": float(att_z),
                "synthetic_attention_reference_percentile": percentile_rank(
                    attention_reference.projections, att_score
                ),
            }
        rows.append(
            {
                "sample_id": window.sample_id,
                "start_s": window.start_s,
                "end_s": window.end_s,
                "timestamp": f"{timestamp(window.start_s)}-{timestamp(window.end_s)}",
                "duration_s": window.duration_s,
                "tribe_memorability_projection": score,
                "bmd_reference_z": float(z),
                "bmd_reference_percentile": percentile_rank(reference.projections, score),
                **attention,
                "feature_path": str(feature_path),
                "local_video_path": str(window.local_path),
                "modal_video_path": window.modal_path,
                "status": "ok",
            }
        )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    percentiles = np.asarray(
        [row["bmd_reference_percentile"] for row in ok_rows], dtype=np.float32
    )
    projections = np.asarray(
        [row["tribe_memorability_projection"] for row in ok_rows], dtype=np.float32
    )
    top_rows = sorted(
        ok_rows, key=lambda row: row["tribe_memorability_projection"], reverse=True
    )
    attention_rows = [
        row for row in ok_rows if "synthetic_attention_projection" in row
    ]
    attention_top_rows = sorted(
        attention_rows,
        key=lambda row: row["synthetic_attention_projection"],
        reverse=True,
    )
    attention_percentiles = np.asarray(
        [
            row["synthetic_attention_reference_percentile"]
            for row in attention_rows
        ],
        dtype=np.float32,
    )
    return {
        "url": url,
        "title": metadata.get("title"),
        "uploader": metadata.get("uploader"),
        "youtube_id": metadata.get("id"),
        "source_duration_s": metadata.get("duration"),
        "segment_seconds": segment_seconds,
        "n_segments": len(windows),
        "n_scored_segments": len(ok_rows),
        "reference": {
            "axis": reference.axis,
            "label_source": reference.label_source,
            "direction": "TRIBE top/bottom contrastive direction",
            "top_frac": 0.30,
            "n_bmd_segments": reference.n_segments,
            "n_bmd_videos": reference.n_videos,
            "positive_set_size": reference.pos_set_size,
            "negative_set_size": reference.neg_set_size,
            "projection_mean": reference.mean,
            "projection_std": reference.std,
        },
        "attention_reference": None
        if attention_reference is None
        else {
            "axis": attention_reference.axis,
            "label_source": attention_reference.label_source,
            "direction": "TRIBE top/bottom contrastive direction",
            "top_frac": 0.30,
            "n_bmd_segments": attention_reference.n_segments,
            "n_bmd_videos": attention_reference.n_videos,
            "positive_set_size": attention_reference.pos_set_size,
            "negative_set_size": attention_reference.neg_set_size,
            "projection_mean": attention_reference.mean,
            "projection_std": attention_reference.std,
        },
        "summary": {
            "mean_projection": float(projections.mean()) if len(projections) else None,
            "max_projection": float(projections.max()) if len(projections) else None,
            "mean_bmd_percentile": float(percentiles.mean()) if len(percentiles) else None,
            "max_bmd_percentile": float(percentiles.max()) if len(percentiles) else None,
            "mean_synthetic_attention_percentile": float(attention_percentiles.mean())
            if len(attention_percentiles)
            else None,
            "max_synthetic_attention_percentile": float(attention_percentiles.max())
            if len(attention_percentiles)
            else None,
            "top_segment": top_rows[0] if top_rows else None,
            "top_attention_segment": attention_top_rows[0]
            if attention_top_rows
            else None,
        },
        "top_segments": top_rows[:5],
        "top_attention_segments": attention_top_rows[:5],
        "segments": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    top = summary["top_segment"]
    top_attention = summary.get("top_attention_segment")
    lines = [
        f"# TRIBE YouTube Score: {report.get('title') or report['url']}",
        "",
        f"- URL: {report['url']}",
        f"- Uploader: {report.get('uploader') or 'unknown'}",
        f"- Source duration: {report.get('source_duration_s')}s",
        f"- Segments scored: {report['n_scored_segments']}/{report['n_segments']}",
        f"- Reference: {report['reference']['n_bmd_segments']} BMD segment features "
        f"from {report['reference']['n_bmd_videos']} videos",
        "",
        "## Summary",
        "",
        f"- Mean BMD-reference percentile: "
        f"{100 * summary['mean_bmd_percentile']:.1f}%"
        if summary["mean_bmd_percentile"] is not None
        else "- Mean BMD-reference percentile: n/a",
        f"- Max BMD-reference percentile: {100 * summary['max_bmd_percentile']:.1f}%"
        if summary["max_bmd_percentile"] is not None
        else "- Max BMD-reference percentile: n/a",
    ]
    if top is not None:
        lines.extend(
            [
                f"- Highest-scoring window: {top['timestamp']} "
                f"({100 * top['bmd_reference_percentile']:.1f} percentile, "
                f"z={top['bmd_reference_z']:+.2f})",
                "",
            ]
        )
    if top_attention is not None:
        lines.extend(
            [
                f"- Highest synthetic-attention window: {top_attention['timestamp']} "
                f"({100 * top_attention['synthetic_attention_reference_percentile']:.1f} "
                f"percentile, z={top_attention['synthetic_attention_reference_z']:+.2f})",
                "",
            ]
        )

    lines.extend(
        [
            "## Top Windows",
            "",
            "| Rank | Timestamp | Percentile | z | Projection |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(report["top_segments"], start=1):
        lines.append(
            f"| {idx} | {row['timestamp']} | "
            f"{100 * row['bmd_reference_percentile']:.1f}% | "
            f"{row['bmd_reference_z']:+.2f} | "
            f"{row['tribe_memorability_projection']:+.4f} |"
        )

    if report.get("top_attention_segments"):
        lines.extend(
            [
                "",
                "## Top Synthetic-Attention Windows",
                "",
                "| Rank | Timestamp | Percentile | z | Projection |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for idx, row in enumerate(report["top_attention_segments"], start=1):
            lines.append(
                f"| {idx} | {row['timestamp']} | "
                f"{100 * row['synthetic_attention_reference_percentile']:.1f}% | "
                f"{row['synthetic_attention_reference_z']:+.2f} | "
                f"{row['synthetic_attention_projection']:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## All Windows",
            "",
            "| Timestamp | Mem %ile | Mem z | Mem proj | Attention %ile | Attention z | Attention proj |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["segments"]:
        if row.get("status") != "ok":
            lines.append(
                f"| {timestamp(row['start_s'])}-{timestamp(row['end_s'])} | "
                "n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        if "synthetic_attention_projection" in row:
            attention_cells = (
                f"{100 * row['synthetic_attention_reference_percentile']:.1f}% | "
                f"{row['synthetic_attention_reference_z']:+.2f} | "
                f"{row['synthetic_attention_projection']:+.4f}"
            )
        else:
            attention_cells = "n/a | n/a | n/a"
        lines.append(
            f"| {row['timestamp']} | "
            f"{100 * row['bmd_reference_percentile']:.1f}% | "
            f"{row['bmd_reference_z']:+.2f} | "
            f"{row['tribe_memorability_projection']:+.4f} | "
            f"{attention_cells} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This is an automatic TRIBE projection score, not fresh human validation.",
            "- Percentiles are relative to the local BMD/TRIBE feature distribution used to train the paper's memorability direction.",
            "- Attention scores use mean synthetic persona attention labels, because BMD does not provide human attention labels.",
            "- The source was scored in non-overlapping windows, so peaks are approximate timestamp targets for follow-up viewing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


async def run(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("data/external/youtube") / args.video_id
    source_path, metadata = ensure_youtube_assets(args.url, out_dir)

    LOGGER.info("segmenting %s", source_path)
    windows = segment_video(
        video_path=source_path,
        video_id=args.video_id,
        out_dir=out_dir,
        segment_seconds=args.segment_seconds,
        overwrite=args.overwrite_segments,
    )
    manifest_path = out_dir / "segment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "url": args.url,
                "video_id": args.video_id,
                "segment_seconds": args.segment_seconds,
                "segments": [
                    {
                        "sample_id": window.sample_id,
                        "local_path": str(window.local_path),
                        "modal_path": window.modal_path,
                        "start_s": window.start_s,
                        "end_s": window.end_s,
                    }
                    for window in windows
                ],
            },
            indent=2,
        )
        + "\n"
    )

    if not args.skip_upload:
        LOGGER.info("uploading %d segments to Modal volume", len(windows))
        upload_segments(windows, force_upload=args.force_upload)

    features_dir = out_dir / "tribe_features"
    LOGGER.info("extracting TRIBE features")
    written = await extract_tribe_features(
        windows=windows,
        features_dir=features_dir,
        concurrency=args.concurrency,
    )
    if len(written) != len(windows):
        LOGGER.warning("extracted %d/%d TRIBE features", len(written), len(windows))

    LOGGER.info("training/loading BMD memorability reference")
    reference = train_reference_direction(
        features_dir=args.bmd_features_dir,
        annotations_path=args.bmd_annotations,
        top_frac=0.30,
    )
    attention_reference = None
    if not args.no_attention:
        LOGGER.info("training/loading synthetic persona attention reference")
        attention_reference = train_persona_axis_reference(
            features_dir=args.bmd_features_dir,
            persona_file=args.persona_file,
            axis=args.attention_axis,
            top_frac=0.30,
        )
    report = build_report(
        url=args.url,
        metadata=metadata,
        windows=windows,
        features_dir=features_dir,
        reference=reference,
        attention_reference=attention_reference,
        segment_seconds=args.segment_seconds,
    )

    json_path = out_dir / "tribe_score_report.json"
    md_path = out_dir / "tribe_score_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, md_path)
    LOGGER.info("wrote %s", json_path)
    LOGGER.info("wrote %s", md_path)
    return 0 if report["n_scored_segments"] == report["n_segments"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/watch?v=nw-2sPa7DAg",
    )
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--force-upload", action="store_true")
    parser.add_argument("--overwrite-segments", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--attention-axis", default="attention")
    parser.add_argument(
        "--persona-file",
        type=Path,
        default=Path("data/labels/synthetic_persona_haiku_clean.parquet"),
    )
    parser.add_argument(
        "--bmd-features-dir",
        type=Path,
        default=Path("data/features/tribe"),
    )
    parser.add_argument(
        "--bmd-annotations",
        type=Path,
        default=Path("data/raw/bold_moments/annotations.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
