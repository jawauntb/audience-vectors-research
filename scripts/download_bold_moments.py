"""Download BOLD Moments stimulus annotations from the public OpenNeuro
S3 bucket. No auth required, no form gating.

Optionally also fetches the stimulus videos by walking each annotation
entry's `MiT_url` link. Those URLs point at MIT csail's bucket and are
not guaranteed to stay live forever — the official BMD authors disclaim
responsibility for upkeep — so failures are logged and the script
continues.

Usage:
    uv run python scripts/download_bold_moments.py
    uv run python scripts/download_bold_moments.py --include-videos --limit 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

from audience_vectors.config import get_config

ANNOTATIONS_S3_URI = (
    "s3://openneuro.org/ds005165/derivatives/stimuli_metadata/annotations.json"
)
FIELDNAMES_S3_URI = (
    "s3://openneuro.org/ds005165/derivatives/stimuli_metadata/annotations_fieldnames.json"
)


def _check_aws_cli() -> None:
    if shutil.which("aws") is None:
        sys.exit(
            "aws CLI not found on PATH. Install via `brew install awscli` "
            "(or pip install awscli) and re-run."
        )


def _s3_cp(s3_uri: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", s3_uri, str(dst)],
        check=True,
    )


def _download_video(url: str, dst: Path, timeout: float = 60.0) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    if not url:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "audience-vectors/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            tmp = dst.with_suffix(dst.suffix + ".part")
            with tmp.open("wb") as out:
                shutil.copyfileobj(r, out)
            tmp.replace(dst)
        return True
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"  [skip] {dst.name}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Where to write annotations + videos (default: BOLD_MOMENTS_ROOT from .env).",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Also fetch stimulus MP4s via each entry's MiT_url. ~1102 clips, may take time.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on videos to fetch when --include-videos is set.",
    )
    parser.add_argument(
        "--refresh-annotations",
        action="store_true",
        help="Re-download annotations.json even if it already exists.",
    )
    args = parser.parse_args()

    cfg = get_config()
    root = args.root or Path("./data/raw/bold_moments").resolve()
    root.mkdir(parents=True, exist_ok=True)

    annotations_path = root / "annotations.json"
    fieldnames_path = root / "annotations_fieldnames.json"
    videos_dir = root / "videos"

    _check_aws_cli()

    if args.refresh_annotations or not annotations_path.exists():
        print(f"[s3] {ANNOTATIONS_S3_URI} -> {annotations_path}")
        _s3_cp(ANNOTATIONS_S3_URI, annotations_path)
    else:
        print(f"[ok] annotations.json already at {annotations_path}")

    if not fieldnames_path.exists():
        print(f"[s3] {FIELDNAMES_S3_URI} -> {fieldnames_path}")
        _s3_cp(FIELDNAMES_S3_URI, fieldnames_path)

    if not args.include_videos:
        print(
            "\n[done] annotations ready. To also fetch videos: "
            "`--include-videos` (slow; MiT_url links may have rotted)."
        )
        return

    with annotations_path.open() as fh:
        ann = json.load(fh)

    videos_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[videos] writing into {videos_dir}")
    ok, miss = 0, 0
    for i, (entry_id, entry) in enumerate(ann.items()):
        if args.limit and i >= args.limit:
            break
        matrix_name = entry.get("bmd_matrixfilename") or f"vid_idx{entry_id}"
        url = entry.get("MiT_url") or ""
        dst = videos_dir / f"{matrix_name}.mp4"
        if _download_video(url, dst):
            ok += 1
        else:
            miss += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}] ok={ok} miss={miss}")
    print(f"\n[done] videos: ok={ok} miss={miss}")
    print(f"       cfg paths root: {cfg.paths.data_root}")


if __name__ == "__main__":
    main()
