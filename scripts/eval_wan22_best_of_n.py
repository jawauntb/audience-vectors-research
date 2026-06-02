"""Upload Wan2.2 best-of-N clips, run TRIBE, and score memorability direction."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, "src")
from audience_vectors.services.tribe_service import TribeService  # noqa: E402


def load_feat(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path, allow_pickle=False)["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def build_global_mem_direction(
    *,
    tribe_dir: Path,
    annotations_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    annotations = json.loads(annotations_path.read_text())
    mem_by_vid = {
        f"bmd_vid_idx{idx}": float(row["memorability_score"])
        for idx, row in annotations.items()
        if "memorability_score" in row
    }

    feats: list[np.ndarray] = []
    mems: list[float] = []
    for feat_path in sorted(tribe_dir.glob("bmd_vid_idx*.npz")):
        sid = feat_path.stem
        vid = sid.split("_seg_")[0]
        if vid not in mem_by_vid:
            continue
        feats.append(load_feat(feat_path))
        mems.append(mem_by_vid[vid])

    if not feats:
        raise RuntimeError(f"no BMD TRIBE features found in {tribe_dir}")

    x = np.stack(feats)
    y = np.asarray(mems, dtype=np.float32)
    order = np.argsort(y)
    n_extreme = int(len(y) * 0.30)
    if n_extreme < 1:
        raise RuntimeError("not enough BMD examples to build memorability direction")

    v_mem = x[order[-n_extreme:]].mean(axis=0) - x[order[:n_extreme]].mean(axis=0)
    v_mem /= np.linalg.norm(v_mem)
    bmd_scores = x @ v_mem
    meta = {
        "n_bmd_features": int(len(y)),
        "n_extreme": int(n_extreme),
        "bmd_projection_mean": float(np.mean(bmd_scores)),
        "bmd_projection_std": float(np.std(bmd_scores)),
        "bmd_projection_min": float(np.min(bmd_scores)),
        "bmd_projection_max": float(np.max(bmd_scores)),
    }
    return v_mem, meta


def modal_cli() -> str:
    cli = shutil.which("modal") or str(Path(".venv/bin/modal"))
    if not Path(cli).exists() and shutil.which("modal") is None:
        raise FileNotFoundError("modal CLI not found")
    return cli


def upload_to_bmd_volume(mp4s: list[Path], volume: str) -> None:
    cli = modal_cli()
    for mp4 in mp4s:
        dest = f"/generated/{mp4.name}"
        print(f"[upload] {mp4.name} -> {volume}:{dest}", flush=True)
        subprocess.run(
            [cli, "volume", "put", "--force", volume, str(mp4), dest],
            check=True,
        )


def seed_key_from_label(label: str) -> str:
    if label.endswith("_base") or label.endswith("_lora"):
        return label.rsplit("_", 1)[0]
    if re.search(r"_m[0-9mp]+_n\d+$", label):
        return re.sub(r"_m[0-9mp]+_n\d+$", "", label)
    match = re.search(r"vid_idx\d{4}", label)
    return match.group(0) if match else "ungrouped"


async def run_tribe(
    *,
    mp4s: list[Path],
    out_dir: Path,
    app_name: str | None,
    timeout: float,
    concurrency: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svc = TribeService(app_name)
    sem = asyncio.Semaphore(concurrency)
    written: list[Path] = []

    async def one(mp4: Path) -> None:
        out_path = out_dir / f"{mp4.stem}.npz"
        if out_path.exists():
            written.append(out_path)
            return
        modal_path = f"/bmd-videos/generated/{mp4.name}"
        async with sem:
            print(f"[tribe] {mp4.stem}", flush=True)
            result = await asyncio.wait_for(
                svc.predict_video(modal_path), timeout=timeout
            )
        if result is None or getattr(result, "frames", None) is None:
            print(f"[tribe] empty result for {mp4.stem}", flush=True)
            return
        frames = np.asarray(result.frames, dtype=np.float32)
        duration = float(getattr(result, "duration_seconds", 0.0) or 0.0)
        np.savez_compressed(
            out_path,
            frames=frames,
            duration_seconds=np.asarray([duration], dtype=np.float32),
            sample_id=np.asarray([mp4.stem]),
        )
        written.append(out_path)
        print(f"[tribe] wrote {out_path} frames={frames.shape}", flush=True)

    await asyncio.gather(*(one(mp4) for mp4 in mp4s))
    return sorted(written)


def score_outputs(
    *,
    feature_paths: list[Path],
    v_mem: np.ndarray,
    report_path: Path,
    manifest_path: Path,
    bmd_meta: dict[str, Any],
) -> None:
    rows = []
    for feature_path in feature_paths:
        vec = load_feat(feature_path)
        rows.append(
            {
                "label": feature_path.stem,
                "feature_path": str(feature_path),
                "v_mem_projection": float(vec @ v_mem),
            }
        )
    rows.sort(key=lambda row: row["v_mem_projection"], reverse=True)
    scores = [row["v_mem_projection"] for row in rows]
    summary = {
        "n": len(rows),
        "best": max(scores) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "min": min(scores) if scores else None,
        "best_minus_median": (
            float(max(scores) - np.median(scores)) if scores else None
        ),
    }
    by_seed: dict[str, dict[str, Any]] = {}
    for row in rows:
        seed = seed_key_from_label(str(row["label"]))
        by_seed.setdefault(seed, {"scores": []})["scores"].append(
            row["v_mem_projection"]
        )
    for seed, seed_payload in by_seed.items():
        seed_scores = seed_payload["scores"]
        seed_payload.update(
            {
                "n": len(seed_scores),
                "best": float(max(seed_scores)),
                "median": float(np.median(seed_scores)),
                "min": float(min(seed_scores)),
                "best_minus_median": float(max(seed_scores) - np.median(seed_scores)),
                "spread": float(max(seed_scores) - min(seed_scores)),
            }
        )
    per_seed_lifts = [
        payload["best_minus_median"]
        for payload in by_seed.values()
        if payload.get("n", 0) > 1
    ]
    if per_seed_lifts:
        summary["per_seed_best_minus_median_mean"] = float(np.mean(per_seed_lifts))
        summary["per_seed_best_minus_median_median"] = float(np.median(per_seed_lifts))
        summary["per_seed_best_minus_median_min"] = float(np.min(per_seed_lifts))
        summary["per_seed_best_minus_median_max"] = float(np.max(per_seed_lifts))
    payload: dict[str, Any] = {
        "summary": summary,
        "bmd_direction": bmd_meta,
        "scores": rows,
        "by_seed": dict(sorted(by_seed.items())),
    }
    if manifest_path.exists():
        payload["manifest"] = json.loads(manifest_path.read_text())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"[score] wrote {report_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("data/generated/wan22_best_of_n"),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features/tribe_wan22_best_of_n"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/reports/wan22_best_of_n_results.json"),
    )
    parser.add_argument("--tribe-app-name", default=None)
    parser.add_argument("--volume", default="bmd-videos-v1")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--tribe-dir", type=Path, default=Path("data/features/tribe"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/bold_moments/annotations.json"),
    )
    args = parser.parse_args()

    mp4s = sorted(args.generated_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"no mp4s found in {args.generated_dir}")

    if not args.skip_upload:
        upload_to_bmd_volume(mp4s, args.volume)

    feature_paths = await run_tribe(
        mp4s=mp4s,
        out_dir=args.feature_dir,
        app_name=args.tribe_app_name,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    v_mem, bmd_meta = build_global_mem_direction(
        tribe_dir=args.tribe_dir,
        annotations_path=args.annotations,
    )
    score_outputs(
        feature_paths=feature_paths,
        v_mem=v_mem,
        report_path=args.report_path,
        manifest_path=args.generated_dir / "manifest.json",
        bmd_meta=bmd_meta,
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
