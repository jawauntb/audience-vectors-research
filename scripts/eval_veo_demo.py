"""Run TRIBE on Veo-generated clips and report steering result.

Steps:
  1. Upload all data/generated/veo/*.mp4 to the bmd-videos Modal volume under
     `generated/`.
  2. For each clip, call TRIBE via the existing service path → save activations.
  3. Project each onto the BMD memorability direction (trained over all 1022
     real BMD clips).
  4. Report paired comparison: did memorable-styled prompts produce a higher
     projection score than neutral prompts?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _upload_to_modal_volume(local_dir: Path) -> None:
    files = sorted(local_dir.glob("*.mp4"))
    if not files:
        raise SystemExit(f"no .mp4 in {local_dir}")
    for f in files:
        print(f"  [up] {f.name}")
        subprocess.run(
            ["modal", "volume", "put", "--force", "bmd-videos-v1",
             str(f), f"/generated/{f.name}"],
            check=True, capture_output=True,
        )


async def _run_tribe(paths: list[Path], output_dir: Path) -> dict[str, Path]:
    """For each generated clip, run TRIBE and persist activations."""
    from audience_vectors.features.tribe_extractor import TribeFeatureExtractor
    from audience_vectors.schemas import Segment

    output_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    for p in paths:
        # synthesize a Segment with the Modal volume path as media_path
        seg = Segment(
            sample_id=p.stem,
            video_id=p.stem,
            source_dataset="veo-demo",
            start_time=0.0,
            end_time=4.0,
            duration=4.0,
            media_path=f"/bmd-videos/generated/{p.name}",
        )
        segments.append(seg)

    extractor = TribeFeatureExtractor(
        output_dir=output_dir, max_concurrency=4,
    )
    print(f"[tribe] dispatching {len(segments)} clips")
    written = await extractor.extract_many(segments)
    return {Path(p).stem: Path(p) for p in written}


def _load_v_mem(features_dir: Path, bmd_annotations: Path) -> np.ndarray:
    """Train memorability direction on ALL BMD TRIBE features."""
    ann = json.loads(bmd_annotations.read_text())
    bmd = {f"bmd_vid_idx{e}": float(a["memorability_score"])
           for e, a in ann.items() if "memorability_score" in a}
    feats, scores = [], []
    for f in sorted(features_dir.glob("bmd_vid_idx*.npz")):
        sid = f.stem
        vid = sid.split("_seg_")[0]
        if vid not in bmd:
            continue
        p = np.load(f, allow_pickle=False)
        arr = np.asarray(p["frames"], dtype=np.float32)
        vec = arr.mean(axis=0) if arr.ndim == 2 else arr
        feats.append(vec)
        scores.append(bmd[vid])
    X = np.stack(feats); y = np.asarray(scores, dtype=np.float32)
    order = np.argsort(y)
    ne = int(len(y) * 0.30)
    v = X[order[-ne:]].mean(axis=0) - X[order[:ne]].mean(axis=0)
    v /= (np.linalg.norm(v) + 1e-12)
    print(f"[v_mem] trained on {len(y)} BMD clips, norm-1 direction in TRIBE space")
    return v


def _load_feature(p: Path) -> np.ndarray:
    payload = np.load(p, allow_pickle=False)
    arr = np.asarray(payload["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


async def main_async(args: argparse.Namespace) -> None:
    local = args.generated_dir
    if args.skip_upload:
        print("[skip] modal volume upload")
    else:
        print(f"[step 1/4] uploading {local} → modal volume bmd-videos-v1/generated/")
        _upload_to_modal_volume(local)

    paths = sorted(local.glob("*.mp4"))
    veo_features_dir = args.veo_features_dir
    veo_features_dir.mkdir(parents=True, exist_ok=True)
    needed = [p for p in paths if not (veo_features_dir / f"{p.stem}.npz").exists()]
    if needed:
        print(f"[step 2/4] running TRIBE on {len(needed)} clips")
        await _run_tribe(needed, veo_features_dir)
    else:
        print(f"[step 2/4] all features already cached")

    print(f"[step 3/4] training v_mem on real BMD clips")
    v_mem = _load_v_mem(args.real_features_dir, args.bmd_annotations)

    print(f"[step 4/4] projecting generated clips onto v_mem")
    pair_results: dict[str, dict[str, float | None]] = {}
    for p in sorted(veo_features_dir.glob("*.npz")):
        label = p.stem
        if "_mem" in label or "_neu" in label:
            pair_id, kind = label.rsplit("_", 1)
        else:
            continue
        score = float(_load_feature(p) @ v_mem)
        pair_results.setdefault(pair_id, {"mem": None, "neu": None})
        pair_results[pair_id][kind] = score

    rows = []
    deltas = []
    for pair_id, scores in sorted(pair_results.items()):
        if scores["mem"] is None or scores["neu"] is None:
            continue
        delta = scores["mem"] - scores["neu"]
        deltas.append(delta)
        rows.append({"pair": pair_id, "mem": scores["mem"], "neu": scores["neu"], "delta": delta})

    deltas_arr = np.asarray(deltas)
    n = len(deltas_arr)
    mean_d = float(deltas_arr.mean())
    std_d = float(deltas_arr.std(ddof=1)) if n > 1 else 0.0
    se = std_d / np.sqrt(n) if n > 1 else 0.0
    t_stat = mean_d / se if se > 0 else 0.0
    win_rate = float((deltas_arr > 0).mean())

    print()
    print(f"=== Veo prompt-level steering result (n={n} pairs) ===")
    print(f"  Mean Δ(memorable - neutral) on v_mem projection: {mean_d:+.4f}")
    print(f"  Stdev across pairs: {std_d:.4f}")
    print(f"  Paired t-stat: {t_stat:+.3f}   win rate (mem > neu): {win_rate*100:.1f}%")
    print()
    for r in rows:
        sign = "✓" if r["delta"] > 0 else "✗"
        print(f"  {sign} {r['pair']}: mem={r['mem']:+.4f}  neu={r['neu']:+.4f}  Δ={r['delta']:+.4f}")

    payload = {
        "n_pairs": n,
        "mean_delta": mean_d,
        "stdev_delta": std_d,
        "se": se,
        "paired_t": t_stat,
        "win_rate": win_rate,
        "rows": rows,
        "v_mem_norm": float(np.linalg.norm(v_mem)),
    }
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] wrote {out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, default=Path("data/generated/veo"))
    parser.add_argument("--real-features-dir", type=Path, default=Path("data/features/tribe"))
    parser.add_argument("--veo-features-dir", type=Path, default=Path("data/features/tribe_veo"))
    parser.add_argument("--bmd-annotations", type=Path, default=Path("data/raw/bold_moments/annotations.json"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/veo_demo.json"))
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
