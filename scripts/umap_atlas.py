"""UMAP atlas of viewer-response space.

Project all 1022 TRIBE feature vectors (20484-dim) to 2D, then render multiple
colored overlays:
  - BMD human memorability
  - Mean persona-predicted memorability
  - Cross-persona disagreement (stdev across personas)
  - K-means cluster ID over the 2D embedding

Saves both a JSON for the HTML paper and an SVG atlas figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import umap
from sklearn.cluster import KMeans


def _load_feature(path: Path) -> np.ndarray:
    p = np.load(path, allow_pickle=False)
    if "frames" in p.files:
        arr = np.asarray(p["frames"], dtype=np.float32)
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    return np.asarray(p["embedding"], dtype=np.float32)


def _bmd_lookup() -> dict[str, float]:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items()
        if "memorability_score" in e
    }


def main() -> None:
    feat_dir = Path("data/features/tribe")
    files = sorted(feat_dir.glob("*.npz"))
    print(f"[atlas] loading {len(files)} features")
    ids = []
    feats = []
    for f in files:
        feats.append(_load_feature(f))
        ids.append(f.stem)
    X = np.stack(feats)
    print(f"[atlas] X.shape={X.shape}")

    print("[atlas] fitting UMAP …")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=2, random_state=42, metric="cosine")
    Y = reducer.fit_transform(X)
    print(f"[atlas] embedding shape={Y.shape}")

    print("[atlas] k-means clustering (k=6)")
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(Y)
    cluster_id = km.labels_

    # BMD memorability per clip
    bmd = _bmd_lookup()
    bmd_score = []
    for sid in ids:
        vid = sid.split("_seg_")[0]
        bmd_score.append(bmd.get(vid, None))

    # Persona-derived overlays from persona_driving.json
    driving = json.loads(Path("data/reports/persona_driving.json").read_text())
    drv_segs = driving["segments"]
    drv_cols = driving["all_columns"]
    drv_scores = np.asarray(driving["scores"])
    sid_to_drv = {s: i for i, s in enumerate(drv_segs)}
    persona_cols = [drv_cols.index(p) for p in driving["persona_ids"]]
    bmd_col = drv_cols.index("BMD_human_global") if "BMD_human_global" in drv_cols else None

    mean_persona = []
    persona_stdev = []
    for sid in ids:
        i = sid_to_drv.get(sid)
        if i is None:
            mean_persona.append(None)
            persona_stdev.append(None)
        else:
            row = drv_scores[i, persona_cols]
            mean_persona.append(float(row.mean()))
            persona_stdev.append(float(row.std()))

    out = {
        "segments": ids,
        "x": Y[:, 0].tolist(),
        "y": Y[:, 1].tolist(),
        "bmd_score": bmd_score,
        "mean_persona": mean_persona,
        "persona_stdev": persona_stdev,
        "cluster": cluster_id.tolist(),
        "n_clusters": 6,
    }
    out_json = Path("data/reports/umap_atlas.json")
    out_json.write_text(json.dumps(out))
    print(f"[done] wrote {out_json} ({len(ids)} points)")

    # Stand-alone SVG (just by cluster for the static PDF / preview)
    svg_path = Path("data/reports/umap_atlas.svg")
    _render_svg(out, svg_path)
    print(f"[done] wrote {svg_path}")


def _render_svg(data: dict, path: Path) -> None:
    W, H = 720, 720
    M = 60
    xs, ys = np.asarray(data["x"]), np.asarray(data["y"])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    sx = lambda v: M + (v - x_min) / (x_max - x_min) * (W - 2 * M)
    sy = lambda v: M + (1 - (v - y_min) / (y_max - y_min)) * (H - 2 * M)

    palette = ["#6b3fa0", "#b8a060", "#2d7a3a", "#b03030", "#3a6ea5", "#a05050"]

    elts: list[str] = []
    elts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Georgia, serif">')
    elts.append(f'<rect width="{W}" height="{H}" fill="#fafaf7"/>')
    elts.append(f'<text x="{W/2}" y="32" text-anchor="middle" font-size="18" font-weight="600" fill="#111">UMAP atlas of viewer-response space (TRIBE activations, n={len(xs)})</text>')

    for i, c in enumerate(data["cluster"]):
        color = palette[int(c) % len(palette)]
        elts.append(f'<circle cx="{sx(xs[i]):.1f}" cy="{sy(ys[i]):.1f}" r="3.6" fill="{color}" fill-opacity="0.75"/>')

    # legend
    for k, color in enumerate(palette):
        ly = M + 20 + k * 22
        elts.append(f'<rect x="{W - M - 16}" y="{ly}" width="14" height="14" fill="{color}"/>')
        elts.append(f'<text x="{W - M - 22}" y="{ly + 12}" text-anchor="end" font-size="13" fill="#333">cluster {k}</text>')

    elts.append('</svg>')
    path.write_text("\n".join(elts))


if __name__ == "__main__":
    main()
