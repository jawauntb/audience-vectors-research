#!/bin/bash
# Wait for all parallel jobs, run TRIBE on new clips, do all analyses,
# update paper + PDF + arena_demo site. Runs autonomously.

set -uo pipefail
cd /Users/jawaun/isc_mod
exec > /tmp/final_chain.log 2>&1
echo "=========================================="
echo "[$(date)] FINAL CHAIN START"
echo "=========================================="

# === STEP 1: wait for all 3 generation jobs ===
while pgrep -f "svd_per_persona_steering|svd_alpha_plus_bestofn|veo_best_of_n_more" >/dev/null; do
  echo "[$(date +%H:%M:%S)] waiting on generation jobs..."
  echo "  per-persona:  $(ls data/generated/svd_per_persona/*.mp4 2>/dev/null | wc -l | tr -d ' ')/108"
  echo "  α+bon:        $(ls data/generated/svd_alpha_bon/*.mp4 2>/dev/null | wc -l | tr -d ' ')/30"
  echo "  veo more:     $(ls data/generated/veo_best_of_n/p04_*.mp4 data/generated/veo_best_of_n/p05_*.mp4 data/generated/veo_best_of_n/p06_*.mp4 data/generated/veo_best_of_n/p07_*.mp4 data/generated/veo_best_of_n/p08_*.mp4 2>/dev/null | wc -l | tr -d ' ')/40"
  sleep 60
done
echo "[$(date)] all generation jobs finished"

# === STEP 2: upload all new clips to Modal volume ===
echo "[$(date)] uploading new clips to Modal volume"
for d in data/generated/svd_per_persona data/generated/svd_alpha_bon data/generated/veo_best_of_n; do
  for f in $d/*.mp4; do
    [ -f "$f" ] || continue
    bn=$(basename $f)
    # Skip if already on volume (best-effort)
    .venv/bin/modal volume put --force bmd-videos-v1 "$f" "/generated/${bn}" >/dev/null 2>&1
  done
done
echo "[$(date)] uploads done"

# === STEP 3: run TRIBE on the new clips ===
echo "[$(date)] running TRIBE on new clips"
.venv/bin/python -u <<'PY'
import asyncio, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from audience_vectors.services.tribe_service import TribeService

PAIRS = [
    ("data/generated/svd_per_persona",   "data/features/tribe_svd_per_persona"),
    ("data/generated/svd_alpha_bon",     "data/features/tribe_svd_alpha_bon"),
    ("data/generated/veo_best_of_n",     "data/features/tribe_veo_bon"),
]

async def main():
    svc = TribeService()
    sem = asyncio.Semaphore(6)
    async def one(p, out_dir):
        out = Path(out_dir) / f"{p.stem}.npz"
        if out.exists(): return
        async with sem:
            try:
                r = await asyncio.wait_for(svc.predict_video(f"/bmd-videos/generated/{p.name}"), timeout=300.0)
            except Exception as exc:
                print(f"  ✗ {p.stem}: {type(exc).__name__}", flush=True); return
        if r is None or getattr(r, "frames", None) is None: return
        np.savez_compressed(out, frames=np.asarray(r.frames, dtype=np.float32), sample_id=np.asarray([p.stem]))
        print(f"  ✓ {p.stem}", flush=True)
    tasks = []
    for src, out_dir in PAIRS:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for p in sorted(Path(src).glob("*.mp4")):
            tasks.append(one(p, out_dir))
    await asyncio.gather(*tasks)
    for src, out_dir in PAIRS:
        print(f"  {Path(out_dir).name}: {len(list(Path(out_dir).glob('*.npz')))} features", flush=True)
asyncio.run(main())
PY
echo "[$(date)] TRIBE done"

# === STEP 4: compute all analyses ===
echo "[$(date)] computing analyses"
.venv/bin/python scripts/analyze_final.py
echo "[$(date)] analyses done"

# === STEP 5: rebuild the arena demo data (Veo got bigger) ===
.venv/bin/python <<'PY'
import json, numpy as np, polars as pl
from pathlib import Path

ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
bmd = {f"bmd_vid_idx{e}": float(a["memorability_score"])
       for e, a in ann.items() if "memorability_score" in a}
def load_feat(p):
    arr = np.asarray(np.load(p, allow_pickle=False)["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr

feats, mems, sids_bmd = [], [], []
for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
    sid = f.stem; vid = sid.split("_seg_")[0]
    if vid not in bmd: continue
    feats.append(load_feat(f)); mems.append(bmd[vid]); sids_bmd.append(sid)
X = np.stack(feats); y = np.asarray(mems, dtype=np.float32)
order = np.argsort(y); ne = int(len(y) * 0.30)
v_global = X[order[-ne:]].mean(axis=0) - X[order[:ne]].mean(axis=0)
v_global /= np.linalg.norm(v_global)

persona_df = pl.read_parquet("data/labels/synthetic_persona_haiku_clean.parquet")
ss = persona_df.select("scores").unnest("scores")
rows = persona_df.with_columns(ss["memorability"].alias("_s")).select(["persona_id","segment_id","_s"]).to_dicts()
by_p = {}
for r in rows:
    if r["_s"] is None: continue
    by_p.setdefault(r["persona_id"], {})[r["segment_id"]] = float(r["_s"])
sid_to_idx = {s: i for i, s in enumerate(sids_bmd)}
pdirs = {}
for p, ssd in by_p.items():
    pairs = [(sid_to_idx[s], v) for s, v in ssd.items() if s in sid_to_idx]
    if len(pairs) < 50: continue
    idxs = np.asarray([i for i, _ in pairs]); pscs = np.asarray([s for _, s in pairs], dtype=np.float32)
    o = np.argsort(pscs); ne_p = int(len(pscs) * 0.30); Xp = X[idxs]
    pdirs[p] = (Xp[o[-ne_p:]].mean(axis=0) - Xp[o[:ne_p]].mean(axis=0))
    pdirs[p] /= np.linalg.norm(pdirs[p])

arena = {"svd": {}, "veo": {}}
for kind, src in [("svd", "data/features/tribe_best_of_n"), ("veo", "data/features/tribe_veo_bon")]:
    for f in sorted(Path(src).glob("*.npz")):
        name = f.stem
        if "_n" not in name: continue
        seed, n_str = name.rsplit("_n", 1)
        try: n = int(n_str)
        except: continue
        vec = load_feat(f)
        sc = {p: float(vec @ vd) for p, vd in pdirs.items()}
        sc["global"] = float(vec @ v_global)
        arena[kind].setdefault(seed, []).append({"n": n, "scores": sc})
for k in arena:
    for seed in arena[k]:
        arena[k][seed].sort(key=lambda x: x["n"])

pp = Path("data/generated/veo_best_of_n/prompts.json")
veo_prompts = json.loads(pp.read_text()) if pp.exists() else []
Path("data/reports/arena_demo_data.json").write_text(json.dumps({
    "arena": arena, "veo_prompts": veo_prompts, "personas": sorted(pdirs.keys()),
}))
print(f"[arena] rebuilt: svd {list(arena['svd'].keys())}, veo {list(arena['veo'].keys())}")
PY

# embed into demo
.venv/bin/python <<'PY'
from pathlib import Path
p = Path("data/reports/arena_demo.html")
text = p.read_text()
js = Path("data/reports/arena_demo_data.json").read_text()
import re
m = re.search(r'(<script id="data" type="application/json">)([^<]*)(</script>)', text)
if m:
    text = text[:m.start(2)] + js + text[m.end(2):]
    p.write_text(text)
    print(f"[demo] re-embedded {len(js)} chars")
else:
    print("[demo] couldn't find data script tag")
PY

# === STEP 6: rebuild paper PDF ===
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless --disable-gpu --print-to-pdf=/Users/jawaun/isc_mod/data/reports/paper.pdf --no-pdf-header-footer file:///Users/jawaun/isc_mod/data/reports/paper.html 2>&1 | tail -1

echo "[$(date)] CHAIN COMPLETE" > /tmp/final_chain_done.flag
echo "[$(date)] CHAIN COMPLETE"
