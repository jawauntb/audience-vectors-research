#!/bin/bash
# Runs after the haiku persona-labeling job finishes:
#   1. Post-process the haiku parquet to drop the `reason` field from the
#      `scores` struct (so persona_disagreement.py doesn't choke on strings).
#   2. Swap the haiku file in where persona analyses expect gemini-format data.
#   3. Re-run persona disagreement + persona-vs-BMD.
#   4. Restore gemini data, mark complete.

set -euo pipefail
cd /Users/jawaun/isc_mod
exec > /tmp/overnight_chain.log 2>&1

echo "[$(date)] wrapper start — waiting for haiku PID"
while pgrep -f label_persona_haiku >/dev/null; do
  sleep 30
done
echo "[$(date)] haiku finished, running downstream"

# 1. Strip `reason` from inside the scores struct (keeps top-level reason col)
.venv/bin/python - <<'PY'
import polars as pl
df = pl.read_parquet("data/labels/synthetic_persona_haiku.parquet")
# unnest scores, drop reason from struct fields, rebuild
scores = df.select("scores").unnest("scores")
score_cols = [c for c in scores.columns if c != "reason"]
df = df.with_columns(
    pl.struct([scores[c] for c in score_cols]).alias("scores"),
)
df.write_parquet("data/labels/synthetic_persona_haiku_clean.parquet")
print(f"[clean] wrote {len(df)} rows, score axes: {score_cols}")
PY

# 2. Backup gemini, swap haiku in at gemini path
mv data/labels/synthetic_persona_gemini.parquet data/labels/synthetic_persona_gemini.parquet.bak
cp data/labels/synthetic_persona_haiku_clean.parquet data/labels/synthetic_persona_gemini.parquet

# 3. Run downstream analyses
echo "[$(date)] running persona_disagreement"
.venv/bin/python scripts/eval_persona_disagreement.py > data/reports/persona_disagreement_haiku.md 2>&1 || true

echo "[$(date)] running persona_vs_bmd"
.venv/bin/python scripts/eval_persona_vs_bmd.py --output data/reports/persona_vs_bmd_haiku.md || true

# 4. Restore gemini data
rm data/labels/synthetic_persona_gemini.parquet
mv data/labels/synthetic_persona_gemini.parquet.bak data/labels/synthetic_persona_gemini.parquet

# 5. Per-persona contrastive directions on TRIBE — memorability axis
echo "[$(date)] running persona_directions (memorability)"
.venv/bin/python scripts/persona_directions.py \
  --axis memorability \
  --output data/reports/persona_directions.md || true

# 6. Per-persona contrastive directions — attention axis
echo "[$(date)] running persona_directions (attention)"
.venv/bin/python scripts/persona_directions.py \
  --axis attention \
  --output data/reports/persona_directions_attention.md || true

# 7. Per-persona contrastive directions — emotional_intensity axis
echo "[$(date)] running persona_directions (emotional_intensity)"
.venv/bin/python scripts/persona_directions.py \
  --axis emotional_intensity \
  --output data/reports/persona_directions_emotional.md || true

# 8. Compile final report tying everything together
echo "[$(date)] compiling final report"
.venv/bin/python scripts/compile_final_report.py || true

echo "[$(date)] OVERNIGHT CHAIN COMPLETE" > /tmp/overnight_done.flag
echo "[$(date)] done"
