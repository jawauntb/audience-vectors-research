#!/bin/bash
# After Veo finishes: upload to Modal → run TRIBE → eval → re-embed paper → regenerate PDF
set -euo pipefail
cd /Users/jawaun/isc_mod
exec > /tmp/veo_chain.log 2>&1

echo "[$(date)] waiting for veo_steering_demo …"
while pgrep -f veo_steering_demo >/dev/null; do
  sleep 30
done
echo "[$(date)] veo done; clips: $(ls data/generated/veo/*.mp4 | wc -l)"

# Step 1+2+3+4: orchestrated by eval_veo_demo (uploads, runs TRIBE, scores)
echo "[$(date)] running eval_veo_demo (upload + TRIBE + project)"
.venv/bin/python scripts/eval_veo_demo.py || true

# Inject the JSON into the paper
echo "[$(date)] embedding veo_demo.json"
.venv/bin/python - <<'PY'
from pathlib import Path
import json
p = Path("data/reports/paper.html")
text = p.read_text()
if not Path("data/reports/veo_demo.json").exists():
    print("[warn] no veo_demo.json — skip embed")
else:
    js = Path("data/reports/veo_demo.json").read_text()
    tag = f'<script id="data-veo-demo" type="application/json">{js}</script>\n'
    marker = '<script id="data-driving"'
    if 'id="data-veo-demo"' in text:
        # already exists — replace by regex-ish search
        import re
        text = re.sub(
            r'<script id="data-veo-demo"[^>]*>.*?</script>\n',
            tag, text, count=1, flags=re.DOTALL,
        )
    elif marker in text:
        text = text.replace(marker, tag + marker, 1)
    p.write_text(text)
    print(f"[ok] embedded veo_demo.json ({len(js)} chars)")
PY

# Regenerate PDF
echo "[$(date)] rebuilding PDF"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless --disable-gpu --print-to-pdf=/Users/jawaun/isc_mod/data/reports/paper.pdf --no-pdf-header-footer file:///Users/jawaun/isc_mod/data/reports/paper.html

echo "[$(date)] VEO CHAIN COMPLETE" > /tmp/veo_chain_done.flag
echo "[$(date)] done"
