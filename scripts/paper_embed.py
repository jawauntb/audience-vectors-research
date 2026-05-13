"""Inline JSON data into paper.html (replaces /*DATA:foo*/ placeholders)."""

from __future__ import annotations

import json
from pathlib import Path

R = Path("data/reports")
HTML = R / "paper.html"

SOURCES = {
    "cv_tribe": R / "cv_tribe_n1022.json",
    "patching_tribe": R / "patching_tribe.json",
    "patching_vjepa": R / "patching_vjepa.json",
    "persona_mem": R / "persona_directions.json",
    "persona_emot": R / "persona_directions_emotional.json",
    "driving": R / "persona_driving.json",
    "umap": R / "umap_atlas.json",
}


def main() -> None:
    text = HTML.read_text()
    for key, path in SOURCES.items():
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        data = json.loads(path.read_text())
        injected = json.dumps(data)
        marker = f"/*DATA:{key}*/"
        if marker not in text:
            print(f"[warn] marker {marker!r} not found in HTML")
            continue
        text = text.replace(marker, injected)
        print(f"[ok] embedded {key} ({len(injected):,} chars)")
    HTML.write_text(text)
    print(f"[done] wrote {HTML} ({len(text):,} chars total)")


if __name__ == "__main__":
    main()
