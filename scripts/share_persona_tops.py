"""Build a shareable per-persona top/bottom clips table with Gemini captions.

For each of the 12 personas, find their top-5 highest-scored and bottom-5 lowest-
scored clips on the memorability projection (TRIBE features · persona_direction),
then join with the original Gemini segment description so the team can SEE
what each persona considers memorable.

Outputs:
  - data/reports/share_persona_tops.md  (paste-friendly markdown)
  - data/reports/share_persona_tops.json (machine-readable)
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def main() -> None:
    driving = json.loads(Path("data/reports/persona_driving.json").read_text())
    segs = driving["segments"]
    cols = driving["all_columns"]
    scores = driving["scores"]
    personas = driving["persona_ids"]

    # Pre-index gemini captions
    g = pl.read_parquet("data/labels/synthetic_gemini.parquet")
    g = g.unique(subset=["segment_id"])
    sid_to_caption = {r["segment_id"]: (r["reason"] or "")
                      for r in g.iter_rows(named=True)}

    md_lines = [
        "# Persona-specific top clips (BMD)",
        "",
        "For each synthetic ICP / persona, the 5 clips that score **highest** on its",
        "memorability direction in TRIBE activation space, plus the 5 that score",
        "lowest. Captions are Gemini's segment descriptions (zero-shot).",
        "",
        "The pattern to see: different personas pick out qualitatively different",
        "content as memorable. Fast-scroll vs cinematic vs technical-evaluator should",
        "look genuinely different.",
        "",
    ]

    share_payload = []
    for p in personas:
        idx = cols.index(p)
        order = sorted(range(len(segs)), key=lambda i: -scores[i][idx])
        top5 = order[:5]
        bot5 = list(reversed(order[-5:]))

        md_lines.append(f"## {p}")
        md_lines.append("")
        md_lines.append(f"**Highest-scoring on {p}'s direction:**")
        md_lines.append("")
        md_lines.append("| rank | clip | projection | Gemini description |")
        md_lines.append("|---:|---|---:|---|")
        share_top = []
        for r, i in enumerate(top5):
            cap = sid_to_caption.get(segs[i], "").strip()[:160].replace("|", "/")
            md_lines.append(f"| {r+1} | `{segs[i]}` | {scores[i][idx]:+.3f} | {cap} |")
            share_top.append({"rank": r+1, "segment_id": segs[i], "projection": scores[i][idx], "caption": cap})
        md_lines.append("")

        md_lines.append(f"**Lowest-scoring on {p}'s direction:**")
        md_lines.append("")
        md_lines.append("| rank | clip | projection | Gemini description |")
        md_lines.append("|---:|---|---:|---|")
        share_bot = []
        for r, i in enumerate(bot5):
            cap = sid_to_caption.get(segs[i], "").strip()[:160].replace("|", "/")
            md_lines.append(f"| {r+1} | `{segs[i]}` | {scores[i][idx]:+.3f} | {cap} |")
            share_bot.append({"rank": r+1, "segment_id": segs[i], "projection": scores[i][idx], "caption": cap})
        md_lines.append("")

        share_payload.append({"persona": p, "top": share_top, "bottom": share_bot})

    out_md = Path("data/reports/share_persona_tops.md")
    out_md.write_text("\n".join(md_lines))
    out_json = Path("data/reports/share_persona_tops.json")
    out_json.write_text(json.dumps(share_payload, indent=2))
    print(f"[done] wrote {out_md} ({len(md_lines)} lines)")
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
