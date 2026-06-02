"""Build a small shareable video gallery for the audience-vectors report."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
BMD_VIDEOS = ROOT / "data" / "raw" / "bold_moments" / "videos"
VEO_VIDEOS = ROOT / "data" / "generated" / "veo"
WAN_VIDEOS = ROOT / "data" / "generated" / "wan22_best_of_n"
OUT = REPORTS / "video_gallery"
OUT_VIDEOS = OUT / "videos"


def segment_to_video(segment_id: str) -> Path:
    vid = segment_id.removeprefix("bmd_").split("_seg_")[0]
    return BMD_VIDEOS / f"{vid}.mp4"


def caption_lookup(persona_rows: list[dict]) -> dict[str, str]:
    captions: dict[str, str] = {}
    for row in persona_rows:
        for side in ("top", "bottom"):
            for item in row.get(side, []):
                captions.setdefault(item["segment_id"], item.get("caption", ""))
    return captions


def copy_video(src: Path, dest_name: str) -> str:
    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)
    dest = OUT_VIDEOS / dest_name
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)
    return f"videos/{dest_name}"


def bmd_card(item: dict) -> str:
    src = copy_video(segment_to_video(item["segment_id"]), f"{item['segment_id']}.mp4")
    score = item.get("score", item.get("projection"))
    score_text = f"{score:+.2f}" if isinstance(score, (int, float)) else ""
    kicker = html.escape(item.get("kicker", "BMD clip"))
    title = html.escape(item.get("title", item["segment_id"]))
    caption = html.escape(item.get("caption") or "No caption available.")
    segment = html.escape(item["segment_id"])
    return f"""
      <article class="clip-card">
        <video controls preload="metadata" src="{src}"></video>
        <div class="clip-body">
          <div class="meta-row"><span>{kicker}</span><span>{score_text}</span></div>
          <h3>{title}</h3>
          <p>{caption}</p>
          <code>{segment}</code>
        </div>
      </article>
    """


def veo_pair_card(pair: dict) -> str:
    pair_id = pair["pair"]
    mem_src = copy_video(VEO_VIDEOS / f"{pair_id}_mem.mp4", f"{pair_id}_mem.mp4")
    neu_src = copy_video(VEO_VIDEOS / f"{pair_id}_neu.mp4", f"{pair_id}_neu.mp4")
    delta = pair["delta"]
    win = "win" if delta > 0 else "loss"
    return f"""
      <article class="pair-card {win}">
        <div class="pair-head">
          <h3>{html.escape(pair_id)}</h3>
          <span>TRIBE delta {delta:+.2f}</span>
        </div>
        <div class="pair-grid">
          <div>
            <video controls preload="metadata" src="{mem_src}"></video>
            <strong>Memorable-styled prompt</strong>
          </div>
          <div>
            <video controls preload="metadata" src="{neu_src}"></video>
            <strong>Neutral prompt</strong>
          </div>
        </div>
      </article>
    """


def wan_card(row: dict) -> str:
    label = row["label"]
    src = copy_video(WAN_VIDEOS / f"{label}.mp4", f"{label}.mp4")
    score = float(row["v_mem_projection"])
    return f"""
      <article class="clip-card">
        <video controls preload="metadata" src="{src}"></video>
        <div class="clip-body">
          <div class="meta-row"><span>Wan2.2 best-of-N</span><span>{score:+.2f}</span></div>
          <h3>{html.escape(label)}</h3>
          <p>Open-weight Wan2.2-TI2V-5B candidate from the dog-in-snow seed. The score is projection on the BMD-derived TRIBE memorability direction.</p>
          <code>{html.escape(label)}.mp4</code>
        </div>
      </article>
    """


def main() -> None:
    persona_rows = json.loads((REPORTS / "share_persona_tops.json").read_text())
    driving = json.loads((REPORTS / "persona_driving.json").read_text())
    veo = json.loads((REPORTS / "veo_demo.json").read_text())
    wan_path = REPORTS / "wan22_best_of_n_results.json"
    wan_rows = []
    if wan_path.exists():
        wan_payload = json.loads(wan_path.read_text())
        wan_rows = [
            row
            for row in wan_payload.get("scores", [])
            if (WAN_VIDEOS / f"{row['label']}.mp4").exists()
        ]
    captions = caption_lookup(persona_rows)

    global_col = driving["all_columns"].index("BMD_human_global")
    global_rows = [
        {
            "segment_id": segment,
            "score": scores[global_col],
            "caption": captions.get(segment, ""),
            "kicker": "Human-pool memorability",
            "title": f"Global top {rank}",
        }
        for rank, (segment, scores) in enumerate(
            sorted(
                zip(driving["segments"], driving["scores"]),
                key=lambda row: row[1][global_col],
                reverse=True,
            )[:8],
            start=1,
        )
    ]

    persona_winners = [
        {
            "segment_id": row["top"][0]["segment_id"],
            "projection": row["top"][0]["projection"],
            "caption": row["top"][0]["caption"],
            "kicker": row["persona"],
            "title": f"{row['persona']} top pick",
        }
        for row in persona_rows
    ]

    persona_cols = [driving["all_columns"].index(p) for p in driving["persona_ids"]]
    polarizing = []
    for segment, scores in zip(driving["segments"], driving["scores"]):
        vals = [scores[i] for i in persona_cols]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        polarizing.append((var**0.5, segment, scores[global_col]))
    polarizing_rows = [
        {
            "segment_id": segment,
            "score": stdev,
            "caption": captions.get(segment, ""),
            "kicker": "Cross-persona disagreement",
            "title": f"Polarizing clip {rank}",
        }
        for rank, (stdev, segment, _global_score) in enumerate(
            sorted(polarizing, reverse=True)[:8],
            start=1,
        )
    ]

    veo_rows = [
        row for row in veo["rows"] if (VEO_VIDEOS / f"{row['pair']}_mem.mp4").exists()
    ]
    selected_veo = sorted(veo_rows, key=lambda row: abs(row["delta"]), reverse=True)[:4]

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Watch Examples · Synthetic Audience Vectors</title>
<style>
  :root {{
    --bg: #f7f5ef;
    --fg: #161514;
    --muted: #625d55;
    --panel: #fffdfa;
    --line: #ded7c8;
    --accent: #6b3fa0;
    --green: #287447;
    --red: #a33b3b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 34px 24px 72px; }}
  header {{ border-bottom: 2px solid var(--fg); padding-bottom: 18px; margin-bottom: 28px; }}
  a {{ color: var(--accent); text-decoration-thickness: 1px; }}
  h1 {{ font-size: clamp(2rem, 4vw, 3.6rem); line-height: 1.02; margin: 0 0 10px; letter-spacing: 0; }}
  h2 {{ margin: 34px 0 12px; font-size: 1.2rem; }}
  h3 {{ margin: 0 0 8px; font-size: 1rem; }}
  p {{ margin: 0 0 12px; color: var(--muted); }}
  .lede {{ max-width: 760px; font-size: 1.04rem; }}
  .note {{ padding: 12px 14px; border: 1px solid var(--line); background: #fff8e8; color: #514638; margin: 16px 0 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
  .clip-card, .pair-card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 0 rgba(0,0,0,0.03);
  }}
  video {{ width: 100%; aspect-ratio: 16 / 9; display: block; background: #151515; }}
  .clip-body {{ padding: 12px; }}
  .meta-row {{ display: flex; justify-content: space-between; gap: 10px; color: var(--accent); font-weight: 700; font-size: 0.77rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 7px; }}
  code {{ display: block; color: var(--muted); font-size: 0.74rem; overflow-wrap: anywhere; }}
  .pair-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
  .pair-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 12px 0; }}
  .pair-head span {{ font-weight: 800; color: var(--green); }}
  .pair-card.loss .pair-head span {{ color: var(--red); }}
  .pair-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 12px; }}
  .pair-grid strong {{ display: block; margin-top: 6px; font-size: 0.82rem; }}
  @media (max-width: 980px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .pair-list {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 560px) {{ main {{ padding: 24px 14px 56px; }} .grid, .pair-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <header>
    <a href="../paper.html">Back to paper</a>
    <h1>Watchable Examples</h1>
    <p class="lede">Selected source clips and generated clips from the Synthetic Audience Vectors experiments. The goal is to make the vector results feel concrete: what does "memorable" or "persona-specific" actually look like?</p>
    <p class="note"><strong>Sharing note:</strong> send this whole <code>video_gallery</code> folder, not only <code>index.html</code>. The videos are companion assets and are intentionally not embedded into the main paper HTML.</p>
  </header>

  <section>
    <h2>Human-Pool Memorability: Top Scored BMD Clips</h2>
    <div class="grid">
      {''.join(bmd_card(item) for item in global_rows)}
    </div>
  </section>

  <section>
    <h2>Persona Winners: Each Synthetic ICP's Top Pick</h2>
    <p>These are useful for the team because they show the audience-decomposition claim in plain English: different viewer profiles do not all pick the same memorable moment.</p>
    <div class="grid">
      {''.join(bmd_card(item) for item in persona_winners)}
    </div>
  </section>

  <section>
    <h2>Most Audience-Polarizing Clips</h2>
    <p>High cross-persona standard deviation: clips where different ICPs disagree most.</p>
    <div class="grid">
      {''.join(bmd_card(item) for item in polarizing_rows)}
    </div>
  </section>

  <section>
    <h2>Veo Prompt-Level Steering: Biggest Swings</h2>
    <p>This is the noisy generation test. It is included to show why the next step is conditioning-space steering rather than prompt rewriting.</p>
    <div class="pair-list">
      {''.join(veo_pair_card(row) for row in selected_veo)}
    </div>
  </section>

  <section>
    <h2>Wan2.2 Open-Weight Best-of-N Pilot</h2>
    <p>One image-seeded N=4 run on Wan2.2-TI2V-5B. This is not a powered result, but it shows the same TRIBE ranking recipe working on a modern open-weight video model with accessible activations.</p>
    <div class="grid">
      {''.join(wan_card(row) for row in wan_rows)}
    </div>
  </section>
</main>
</body>
</html>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html_text)
    print(f"wrote {OUT / 'index.html'}")
    print(f"copied videos to {OUT_VIDEOS}")


if __name__ == "__main__":
    main()
