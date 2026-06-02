"""Build a standalone Prolific-style HTML survey from selector pairwise tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video Memorability Selector Study</title>
<style>
  :root {
    --bg: #ece8df;
    --panel: #ece8df;
    --ink: #24231f;
    --muted: #68645b;
    --line-light: #fffaf0;
    --line-dark: #cfc7b8;
    --accent: #3e6f64;
    --accent-2: #8d4f3a;
    --shadow-a: rgba(255,255,255,.86);
    --shadow-b: rgba(110,96,76,.24);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.45;
  }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 32px 22px 72px; }
  .soft {
    background: var(--panel);
    border: 1px solid rgba(255,255,255,.42);
    border-radius: 18px;
    box-shadow: 14px 14px 30px var(--shadow-b), -14px -14px 30px var(--shadow-a);
  }
  .intro, .trial, .done { padding: 28px; }
  h1 { margin: 0 0 8px; font-size: clamp(28px, 4vw, 46px); letter-spacing: 0; }
  h2 { margin: 0 0 10px; font-size: 22px; letter-spacing: 0; }
  p { margin: 0 0 14px; }
  .muted { color: var(--muted); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 22px 0; }
  .video-card {
    padding: 14px;
    border-radius: 16px;
    box-shadow: inset 7px 7px 16px rgba(110,96,76,.18), inset -7px -7px 16px rgba(255,255,255,.78);
  }
  video { width: 100%; aspect-ratio: 16/9; object-fit: contain; background: #111; border-radius: 12px; display: block; }
  .pick {
    width: 100%;
    margin-top: 12px;
    padding: 14px 16px;
    border: 0;
    border-radius: 14px;
    color: var(--ink);
    background: var(--panel);
    font-weight: 750;
    cursor: pointer;
    box-shadow: 8px 8px 16px var(--shadow-b), -8px -8px 16px var(--shadow-a);
  }
  .pick:hover { color: var(--accent); }
  .pick.selected { background: var(--accent); color: white; box-shadow: inset 5px 5px 10px rgba(0,0,0,.16), inset -5px -5px 10px rgba(255,255,255,.10); }
  .bar { height: 10px; background: var(--line-dark); border-radius: 999px; overflow: hidden; margin: 18px 0 4px; }
  .fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
  input, select, textarea {
    width: 100%;
    padding: 12px 14px;
    border: 1px solid var(--line-dark);
    border-radius: 12px;
    background: rgba(255,255,255,.35);
    color: var(--ink);
    font: inherit;
  }
  label { display: block; margin: 14px 0 6px; font-weight: 700; }
  .actions { display: flex; justify-content: flex-end; gap: 12px; margin-top: 18px; }
  .btn {
    border: 0;
    border-radius: 14px;
    padding: 12px 18px;
    font-weight: 800;
    cursor: pointer;
    background: var(--accent);
    color: white;
    box-shadow: 8px 8px 16px var(--shadow-b), -8px -8px 16px var(--shadow-a);
  }
  .btn.secondary { background: var(--panel); color: var(--ink); }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .pill {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    color: var(--muted);
    box-shadow: inset 4px 4px 8px rgba(110,96,76,.14), inset -4px -4px 8px rgba(255,255,255,.7);
    font-size: 13px;
    font-weight: 750;
  }
  .hidden { display: none; }
  .json { height: 220px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
  @media (max-width: 760px) {
    .grid { grid-template-columns: 1fr; }
    .intro, .trial, .done { padding: 20px; }
  }
</style>
</head>
<body>
<main class="wrap">
  <section id="intro" class="intro soft">
    <span class="pill">Video memory study</span>
    <h1>Which clip sticks?</h1>
    <p class="muted">You will watch short AI-generated video pairs and choose the one that feels more memorable. There are no right answers; we want your first honest impression.</p>
    <p>Watch both videos fully before choosing. Use headphones only if you want; the clips are evaluated visually.</p>
    <label for="prolific-id">Prolific ID</label>
    <input id="prolific-id" autocomplete="off" placeholder="Auto-filled from URL when available">
    <label for="age">Age range</label>
    <select id="age">
      <option value="">Select...</option>
      <option>18-24</option><option>25-34</option><option>35-44</option><option>45-54</option><option>55+</option>
    </select>
    <label for="native-en">Native English speaker?</label>
    <select id="native-en">
      <option value="">Select...</option>
      <option value="yes">Yes</option>
      <option value="no">No</option>
    </select>
    <div class="actions">
      <button class="btn" onclick="startStudy()">Start study</button>
    </div>
  </section>

  <section id="trial" class="trial soft hidden">
    <span class="pill">Pair <span id="trial-idx">1</span> / <span id="trial-total">24</span></span>
    <div class="bar"><div id="progress" class="fill"></div></div>
    <h2>Which video would be more memorable?</h2>
    <p class="muted">Watch both clips, then pick the one you would be more likely to remember after seeing many similar clips.</p>
    <div class="grid">
      <div class="video-card">
        <video id="left-video" controls preload="metadata" playsinline></video>
        <button id="left-pick" class="pick" onclick="pickSide('left')">Choose left video</button>
      </div>
      <div class="video-card">
        <video id="right-video" controls preload="metadata" playsinline></video>
        <button id="right-pick" class="pick" onclick="pickSide('right')">Choose right video</button>
      </div>
    </div>
    <div class="actions">
      <button class="btn secondary" onclick="replayBoth()">Replay both</button>
      <button id="next" class="btn" onclick="nextTrial()" disabled>Next</button>
    </div>
  </section>

  <section id="done" class="done soft hidden">
    <span class="pill">Complete</span>
    <h1>Thank you</h1>
    <p class="muted">Download or copy this response JSON for collection. If this is hosted with a response endpoint later, the same payload can be POSTed automatically.</p>
    <textarea id="response-json" class="json" readonly></textarea>
    <div class="actions">
      <button class="btn" onclick="downloadJSON()">Download JSON</button>
    </div>
  </section>
</main>

<script>
const TASKS = __TASK_JSON__;
const TRIALS_PER_PARTICIPANT = __TRIAL_COUNT__;
const ASSET_BASE = "../../../";

let trials = [];
let idx = 0;
let currentChoice = null;
let responses = [];
let startedAt = null;

function queryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name) || params.get(name.toUpperCase()) || "";
}

function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(a) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

function shuffle(arr, rng) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function chooseTrials(pid) {
  const rng = mulberry32(hashString(pid || String(Date.now())));
  const byComparison = {};
  for (const task of TASKS) {
    if (!byComparison[task.comparison]) byComparison[task.comparison] = [];
    byComparison[task.comparison].push(task);
  }
  let selected = [];
  const comparisons = shuffle(Object.keys(byComparison), rng);
  const perComparison = Math.max(1, Math.floor(TRIALS_PER_PARTICIPANT / comparisons.length));
  for (const comparison of comparisons) {
    selected = selected.concat(shuffle(byComparison[comparison], rng).slice(0, perComparison));
  }
  if (selected.length < TRIALS_PER_PARTICIPANT) {
    const used = new Set(selected.map(t => t.task_id));
    const rest = shuffle(TASKS.filter(t => !used.has(t.task_id)), rng);
    selected = selected.concat(rest.slice(0, TRIALS_PER_PARTICIPANT - selected.length));
  }
  return shuffle(selected.slice(0, TRIALS_PER_PARTICIPANT), rng);
}

function assetURL(path) {
  if (path.startsWith("http") || path.startsWith("file:")) return path;
  return ASSET_BASE + path;
}

function startStudy() {
  const pid = document.getElementById("prolific-id").value.trim() || "anonymous";
  trials = chooseTrials(pid);
  startedAt = new Date().toISOString();
  document.getElementById("trial-total").textContent = String(trials.length);
  document.getElementById("intro").classList.add("hidden");
  document.getElementById("trial").classList.remove("hidden");
  renderTrial();
}

function renderTrial() {
  const trial = trials[idx];
  currentChoice = null;
  document.getElementById("next").disabled = true;
  document.getElementById("left-pick").classList.remove("selected");
  document.getElementById("right-pick").classList.remove("selected");
  document.getElementById("trial-idx").textContent = String(idx + 1);
  document.getElementById("progress").style.width = `${(idx / trials.length) * 100}%`;
  const left = document.getElementById("left-video");
  const right = document.getElementById("right-video");
  left.src = assetURL(trial.left.path);
  right.src = assetURL(trial.right.path);
  left.load();
  right.load();
}

function pickSide(side) {
  currentChoice = side;
  document.getElementById("left-pick").classList.toggle("selected", side === "left");
  document.getElementById("right-pick").classList.toggle("selected", side === "right");
  document.getElementById("next").disabled = false;
}

function replayBoth() {
  const left = document.getElementById("left-video");
  const right = document.getElementById("right-video");
  left.currentTime = 0;
  right.currentTime = 0;
  left.play();
  right.play();
}

function nextTrial() {
  if (!currentChoice) return;
  const trial = trials[idx];
  const chosen = trial[currentChoice];
  const other = trial[currentChoice === "left" ? "right" : "left"];
  responses.push({
    task_id: trial.task_id,
    seed: trial.seed,
    comparison: trial.comparison,
    chosen_side: currentChoice,
    chosen_policy: chosen.policy,
    chosen_label: chosen.label,
    other_policy: other.policy,
    target_policy: trial.target_policy,
    baseline_policy: trial.baseline_policy,
    chose_target: chosen.policy === trial.target_policy,
    timestamp: new Date().toISOString()
  });
  idx += 1;
  if (idx >= trials.length) finishStudy();
  else renderTrial();
}

function finishStudy() {
  document.getElementById("trial").classList.add("hidden");
  document.getElementById("done").classList.remove("hidden");
  document.getElementById("progress").style.width = "100%";
  const payload = {
    schema_version: 1,
    study: "neurips_memorability_selector_pilot",
    prolific_id: document.getElementById("prolific-id").value.trim(),
    age: document.getElementById("age").value,
    native_english: document.getElementById("native-en").value,
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    n_trials: trials.length,
    responses
  };
  document.getElementById("response-json").value = JSON.stringify(payload, null, 2);
}

function downloadJSON() {
  const blob = new Blob([document.getElementById("response-json").value], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const pid = document.getElementById("prolific-id").value.trim() || "anonymous";
  a.href = url;
  a.download = `selector_pilot_${pid}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

document.getElementById("prolific-id").value = queryParam("PROLIFIC_PID") || queryParam("participant_id");
</script>
</body>
</html>
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "current_selector_pairwise_tasks.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "current_selector_prolific_survey.html"
        ),
    )
    parser.add_argument("--trials-per-participant", type=int, default=24)
    args = parser.parse_args()

    payload = load_json(args.tasks)
    tasks_json = json.dumps(payload["tasks"], separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__TASK_JSON__", tasks_json).replace(
        "__TRIAL_COUNT__", str(args.trials_per_participant)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"[done] wrote {args.out}")
    print(f"[done] embedded tasks: {len(payload['tasks'])}")


if __name__ == "__main__":
    main()
