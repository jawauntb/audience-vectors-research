"""Build static launch assets for the two-session recognition-memory study.

This is a launch-prep utility, not a study runner. It reads the screened
recognition production manifest and recognition-memory design, gathers every
required MP4, writes Session 1 and Session 2 Prolific-ready HTML, and builds a
static-site archive suitable for hosting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_DESIGN = ARTIFACT_DIR / "content_pocket_recognition_memory_design_20260608.json"
DEFAULT_PRODUCTION_MANIFEST = (
    ARTIFACT_DIR / "content_pocket_recognition_stimulus_production_manifest_20260608.json"
)
DEFAULT_VIDEO_SCREENING = (
    ARTIFACT_DIR / "content_pocket_recognition_video_screening_20260608.json"
)
DEFAULT_OUT_MANIFEST = ARTIFACT_DIR / "content_pocket_recognition_launch_assets_20260608.json"
DEFAULT_OUT_MD = ARTIFACT_DIR / "content_pocket_recognition_launch_assets_20260608.md"
DEFAULT_URL_MAP = ARTIFACT_DIR / "content_pocket_recognition_hosted_video_url_map_20260608.json"
DEFAULT_SESSION1_HTML = ARTIFACT_DIR / "content_pocket_recognition_session1_prolific_20260608.html"
DEFAULT_SESSION2_HTML = ARTIFACT_DIR / "content_pocket_recognition_session2_prolific_20260608.html"
DEFAULT_SITE_DIR = Path("data/sites/content_pocket_recognition_launch_20260608")
DEFAULT_ARCHIVE = Path("data/sites/content_pocket_recognition_launch_20260608.tar.gz")
DEFAULT_FALLBACK_ROOTS = (
    Path("/Users/jawaun/isc_mod"),
    Path("/Users/jawaun/.codex/worktrees/descriptor-conditioned-replication-run-20260608/isc_mod"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "video"


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def source_roots(raw_roots: list[Path]) -> list[Path]:
    roots = [Path.cwd(), *raw_roots]
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        key = str(resolved)
        if key not in seen and resolved.exists():
            deduped.append(resolved)
            seen.add(key)
    return deduped


def resolve_local_path(local_path: str, roots: list[Path]) -> Path:
    path = Path(local_path)
    if path.is_absolute() and path.exists():
        return path
    for root in roots:
        candidate = root / local_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(local_path)


def generation_jobs_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(job["job_id"]): job for job in manifest["generation_jobs"]}


def required_video_paths(
    *,
    design: dict[str, Any],
    production_manifest: dict[str, Any],
) -> list[str]:
    paths: set[str] = set()
    for job in production_manifest["generation_jobs"]:
        paths.add(str(job["output_video"]["path"]))
    for form in design["session_forms"]:
        for row in form["session1"]["analysis_encoding_targets"]:
            paths.add(str(row["old_video_path"]))
        for row in form["session2"]["analysis_recognition_trials"]:
            paths.add(str(row["old_video_path"]))
    return sorted(paths)


def asset_filename(local_path: str, source_path: Path) -> str:
    digest = sha256_file(source_path)[:12]
    return f"videos/{slug(Path(local_path).stem)}_{digest}.mp4"


def build_video_assets(
    *,
    local_paths: list[str],
    roots: list[Path],
    site_dir: Path,
    base_url: str,
    video_base_url: str,
    copy_videos: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    videos_dir = site_dir / "videos"
    if copy_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for local_path in local_paths:
        try:
            source_path = resolve_local_path(local_path, roots)
        except FileNotFoundError:
            missing.append(local_path)
            continue
        asset_path = asset_filename(local_path, source_path)
        dest = site_dir / asset_path
        if copy_videos:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest)
        if video_base_url:
            hosted_url = f"{normalize_base_url(video_base_url)}/{Path(asset_path).name}"
        elif base_url:
            hosted_url = f"{normalize_base_url(base_url)}/{asset_path}"
        else:
            hosted_url = ""
        assets[local_path] = {
            "local_path": local_path,
            "source_path": str(source_path),
            "asset_path": asset_path,
            "hosted_url": hosted_url,
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        }
    return assets, missing


def video_ref(local_path: str, assets: dict[str, dict[str, Any]]) -> str:
    return assets[local_path]["hosted_url"] or assets[local_path]["asset_path"]


def filler_trials(
    jobs: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    trials = []
    for index in range(count):
        old_id = f"filler_old_v{index:02d}"
        lure_id = f"filler_lure_v{index:02d}"
        old_job = jobs[old_id]
        lure_job = jobs[lure_id]
        old_side = "left" if index % 2 == 0 else "right"
        trials.append(
            {
                "trial_id": f"{old_id}_vs_{lure_id}_recognition",
                "target_id": old_id,
                "arm_id": "unrelated_filler",
                "analysis_group": "unrelated_filler",
                "old_video_url": video_ref(str(old_job["output_video"]["path"]), assets),
                "lure_video_url": video_ref(str(lure_job["output_video"]["path"]), assets),
                "old_side": old_side,
                "correct_choice": old_side,
                "question": "Which clip did you see in the first part?",
            }
        )
    return trials


def session_payload(
    *,
    design: dict[str, Any],
    production_manifest: dict[str, Any],
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    jobs = generation_jobs_by_id(production_manifest)
    filler_old_count = sum(
        1 for job in jobs.values() if job.get("role") == "filler_old_video"
    )
    filler_recognition_count = sum(
        1 for job in jobs.values() if job.get("role") == "filler_lure_video"
    )
    filler_old_jobs = [
        jobs[f"filler_old_v{index:02d}"]
        for index in range(filler_old_count)
    ]
    filler_encoding_targets = [
        {
            "trial_id": str(job["job_id"]),
            "target_id": str(job["job_id"]),
            "arm_id": "unrelated_filler",
            "analysis_group": "unrelated_filler",
            "video_url": video_ref(str(job["output_video"]["path"]), assets),
        }
        for job in filler_old_jobs
    ]
    filler_recognition = filler_trials(
        jobs,
        assets,
        count=filler_recognition_count,
    )

    forms = []
    for form in design["session_forms"]:
        session1_analysis = []
        for row in form["session1"]["analysis_encoding_targets"]:
            session1_analysis.append(
                {
                    "trial_id": f"{row['target_id']}_encoding",
                    "target_id": row["target_id"],
                    "arm_id": row["arm_id"],
                    "analysis_group": row["analysis_group"],
                    "video_url": video_ref(str(row["old_video_path"]), assets),
                }
            )

        session2_analysis = []
        for row in form["session2"]["analysis_recognition_trials"]:
            lure_path = str(jobs[str(row["lure_id"])]["output_video"]["path"])
            session2_analysis.append(
                {
                    "trial_id": row["trial_id"],
                    "target_id": row["target_id"],
                    "arm_id": row["arm_id"],
                    "analysis_group": row["analysis_group"],
                    "old_video_url": video_ref(str(row["old_video_path"]), assets),
                    "lure_video_url": video_ref(lure_path, assets),
                    "old_side": row["old_side"],
                    "correct_choice": row["correct_choice"],
                    "question": row["question"],
                }
            )

        forms.append(
            {
                "form_id": form["form_id"],
                "session1_trials": session1_analysis + filler_encoding_targets,
                "session2_trials": session2_analysis + filler_recognition,
            }
        )

    return {
        "schema_version": "content_pocket_recognition_launch_payload.v1",
        "study": "content_pocket_recognition_memory_20260608",
        "form_count": len(forms),
        "forms": forms,
    }


CSS = """
:root{--bg:#f6f3ed;--ink:#1f2428;--muted:#66706f;--line:#d8d1c6;--accent:#2f6d67;--danger:#8b3d32}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}.wrap{max-width:980px;margin:0 auto;padding:28px 18px 64px}.panel{border:1px solid var(--line);background:#fffdf8;padding:22px;border-radius:8px;box-shadow:0 12px 30px rgba(40,34,24,.08)}h1{margin:0 0 8px;font-size:30px;letter-spacing:0}h2{margin:0 0 8px;font-size:20px;letter-spacing:0}p{margin:0 0 12px}.muted{color:var(--muted)}label{display:block;margin:14px 0 6px;font-weight:700}input,select,textarea{width:100%;border:1px solid var(--line);border-radius:6px;background:white;color:var(--ink);font:inherit;padding:11px 12px}.actions{display:flex;gap:10px;justify-content:flex-end;margin-top:18px}.btn{border:0;border-radius:6px;padding:11px 16px;font-weight:800;background:var(--accent);color:white;cursor:pointer}.btn.secondary{background:#ebe5da;color:var(--ink)}.btn:disabled{opacity:.45;cursor:not-allowed}.hidden{display:none}.bar{height:8px;background:#e1dbd1;border-radius:999px;overflow:hidden;margin:12px 0}.fill{height:100%;width:0;background:var(--accent)}video{width:100%;aspect-ratio:16/9;display:block;background:#111;border-radius:6px;object-fit:contain}.rating{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px}.rating button,.choice{border:1px solid var(--line);background:#fff;border-radius:6px;padding:10px;font-weight:750;cursor:pointer}.rating button.selected,.choice.selected{border-color:var(--accent);background:var(--accent);color:white}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}.json{height:220px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}.warn{color:var(--danger);font-weight:700}@media(max-width:760px){.grid{grid-template-columns:1fr}.panel{padding:18px}}
""".strip()


def shared_js(payload: dict[str, Any], session_number: int) -> str:
    return f"""
const PAYLOAD = {json.dumps(payload, separators=(",", ":"))};
const SESSION_NUMBER = {session_number};
let form = null;
let trials = [];
let index = 0;
let responses = [];
let currentStartedAt = null;
let startedAt = null;

function queryParam(name) {{
  const params = new URLSearchParams(window.location.search);
  return params.get(name) || params.get(name.toUpperCase()) || "";
}}
function hashString(s) {{
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {{
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }}
  return h >>> 0;
}}
function mulberry32(a) {{
  return function() {{
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }}
}}
function shuffle(items, seed) {{
  const rng = mulberry32(seed);
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {{
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }}
  return out;
}}
function participantId() {{
  return document.getElementById("participant-id").value.trim() || "anonymous";
}}
function assignForm(pid) {{
  return PAYLOAD.forms[hashString(pid) % PAYLOAD.forms.length];
}}
function endpoint() {{
  return queryParam("submit_url") || queryParam("endpoint") || "";
}}
function completionCode() {{
  return SESSION_NUMBER === 1 ? "CPR_SESSION1_DONE" : "CPR_SESSION2_DONE";
}}
function basePayload() {{
  return {{
    schema_version: "content_pocket_recognition_response.v1",
    study: PAYLOAD.study,
    session_number: SESSION_NUMBER,
    participant_id: participantId(),
    prolific_pid: queryParam("PROLIFIC_PID"),
    prolific_study_id: queryParam("STUDY_ID"),
    prolific_session_id: queryParam("SESSION_ID"),
    form_id: form.form_id,
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    n_trials: trials.length,
    responses
  }};
}}
async function submitPayload(payload) {{
  const text = JSON.stringify(payload, null, 2);
  document.getElementById("response-json").value = text;
  if (!endpoint()) return;
  const response = await fetch(endpoint(), {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: text
  }});
  if (!response.ok) throw new Error(`submit failed: ${{response.status}}`);
}}
function finish() {{
  document.getElementById("trial").classList.add("hidden");
  document.getElementById("done").classList.remove("hidden");
  document.getElementById("completion-code").textContent = completionCode();
  submitPayload(basePayload()).catch(error => {{
    document.getElementById("submit-warning").textContent = error.message;
  }});
}}
function downloadJSON() {{
  const blob = new Blob([document.getElementById("response-json").value], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${{PAYLOAD.study}}_session${{SESSION_NUMBER}}_${{participantId()}}.json`;
  a.click();
  URL.revokeObjectURL(url);
}}
document.getElementById("participant-id").value = queryParam("PROLIFIC_PID") || queryParam("participant_id");
""".strip()


def render_session1(payload: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Recognition Study Session 1</title><style>{CSS}</style></head>
<body><main class="wrap">
<section id="intro" class="panel">
<h1>Video Viewing Study</h1>
<p class="muted">Watch each short clip fully, then rate how visually clear it was. Please complete this in one sitting.</p>
<label for="participant-id">Prolific ID</label><input id="participant-id" autocomplete="off">
<div class="actions"><button class="btn" onclick="start()">Start</button></div>
</section>
<section id="trial" class="panel hidden">
<h2>Clip <span id="trial-index">1</span> / <span id="trial-total">30</span></h2><div class="bar"><div id="progress" class="fill"></div></div>
<p class="muted">Watch the full clip before rating.</p>
<video id="clip" controls playsinline preload="metadata"></video>
<label>How visually clear was this clip?</label>
<div class="rating" id="rating-buttons"></div>
<div class="actions"><button class="btn secondary" onclick="replay()">Replay</button><button id="next" class="btn" onclick="next()" disabled>Next</button></div>
</section>
<section id="done" class="panel hidden">
<h1>Session 1 Complete</h1>
<p>Completion code: <strong id="completion-code"></strong></p>
<p id="submit-warning" class="warn"></p>
<textarea id="response-json" class="json" readonly></textarea>
<div class="actions"><button class="btn" onclick="downloadJSON()">Download JSON</button></div>
</section></main>
<script>{shared_js(payload, 1)}
let videoEnded = false;
let videoError = false;
let rating = null;
function start() {{
  form = assignForm(participantId());
  trials = shuffle(form.session1_trials, hashString(participantId() + ":session1"));
  startedAt = new Date().toISOString();
  document.getElementById("trial-total").textContent = String(trials.length);
  document.getElementById("intro").classList.add("hidden");
  document.getElementById("trial").classList.remove("hidden");
  render();
}}
function renderRatingButtons() {{
  const root = document.getElementById("rating-buttons");
  root.innerHTML = "";
  for (let value = 1; value <= 5; value++) {{
    const button = document.createElement("button");
    button.textContent = String(value);
    button.disabled = !videoEnded;
    button.className = rating === value ? "selected" : "";
    button.onclick = () => {{ rating = value; renderRatingButtons(); document.getElementById("next").disabled = false; }};
    root.appendChild(button);
  }}
}}
function render() {{
  const trial = trials[index];
  videoEnded = false; videoError = false; rating = null; currentStartedAt = new Date().toISOString();
  document.getElementById("next").disabled = true;
  document.getElementById("trial-index").textContent = String(index + 1);
  document.getElementById("progress").style.width = `${{(index / trials.length) * 100}}%`;
  const clip = document.getElementById("clip");
  clip.src = trial.video_url;
  clip.onended = () => {{ videoEnded = true; renderRatingButtons(); }};
  clip.onerror = () => {{ videoError = true; videoEnded = true; renderRatingButtons(); }};
  clip.load();
  renderRatingButtons();
}}
function replay() {{ const clip = document.getElementById("clip"); clip.currentTime = 0; clip.play(); }}
function next() {{
  const trial = trials[index];
  responses.push({{...trial, cover_task_rating: rating, exposure_completed: videoEnded, media_error: videoError, started_at: currentStartedAt, completed_at: new Date().toISOString()}});
  index += 1;
  if (index >= trials.length) finish(); else render();
}}
</script></body></html>
"""


def render_session2(payload: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Recognition Study Session 2</title><style>{CSS}</style></head>
<body><main class="wrap">
<section id="intro" class="panel">
<h1>Video Recognition Study</h1>
<p class="muted">For each pair, choose the clip you saw in Session 1. Please watch both clips before choosing.</p>
<label for="participant-id">Prolific ID</label><input id="participant-id" autocomplete="off">
<div class="actions"><button class="btn" onclick="start()">Start</button></div>
</section>
<section id="trial" class="panel hidden">
<h2>Pair <span id="trial-index">1</span> / <span id="trial-total">25</span></h2><div class="bar"><div id="progress" class="fill"></div></div>
<p class="muted">Which clip did you see in Session 1?</p>
<div class="grid">
<div><video id="left-video" controls playsinline preload="metadata"></video><button id="left-choice" class="choice" onclick="choose('left')" disabled>Choose left</button></div>
<div><video id="right-video" controls playsinline preload="metadata"></video><button id="right-choice" class="choice" onclick="choose('right')" disabled>Choose right</button></div>
</div>
<div class="actions"><button class="btn secondary" onclick="replay()">Replay both</button><button id="next" class="btn" onclick="next()" disabled>Next</button></div>
</section>
<section id="done" class="panel hidden">
<h1>Session 2 Complete</h1>
<p>Completion code: <strong id="completion-code"></strong></p>
<p id="submit-warning" class="warn"></p>
<textarea id="response-json" class="json" readonly></textarea>
<div class="actions"><button class="btn" onclick="downloadJSON()">Download JSON</button></div>
</section></main>
<script>{shared_js(payload, 2)}
let ended = {{left:false,right:false}};
let mediaError = {{left:false,right:false}};
let choice = null;
function start() {{
  form = assignForm(participantId());
  trials = shuffle(form.session2_trials, hashString(participantId() + ":session2"));
  startedAt = new Date().toISOString();
  document.getElementById("trial-total").textContent = String(trials.length);
  document.getElementById("intro").classList.add("hidden");
  document.getElementById("trial").classList.remove("hidden");
  render();
}}
function setChoiceEnabled() {{
  const ready = ended.left && ended.right;
  document.getElementById("left-choice").disabled = !ready;
  document.getElementById("right-choice").disabled = !ready;
}}
function sideUrl(trial, side) {{
  if (trial.old_side === side) return trial.old_video_url;
  return trial.lure_video_url;
}}
function render() {{
  const trial = trials[index];
  ended = {{left:false,right:false}}; mediaError = {{left:false,right:false}}; choice = null; currentStartedAt = new Date().toISOString();
  document.getElementById("next").disabled = true;
  document.getElementById("left-choice").classList.remove("selected");
  document.getElementById("right-choice").classList.remove("selected");
  document.getElementById("trial-index").textContent = String(index + 1);
  document.getElementById("progress").style.width = `${{(index / trials.length) * 100}}%`;
  const left = document.getElementById("left-video");
  const right = document.getElementById("right-video");
  left.src = sideUrl(trial, "left"); right.src = sideUrl(trial, "right");
  left.onended = () => {{ ended.left = true; setChoiceEnabled(); }};
  right.onended = () => {{ ended.right = true; setChoiceEnabled(); }};
  left.onerror = () => {{ mediaError.left = true; ended.left = true; setChoiceEnabled(); }};
  right.onerror = () => {{ mediaError.right = true; ended.right = true; setChoiceEnabled(); }};
  left.load(); right.load(); setChoiceEnabled();
}}
function replay() {{
  const left = document.getElementById("left-video"); const right = document.getElementById("right-video");
  left.currentTime = 0; right.currentTime = 0; left.play(); right.play();
}}
function choose(side) {{
  choice = side;
  document.getElementById("left-choice").classList.toggle("selected", side === "left");
  document.getElementById("right-choice").classList.toggle("selected", side === "right");
  document.getElementById("next").disabled = false;
}}
function next() {{
  const trial = trials[index];
  responses.push({{...trial, choice_side: choice, is_correct: choice === trial.correct_choice, left_media_error: mediaError.left, right_media_error: mediaError.right, any_media_error: mediaError.left || mediaError.right, response_time_ms: Date.now() - Date.parse(currentStartedAt), started_at: currentStartedAt, completed_at: new Date().toISOString()}});
  index += 1;
  if (index >= trials.length) finish(); else render();
}}
</script></body></html>
"""


def render_index(*, base_url: str) -> str:
    canonical = normalize_base_url(base_url) if base_url else ""
    prefix = f"<p class=\"muted\">Stable base URL: {canonical}</p>" if canonical else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Content Pocket Recognition Study</title><style>{CSS}</style></head>
<body><main class="wrap"><section class="panel"><h1>Content Pocket Recognition Study</h1>{prefix}
<p>Use these two pages for the delayed-recognition Prolific setup.</p>
<p><a href="session1.html">Session 1 encoding task</a></p>
<p><a href="session2.html">Session 2 recognition task</a></p>
<p class="muted">No human-memory evidence exists until responses are collected and analyzed.</p>
</section></main></body></html>
"""


def build_url_map(
    *,
    assets: dict[str, dict[str, Any]],
    video_screening: dict[str, Any],
    site_base_url: str,
) -> dict[str, Any]:
    accepted = video_screening.get("accepted_for_hosting") is True
    return {
        "schema_version": "content_pocket_recognition_hosted_video_url_map.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "prepared_for": [
            "content_pocket_recognition_session1_prolific_20260608.html",
            "content_pocket_recognition_session2_prolific_20260608.html",
        ],
        "site_base_url": normalize_base_url(site_base_url) if site_base_url else "",
        "accepted_video_screening": accepted,
        "videos": [
            {
                "local_path": local_path,
                "asset_path": item["asset_path"],
                "hosted_url": item["hosted_url"],
                "screened": accepted,
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
            for local_path, item in sorted(assets.items())
        ],
        "instructions": [
            "Use session1_url and session2_url from the launch asset manifest as the Prolific study URLs.",
            "Pass ?submit_url=<endpoint> if using automatic response POST collection.",
            "Do not treat this URL map as human-memory evidence.",
        ],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Content-Pocket Recognition Launch Assets",
            "",
            f"Date: {manifest['created_at_utc']}",
            "",
            "## Status",
            "",
            f"- Status: `{manifest['status']}`",
            f"- Required videos: {manifest['counts']['required_videos']}",
            f"- Missing videos: {manifest['counts']['missing_videos']}",
            f"- Session forms: {manifest['counts']['session_forms']}",
            f"- Session 1 trials per form: {manifest['counts']['session1_trials_per_form']}",
            f"- Session 2 trials per form: {manifest['counts']['session2_trials_per_form']}",
            "",
            "## Hosted Entry Points",
            "",
            f"- Session 1: {manifest['session_urls']['session1_url'] or 'pending deployment URL'}",
            f"- Session 2: {manifest['session_urls']['session2_url'] or 'pending deployment URL'}",
            "",
            "## Artifacts",
            "",
            f"- URL map: `{manifest['artifacts']['url_map']}`",
            f"- Session 1 HTML: `{manifest['artifacts']['session1_html']}`",
            f"- Session 2 HTML: `{manifest['artifacts']['session2_html']}`",
            f"- Static site directory: `{manifest['artifacts']['site_dir']}`",
            f"- Static site archive: `{manifest['artifacts']['archive']}`",
            "",
            "## Claim Boundary",
            "",
            "- These are launch-prep artifacts only.",
            "- No human recognition-memory or measured-BMD claim is made.",
            "- Response collection, participant exclusion, and analysis remain separate gates.",
            "",
        ]
    )


def make_archive(site_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(site_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(site_dir))


def write_site_files(
    *,
    site_dir: Path,
    session1_html: str,
    session2_html: str,
    index_html: str,
    payload: dict[str, Any],
    url_map: dict[str, Any],
) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "index.js").write_text(
        "\n".join(
            [
                "export default {",
                "  fetch(request, env) {",
                "    return env.ASSETS.fetch(request);",
                "  },",
                "};",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (site_dir / "index.html").write_text(index_html, encoding="utf-8")
    (site_dir / "session1.html").write_text(session1_html, encoding="utf-8")
    (site_dir / "session2.html").write_text(session2_html, encoding="utf-8")
    write_json(site_dir / "recognition_payload.json", payload)
    write_json(site_dir / "hosted_video_url_map.json", url_map)


def build_launch_assets(
    *,
    design_path: Path,
    production_manifest_path: Path,
    video_screening_path: Path,
    out_manifest: Path,
    out_md: Path,
    out_url_map: Path,
    out_session1_html: Path,
    out_session2_html: Path,
    site_dir: Path,
    archive: Path,
    base_url: str,
    video_base_url: str,
    fallback_roots: list[Path],
    copy_videos: bool,
) -> dict[str, Any]:
    if site_dir.exists():
        shutil.rmtree(site_dir)
    design = load_json(design_path)
    production_manifest = load_json(production_manifest_path)
    video_screening = load_json(video_screening_path)
    local_paths = required_video_paths(
        design=design,
        production_manifest=production_manifest,
    )
    assets, missing = build_video_assets(
        local_paths=local_paths,
        roots=source_roots(fallback_roots),
        site_dir=site_dir,
        base_url=base_url,
        video_base_url=video_base_url,
        copy_videos=copy_videos,
    )
    if missing:
        status = "missing_required_videos"
        payload = {"forms": []}
        session1_html = ""
        session2_html = ""
    else:
        status = "hosted_launch_assets_ready" if base_url else "static_launch_assets_ready"
        payload = session_payload(
            design=design,
            production_manifest=production_manifest,
            assets=assets,
        )
        session1_html = render_session1(payload)
        session2_html = render_session2(payload)
        out_session1_html.parent.mkdir(parents=True, exist_ok=True)
        out_session2_html.parent.mkdir(parents=True, exist_ok=True)
        out_session1_html.write_text(session1_html, encoding="utf-8")
        out_session2_html.write_text(session2_html, encoding="utf-8")

    url_map = build_url_map(
        assets=assets,
        video_screening=video_screening,
        site_base_url=base_url,
    )
    write_json(out_url_map, url_map)

    if not missing:
        write_site_files(
            site_dir=site_dir,
            session1_html=session1_html,
            session2_html=session2_html,
            index_html=render_index(base_url=base_url),
            payload=payload,
            url_map=url_map,
        )
        make_archive(site_dir, archive)

    forms = payload.get("forms", []) if isinstance(payload, dict) else []
    manifest = {
        "schema_version": "content_pocket_recognition_launch_assets.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "source_design": str(design_path),
        "source_production_manifest": str(production_manifest_path),
        "source_video_screening": str(video_screening_path),
        "site_base_url": normalize_base_url(base_url) if base_url else "",
        "video_base_url": normalize_base_url(video_base_url) if video_base_url else "",
        "session_urls": {
            "session1_url": f"{normalize_base_url(base_url)}/session1.html" if base_url else "",
            "session2_url": f"{normalize_base_url(base_url)}/session2.html" if base_url else "",
        },
        "counts": {
            "required_videos": len(local_paths),
            "missing_videos": len(missing),
            "copied_videos": len(assets),
            "session_forms": len(forms),
            "session1_trials_per_form": len(forms[0]["session1_trials"]) if forms else 0,
            "session2_trials_per_form": len(forms[0]["session2_trials"]) if forms else 0,
        },
        "video_roles": dict(
            sorted(
                Counter(
                    "recognition_generated"
                    if path.startswith("data/generated/content_pocket_recognition_memory_20260608")
                    else "analysis_old_target"
                    for path in local_paths
                ).items()
            )
        ),
        "missing_videos": missing,
        "payload_sha256": sha256_text(json.dumps(payload, sort_keys=True)),
        "session_html_sha256": {
            "session1": sha256_text(session1_html) if session1_html else "",
            "session2": sha256_text(session2_html) if session2_html else "",
        },
        "artifacts": {
            "url_map": str(out_url_map),
            "session1_html": str(out_session1_html),
            "session2_html": str(out_session2_html),
            "site_dir": str(site_dir),
            "archive": str(archive),
        },
        "launch_blockers": [
            "Final human/IRB-facing content review is not recorded in this artifact.",
            "Prolific study configuration, completion URLs, compensation, and invite timing must be set in Prolific.",
            "Human recognition-memory validation has not run.",
        ],
    }
    write_json(out_manifest, manifest)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def parse_roots(raw: list[str]) -> list[Path]:
    if raw:
        return [Path(item) for item in raw]
    return list(DEFAULT_FALLBACK_ROOTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--production-manifest", type=Path, default=DEFAULT_PRODUCTION_MANIFEST)
    parser.add_argument("--video-screening", type=Path, default=DEFAULT_VIDEO_SCREENING)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-url-map", type=Path, default=DEFAULT_URL_MAP)
    parser.add_argument("--out-session1-html", type=Path, default=DEFAULT_SESSION1_HTML)
    parser.add_argument("--out-session2-html", type=Path, default=DEFAULT_SESSION2_HTML)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--video-base-url", default="")
    parser.add_argument("--no-copy-videos", action="store_true")
    parser.add_argument("--fallback-root", action="append", default=[])
    args = parser.parse_args()

    manifest = build_launch_assets(
        design_path=args.design,
        production_manifest_path=args.production_manifest,
        video_screening_path=args.video_screening,
        out_manifest=args.out_manifest,
        out_md=args.out_md,
        out_url_map=args.out_url_map,
        out_session1_html=args.out_session1_html,
        out_session2_html=args.out_session2_html,
        site_dir=args.site_dir,
        archive=args.archive,
        base_url=args.base_url,
        video_base_url=args.video_base_url,
        fallback_roots=parse_roots(args.fallback_root),
        copy_videos=not args.no_copy_videos,
    )
    print(f"[done] wrote {args.out_manifest}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] wrote {args.out_url_map}")
    print(f"[done] wrote {args.out_session1_html}")
    print(f"[done] wrote {args.out_session2_html}")
    print(f"[done] archive: {args.archive}")
    print(f"[done] status: {manifest['status']}")
    return 0 if manifest["status"] != "missing_required_videos" else 1


if __name__ == "__main__":
    raise SystemExit(main())
