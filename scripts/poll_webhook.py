"""Poll webhook.site for incoming Prolific responses and write them
to data/raw/prolific_responses/ as one JSON per rater.

Runs every 5 minutes. Idempotent — uses webhook.site request IDs as
filenames, so re-poll doesn't double-count.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TOKEN = os.environ.get("WEBHOOK_SITE_TOKEN", "")
if not TOKEN:
    raise RuntimeError("Set WEBHOOK_SITE_TOKEN before polling webhook.site.")
API = f"https://webhook.site/token/{TOKEN}/requests"
OUT = Path("data/raw/prolific_responses")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_page(page: int = 1, per_page: int = 50) -> dict:
    url = f"{API}?page={page}&per_page={per_page}&sorting=newest"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def parse_request(req: dict) -> dict | None:
    """Extract the JSON body posted by the survey, or skip non-JSON requests."""
    body = req.get("content", "")
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    # Must look like a survey response
    if "responses" not in payload or "prolific_id" not in payload:
        return None
    return payload


def poll_once() -> dict:
    """Fetch all pages, save any new responses, return summary."""
    new = 0
    skipped = 0
    total_pages = 1
    page = 1
    while page <= total_pages:
        try:
            data = fetch_page(page)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  [error] fetch failed: {exc}", flush=True)
            return {"new": new, "error": str(exc)}
        total_pages = data.get("last_page", 1)
        for req in data.get("data", []):
            uuid = req.get("uuid", "")
            if not uuid:
                continue
            out_path = OUT / f"{uuid}.json"
            if out_path.exists():
                skipped += 1
                continue
            parsed = parse_request(req)
            if parsed is None:
                continue
            out_path.write_text(json.dumps(parsed, indent=2))
            new += 1
            pid = parsed.get("prolific_id", "?")
            n_resp = len(parsed.get("responses", []))
            print(f"  [+] {uuid[:8]} prolific_id={pid[:8]} n_responses={n_resp}", flush=True)
        page += 1
    return {"new": new, "skipped": skipped}


def main() -> None:
    print(f"[poller] webhook = {API}", flush=True)
    print(f"[poller] saving to = {OUT}", flush=True)
    print("[poller] starting; polling every 5 min", flush=True)
    n_polls = 0
    while True:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{ts}] poll #{n_polls+1}...", flush=True)
        result = poll_once()
        n_total = len(list(OUT.glob("*.json")))
        print(f"  {result.get('new', 0)} new, {result.get('skipped', 0)} cached, "
              f"{n_total} total responses on disk", flush=True)
        n_polls += 1
        # Stop after 24 h max (~288 polls)
        if n_polls >= 288:
            print("[poller] 24h reached; exiting", flush=True)
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
