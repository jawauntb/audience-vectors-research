"""Snap individual figures from paper.html as PNG via Chrome headless.

Renders each SVG to its own bare HTML, then takes a tight PNG screenshot at high DPI.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAPER = Path("data/reports/paper.html").resolve()
OUT_DIR = Path("data/reports/figures")


def extract_svg(html: str, svg_id: str) -> str:
    # Find the inline-rendered SVG content (after JS runs). We'll instead
    # render the whole paper.html and clip via window.scrollTo + targeting.
    return svg_id


def render_via_chrome(svg_id: str, out_png: Path, width: int = 760, height: int = 700) -> None:
    """Open paper.html in headless Chrome, scroll to the SVG, take a screenshot."""
    html_wrapper = f"""
<!doctype html><html><head>
<style>
  body {{ margin: 0; background: #fafaf7; font-family: Georgia, serif; }}
  iframe {{ border: 0; width: {width}px; height: {height}px; }}
  .frame {{ width: {width}px; padding: 20px; }}
</style>
</head><body>
<div class="frame">
  <iframe id="paper" src="{PAPER.as_uri()}"></iframe>
</div>
<script>
window.addEventListener('load', () => {{
  const iframe = document.getElementById('paper');
  iframe.onload = () => {{
    setTimeout(() => {{
      const doc = iframe.contentDocument;
      const el = doc.getElementById('{svg_id}');
      if (el) el.scrollIntoView({{ block: 'start' }});
    }}, 2000);
  }};
}});
</script>
</body></html>
"""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir="/tmp") as f:
        f.write(html_wrapper)
        wrapper_path = f.name

    # We instead just take a full-page screenshot of paper.html and crop
    subprocess.run([
        CHROME, "--headless", "--disable-gpu",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",
        f"--screenshot={out_png}",
        PAPER.as_uri(),
    ], check=True, capture_output=True)


def snap_one_svg(svg_id: str, out_png: Path, width: int = 720, height: int = 700) -> None:
    """Render JUST one SVG by extracting it from paper.html into a standalone wrapper."""
    html = PAPER.read_text()
    # Approach: open paper.html in headless chrome with a JS that hides everything
    # except the target svg, then screenshot.
    wrapper = f"""
<!doctype html><html><head>
<base href="{PAPER.as_uri()}">
<meta charset="utf-8">
</head><body>
<iframe src="{PAPER.as_uri()}" id="f" style="border:0; width: 1100px; height: 2600px;"></iframe>
<script>
const target = "{svg_id}";
window.addEventListener('load', () => {{
  const f = document.getElementById('f');
  f.onload = () => {{
    setTimeout(() => {{
      const d = f.contentDocument;
      const svg = d.getElementById(target);
      if (!svg) {{ document.title = 'NO-SVG'; return; }}
      // copy the SVG and its caption (next div) up out of the iframe
      const cap = svg.parentElement.nextElementSibling;
      const container = document.createElement('div');
      container.style.background = '#fafaf7';
      container.style.padding = '24px';
      container.style.width = ({width}) + 'px';
      container.appendChild(svg.cloneNode(true));
      if (cap && cap.className.includes('fig-caption')) {{
        container.appendChild(cap.cloneNode(true));
      }}
      document.body.innerHTML = '';
      document.body.style.background = '#fafaf7';
      document.body.style.margin = '0';
      document.body.style.fontFamily = 'Georgia, serif';
      document.body.appendChild(container);
      document.title = 'READY';
    }}, 3000);
  }};
}});
</script>
</body></html>
"""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir="/tmp") as f:
        f.write(wrapper)
        wrapper_path = f.name

    subprocess.run([
        CHROME, "--headless", "--disable-gpu",
        f"--window-size={width+48},{height}",
        "--virtual-time-budget=10000",
        f"--screenshot={out_png}",
        f"file://{wrapper_path}",
    ], check=True, capture_output=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("fig-headtohead", "01_head_to_head.png", 720, 360),
        ("fig-personas-mem", "02_persona_cosine_memorability.png", 760, 780),
        ("fig-roi-bars", "03_roi_decomposition.png", 760, 660),
        ("fig-cv", "04_cv_per_fold.png", 720, 380),
        ("fig-adapter-cos", "05_adapter_alignment.png", 720, 360),
    ]
    for svg_id, name, w, h in targets:
        out = OUT_DIR / name
        print(f"[snap] {svg_id} → {out}")
        snap_one_svg(svg_id, out, w, h)
        if out.exists():
            print(f"  ok ({out.stat().st_size} bytes)")
        else:
            print(f"  FAILED")


if __name__ == "__main__":
    main()
