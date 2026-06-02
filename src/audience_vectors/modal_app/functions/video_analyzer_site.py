"""Modal-hosted upload UI for TRIBE video scoring.

MVP scope:
  - upload one or more short videos from a browser
  - segment each upload into TRIBE-safe windows
  - run BMD memorability, TRIBE engagement dimensions, and legacy audience axes
  - compute palette / visual-density proxies
  - return natural-language commentary about score spikes and lulls

This intentionally does not update the paper. It is an exploratory product
surface around the research artifacts.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError

import modal
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from audience_vectors.modal_app.app import app, env_secrets
from audience_vectors.modal_app.image_factory import base_image

BMD_VIDEOS_VOLUME_NAME = "bmd-videos-v1"
ANALYZER_RUNS_VOLUME_NAME = "audience-analyzer-runs-v1"
ANALYZER_RUNS_MOUNT = Path("/analyzer-runs")
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_VIDEO_SECONDS = 6 * 60
MAX_TEXT_CHARS = 12_000
TRIBE_MAX_SECONDS = 30.0
DEFAULT_SEGMENT_SECONDS = 10.0
VISUAL_SAMPLE_FPS = 2.0
STATIC_STIMULUS_SECONDS = 6.0
TEXT_NATIVE_TIMEOUT_SECONDS = 45.0
PUBLIC_SITE_URL = "https://jawaun--video-analyzer.modal.run"

TRIBE_TOTAL_VERTICES = 20_000
TRIBE_VISUAL_RANGE = (0, 4_000)
TRIBE_LANGUAGE_RANGE = (4_000, 8_000)
TRIBE_EMOTION_RANGE = (2_000, 5_000)
TRIBE_MEMORY_RANGE = (5_000, 7_000)
TRIBE_ATTENTION_RANGE_A = (14_000, 20_000)
TRIBE_ATTENTION_RANGE_B = (12_000, 16_000)
TRIBE_VIDEO_REGIONS = (
    ("attention", *TRIBE_ATTENTION_RANGE_A),
    ("emotion", *TRIBE_EMOTION_RANGE),
    ("memory", *TRIBE_MEMORY_RANGE),
    ("visual", *TRIBE_VISUAL_RANGE),
    ("language", *TRIBE_LANGUAGE_RANGE),
)
TRIBE_COGNITIVE_EASE_DISPERSION_K = 2.0
TRIBE_COGNITIVE_EASE_LANGUAGE_CEILING = 75.0
TRIBE_FRAME_STEP_SECONDS = 0.5
TRIBE_HOOK_SECONDS = 2.0
TRIBE_MAX_TIMELINE_POINTS = 30
TRIBE_MAX_PEAK_MOMENTS = 3
TRIBE_PEAK_WINDOW_FRAMES = 3
AFFECT_CLASSES = ("happy", "anger", "fear", "sadness", "disgust", "neutral")
YTDLP_COOKIES_PATH_ENV = "YTDLP_COOKIES_PATH"
YTDLP_COOKIES_TEXT_ENV = "YTDLP_COOKIES_TEXT"
YTDLP_COOKIES_B64_ENV = "YTDLP_COOKIES_B64"
YTDLP_EXTRACTOR_ARGS_ENV = "YTDLP_EXTRACTOR_ARGS"

_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm"})
_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)
_TEXT_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}
)
_VIDEO_PAGE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "vimeo.com",
        "www.vimeo.com",
        "tiktok.com",
        "www.tiktok.com",
        "vm.tiktok.com",
        "instagram.com",
        "www.instagram.com",
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
    }
)

bmd_videos_volume = modal.Volume.from_name(
    BMD_VIDEOS_VOLUME_NAME, create_if_missing=True
)
analyzer_runs_volume = modal.Volume.from_name(
    ANALYZER_RUNS_VOLUME_NAME, create_if_missing=True
)


def _favicon_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="12" dy="14" stdDeviation="18" flood-color="#52422a" flood-opacity=".22"/>
      <feDropShadow dx="-10" dy="-10" stdDeviation="14" flood-color="#fff" flood-opacity=".72"/>
    </filter>
  </defs>
  <rect width="512" height="512" rx="120" fill="#ece5d5"/>
  <circle cx="256" cy="256" r="172" fill="#f5efe3" stroke="#d9cfb8" stroke-width="10" filter="url(#s)"/>
  <circle cx="256" cy="256" r="112" fill="none" stroke="#7a6e5c" stroke-width="24"/>
  <path d="M256 120 292 220 392 256 292 292 256 392 220 292 120 256 220 220Z" fill="#f15539"/>
  <circle cx="256" cy="256" r="22" fill="#1a1410"/>
</svg>"""


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _png_response_bytes(image: Any) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _icon_png(size: int = 512) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "#ece5d5")
    draw = ImageDraw.Draw(img)
    scale = size / 512

    def xy(vals: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(round(v * scale) for v in vals)

    draw.rounded_rectangle(
        xy((36, 36, 476, 476)), radius=round(116 * scale), fill="#f5efe3"
    )
    draw.ellipse(
        xy((92, 92, 420, 420)),
        fill="#fbf7ed",
        outline="#d9cfb8",
        width=round(9 * scale),
    )
    draw.ellipse(xy((146, 146, 366, 366)), outline="#7a6e5c", width=round(22 * scale))
    draw.polygon(
        [
            xy((256, 120)),
            xy((292, 220)),
            xy((392, 256)),
            xy((292, 292)),
            xy((256, 392)),
            xy((220, 292)),
            xy((120, 256)),
            xy((220, 220)),
        ],
        fill="#f15539",
    )
    draw.ellipse(xy((236, 236, 276, 276)), fill="#1a1410")
    return _png_response_bytes(img)


def _favicon_ico() -> bytes:
    from PIL import Image

    png = io.BytesIO(_icon_png(256))
    img = Image.open(png)
    buf = io.BytesIO()
    img.save(buf, format="ICO", sizes=[(16, 16), (32, 32), (64, 64), (128, 128)])
    return buf.getvalue()


def _og_image_png() -> bytes:
    from PIL import Image, ImageDraw

    width, height = 1200, 630
    img = Image.new("RGB", (width, height), "#ece5d5")
    draw = ImageDraw.Draw(img)
    hair = "#d9cfb8"
    ink = "#1a1410"
    muted = "#7a6e5c"
    coral = "#f15539"
    signal = "#176f68"
    gold = "#8a6b3e"

    for x in range(0, width, 42):
        draw.line((x, 0, x, height), fill="#e2d9c5")
    for y in range(0, height, 42):
        draw.line((0, y, width, y), fill="#e2d9c5")

    draw.rectangle((54, 54, 1146, 576), fill="#f5efe3", outline=hair, width=2)
    draw.rectangle((88, 92, 690, 538), fill="#fbf7ed", outline=hair, width=2)
    draw.text(
        (124, 132), "MEMORABILITY LAB", fill=muted, font=_font(21, bold=True), spacing=8
    )
    draw.text((124, 190), "Audience Vector", fill=ink, font=_font(58, bold=True))
    draw.text((128, 262), "Media Analyzer", fill=ink, font=_font(58, bold=True))
    draw.text(
        (128, 372),
        "Upload files, paste URLs, or add copy.\nScore memory + attention.\nInspect density and natural-language reads.",
        fill="#3a322a",
        font=_font(28),
        spacing=10,
    )
    draw.text(
        (128, 486),
        "TRIBE v2  /  video + image + text  /  JSON readout",
        fill=muted,
        font=_font(22, bold=True),
    )

    x0, y0 = 730, 92
    tile_w, tile_h, gap = 176, 190, 16
    tiles = [
        ((x0, y0, x0 + tile_w, y0 + tile_h), "#7c6f52", "EDGE"),
        (
            (x0 + tile_w + gap, y0, x0 + 2 * tile_w + gap, y0 + tile_h),
            "#b89766",
            "HOOK",
        ),
        (
            (x0, y0 + tile_h + gap, x0 + tile_w, y0 + 2 * tile_h + gap),
            "#221a12",
            "SIGNAL",
        ),
        (
            (
                x0 + tile_w + gap,
                y0 + tile_h + gap,
                x0 + 2 * tile_w + gap,
                y0 + 2 * tile_h + gap,
            ),
            "#e4dcc8",
            "PALETTE",
        ),
    ]
    for box, color, label in tiles:
        draw.rectangle(box, fill=color)
        draw.text(
            (box[0] + 18, box[3] - 38), label, fill="#fff", font=_font(17, bold=True)
        )
    for i in range(0, 18):
        draw.line(
            (x0 + i * 22, y0, x0 + i * 22 + 230, y0 + tile_h), fill="#24333b", width=8
        )
        draw.line(
            (x0 + i * 22, y0 + tile_h + gap, x0 + i * 22 + 230, y0 + 2 * tile_h + gap),
            fill=signal,
            width=6,
        )
    draw.ellipse((946, 166, 1034, 254), fill=coral)
    for row in range(3):
        for col in range(4):
            cx = 932 + col * 46
            cy = 348 + row * 46
            draw.ellipse(
                (cx, cy, cx + 18, cy + 18), fill=coral if (row + col) % 2 == 0 else gold
            )

    draw.ellipse((940, 428, 1098, 586), fill="#fbf7ed", outline=hair, width=4)
    draw.ellipse((976, 464, 1062, 550), outline=muted, width=11)
    draw.polygon(
        [
            (1019, 442),
            (1038, 502),
            (1096, 521),
            (1038, 540),
            (1019, 598),
            (1000, 540),
            (942, 521),
            (1000, 502),
        ],
        fill=coral,
    )
    return _png_response_bytes(img)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _ffprobe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _classify_media(path: Path, content_type: str | None = None) -> str:
    media_type = (content_type or "").split(";")[0].strip().lower()
    suffix = path.suffix.lower()
    if media_type.startswith("video/") or suffix in _VIDEO_SUFFIXES:
        return "video"
    if media_type.startswith("image/") or suffix in _IMAGE_SUFFIXES:
        return "image"
    if (
        media_type.startswith("text/")
        or media_type in {"application/json", "application/xml"}
        or suffix in _TEXT_SUFFIXES
    ):
        return "text"
    return "video"


def _extension_from_content_type(content_type: str | None, fallback: str) -> str:
    media_type = (content_type or "").split(";")[0].strip().lower()
    if media_type == "text/html":
        return ".html"
    if media_type == "text/plain":
        return ".txt"
    guessed = mimetypes.guess_extension(media_type) if media_type else None
    return guessed or fallback


def _peek_content_type(url: str) -> str | None:
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "audience-vectors/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return response.headers.get("Content-Type")
    except (HTTPError, TimeoutError, URLError, OSError, ValueError):
        return None


def _stream_url_to_file(url: str, out_dir: Path) -> tuple[Path, str, str | None]:
    parsed = urllib.parse.urlparse(url)
    req = urllib.request.Request(url, headers={"User-Agent": "audience-vectors/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        content_type = response.headers.get("Content-Type")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_UPLOAD_BYTES:
            raise ValueError("Downloaded media exceeds the 250 MB MVP limit.")
        suffix = Path(parsed.path).suffix.lower()
        suffix = (
            suffix if suffix else _extension_from_content_type(content_type, ".bin")
        )
        local_path = out_dir / f"source{suffix}"
        written = 0
        with local_path.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError("Downloaded media exceeds the 250 MB MVP limit.")
                out.write(chunk)
    return local_path, parsed.netloc, content_type


def _is_video_page_url(parsed: urllib.parse.ParseResult) -> bool:
    host = parsed.netloc.lower().removeprefix("www.")
    return parsed.netloc.lower() in _VIDEO_PAGE_HOSTS or host in _VIDEO_PAGE_HOSTS


def _yt_dlp_command() -> list[str]:
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "yt-dlp"]
    raise ValueError("Video-page URL support requires yt-dlp, but it is not available.")


def _yt_dlp_auth_args(out_dir: Path) -> list[str]:
    args: list[str] = []
    if deno := shutil.which("deno"):
        args.extend(["--js-runtimes", f"deno:{deno}"])
    elif node := shutil.which("node"):
        args.extend(["--js-runtimes", f"node:{node}"])
    elif bun := shutil.which("bun"):
        args.extend(["--js-runtimes", f"bun:{bun}"])

    cookies_path = os.environ.get(YTDLP_COOKIES_PATH_ENV)
    if cookies_path and Path(cookies_path).exists():
        args.extend(["--cookies", cookies_path])
    elif cookies_b64 := os.environ.get(YTDLP_COOKIES_B64_ENV):
        cookies_file = out_dir / "yt_cookies.txt"
        cookies_file.write_bytes(base64.b64decode(cookies_b64))
        cookies_file.chmod(0o600)
        args.extend(["--cookies", str(cookies_file)])
    elif cookies_text := os.environ.get(YTDLP_COOKIES_TEXT_ENV):
        cookies_file = out_dir / "yt_cookies.txt"
        cookies_file.write_text(cookies_text, encoding="utf-8")
        cookies_file.chmod(0o600)
        args.extend(["--cookies", str(cookies_file)])

    if extractor_args := os.environ.get(YTDLP_EXTRACTOR_ARGS_ENV):
        args.extend(["--extractor-args", extractor_args])
    return args


def _clean_ytdlp_failure(exc: subprocess.CalledProcessError) -> str:
    detail = (exc.stderr or exc.stdout or "").strip().splitlines()[-4:]
    reason = " ".join(line.strip() for line in detail if line.strip())
    if "Sign in to confirm" in reason or "not a bot" in reason:
        return (
            "YouTube blocked the cloud downloader as an unauthenticated bot. "
            "Upload the MP4 directly, or configure yt-dlp cookies/PO-token env vars "
            "on Modal for YouTube URL scoring."
        )
    return reason or "yt-dlp could not resolve the URL."


def _download_video_page_with_ytdlp(
    url: str,
    out_dir: Path,
) -> tuple[Path, str]:
    output_template = str(out_dir / "source.%(ext)s")
    section_end = _timestamp(MAX_VIDEO_SECONDS)
    try:
        result = subprocess.run(
            [
                *_yt_dlp_command(),
                "--no-playlist",
                "-f",
                "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
                "--merge-output-format",
                "mp4",
                "--download-sections",
                f"*0:00-{section_end}",
                "--force-keyframes-at-cuts",
                "--max-filesize",
                f"{MAX_UPLOAD_BYTES}",
                *_yt_dlp_auth_args(out_dir),
                "--print",
                "before_dl:TITLE:%(title).160B",
                "--print",
                "after_move:FILE:%(filepath)s",
                "-o",
                output_template,
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Could not resolve that video page as media: {_clean_ytdlp_failure(exc)}"
        ) from exc

    printed_paths: list[Path] = []
    title = urllib.parse.urlparse(url).netloc
    for line in result.stdout.splitlines():
        if line.startswith("TITLE:"):
            title = line.removeprefix("TITLE:").strip() or title
        elif line.startswith("FILE:"):
            printed_paths.append(Path(line.removeprefix("FILE:").strip()))

    candidates = printed_paths + sorted(out_dir.glob("source.*"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            if candidate.stat().st_size > MAX_UPLOAD_BYTES:
                raise ValueError("Downloaded media exceeds the 250 MB MVP limit.")
            return candidate, title
    raise ValueError("Could not download a playable media file from that video page.")


def _download_url_to_source(url: str, out_dir: Path) -> tuple[Path, str, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) media URLs are supported.")
    if not parsed.netloc:
        raise ValueError("URL is missing a host.")

    suffix = Path(parsed.path).suffix.lower()
    is_video_page = _is_video_page_url(parsed)
    if suffix in _IMAGE_SUFFIXES or suffix in _TEXT_SUFFIXES:
        source_path, host, content_type = _stream_url_to_file(url, out_dir)
        return source_path, host, _classify_media(source_path, content_type)
    content_type = _peek_content_type(url)
    direct_kind = _classify_media(Path("source"), content_type)
    if (
        not is_video_page
        and direct_kind in {"image", "text"}
        and not (content_type or "").lower().startswith("text/html")
    ):
        source_path, host, content_type = _stream_url_to_file(url, out_dir)
        return source_path, host, _classify_media(source_path, content_type)

    try:
        candidate, title = _download_video_page_with_ytdlp(url, out_dir)
    except subprocess.CalledProcessError:
        if is_video_page:
            raise
        source_path, host, content_type = _stream_url_to_file(url, out_dir)
        return source_path, host, _classify_media(source_path, content_type)
    except ValueError:
        if is_video_page:
            raise
        source_path, host, content_type = _stream_url_to_file(url, out_dir)
        return source_path, host, _classify_media(source_path, content_type)
    return candidate, title, _classify_media(candidate)


def _read_text_source(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    text = raw
    if suffix in {".html", ".htm"} or "<html" in raw[:1000].lower():
        parser = _TextExtractor()
        parser.feed(raw)
        text = parser.text()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Text input was empty after cleanup.")
    return text[:MAX_TEXT_CHARS]


def _write_text_stimulus(text: str, out_path: Path) -> None:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("Text input is empty.")
    out_path.write_text(cleaned[:MAX_TEXT_CHARS], encoding="utf-8")


def _wrap_text(
    draw: Any, text: str, font: Any, max_width: int, max_lines: int
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current.append(word)
            continue
        if current:
            lines.append(" ".join(current))
        current = [word]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def _render_text_card(text: str, out_path: Path, title: str = "Text stimulus") -> None:
    from PIL import Image, ImageDraw

    width, height = 1280, 720
    img = Image.new("RGB", (width, height), "#ece5d5")
    draw = ImageDraw.Draw(img)
    hair = "#d9cfb8"
    ink = "#1a1410"
    muted = "#7a6e5c"
    coral = "#f15539"
    draw.rectangle(
        (58, 52, width - 58, height - 52), fill="#f5efe3", outline=hair, width=2
    )
    draw.text(
        (96, 92), "AUDIENCE VECTOR TEXT INPUT", fill=muted, font=_font(22, bold=True)
    )
    draw.text((96, 142), title[:52], fill=ink, font=_font(38, bold=True))
    body_font = _font(34)
    lines = _wrap_text(draw, text, body_font, width - 192, 8)
    y = 238
    for line in lines:
        draw.text((96, y), line, fill="#3a322a", font=body_font)
        y += 48
    draw.rectangle((96, height - 112, width - 96, height - 108), fill=hair)
    draw.rectangle(
        (96, height - 112, min(width - 96, 96 + len(text) * 3), height - 108),
        fill=coral,
    )
    img.save(out_path)


def _render_image_card(image_path: Path, out_path: Path) -> None:
    from PIL import Image, ImageFilter, ImageOps

    width, height = 1280, 720
    src = Image.open(image_path).convert("RGB")
    background = ImageOps.fit(src, (width, height)).filter(ImageFilter.GaussianBlur(24))
    overlay = Image.new("RGB", (width, height), "#ece5d5")
    background = Image.blend(background, overlay, 0.28)
    foreground = ImageOps.contain(src, (width - 160, height - 112))
    x = (width - foreground.width) // 2
    y = (height - foreground.height) // 2
    background.paste(foreground, (x, y))
    background.save(out_path)


def _still_image_to_video(image_path: Path, out_path: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            "24",
            "-t",
            f"{STATIC_STIMULUS_SECONDS:.3f}",
            "-i",
            str(image_path),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )


def _timestamp(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _segment_video(
    *,
    source_path: Path,
    job_id: str,
    segment_seconds: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    duration = _ffprobe_duration(source_path)
    if duration <= 0:
        raise ValueError("Video has non-positive duration.")
    if duration > MAX_VIDEO_SECONDS:
        raise ValueError(
            f"Video is {duration:.1f}s; MVP limit is {MAX_VIDEO_SECONDS:.0f}s."
        )

    segment_seconds = max(2.0, min(segment_seconds, TRIBE_MAX_SECONDS))
    n_segments = int(math.ceil(duration / segment_seconds))
    segments: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(n_segments):
        start = idx * segment_seconds
        end = min(duration, start + segment_seconds)
        if end - start < 0.5:
            continue
        sample_id = f"upload_{job_id}_seg_{idx:04d}"
        local_path = out_dir / f"{sample_id}.mp4"
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{end - start:.3f}",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                str(local_path),
            ]
        )
        remote_volume_path = f"/video-analyzer/{job_id}/{local_path.name}"
        segments.append(
            {
                "sample_id": sample_id,
                "local_path": str(local_path),
                "modal_path": f"/bmd-videos{remote_volume_path}",
                "remote_volume_path": remote_volume_path,
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "timestamp": f"{_timestamp(start)}-{_timestamp(end)}",
            }
        )
    return segments


def _upload_segments_to_volume(segments: list[dict[str, Any]]) -> None:
    with bmd_videos_volume.batch_upload(force=True) as batch:
        for segment in segments:
            batch.put_file(Path(segment["local_path"]), segment["remote_volume_path"])


def _upload_file_to_volume(local_path: Path, remote_volume_path: str) -> str:
    with bmd_videos_volume.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_volume_path)
    return f"/bmd-videos{remote_volume_path}"


def _load_reference_axes() -> dict[str, Any]:
    axes_ref = files("audience_vectors.web").joinpath("reference_axes.npz")
    with as_file(axes_ref) as axes_path:
        payload = __import__("numpy").load(axes_path, allow_pickle=False)
        return {
            "mem_direction": payload["mem_direction"].astype("float32"),
            "mem_projection": payload["mem_reference_projection"].astype("float32"),
            "mem_mean": float(payload["mem_reference_mean"]),
            "mem_std": float(payload["mem_reference_std"]),
            "mem_n": int(payload["mem_n"]),
            "attention_direction": payload["attention_direction"].astype("float32"),
            "attention_projection": payload["attention_reference_projection"].astype(
                "float32"
            ),
            "attention_mean": float(payload["attention_reference_mean"]),
            "attention_std": float(payload["attention_reference_std"]),
            "attention_n": int(payload["attention_n"]),
        }


def _percentile(reference: Any, score: float) -> float:
    np = __import__("numpy")
    return float(
        np.searchsorted(np.sort(reference), score, side="right") / len(reference)
    )


def _round_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


def _region_mean(activations: Any, start: int, end: int) -> float:
    np = __import__("numpy")
    values = np.asarray(activations, dtype=np.float32).reshape(-1)
    total = int(values.shape[0])
    if total == 0:
        return 0.0
    scale = total / TRIBE_TOTAL_VERTICES
    s = max(0, min(int(start * scale), total))
    e = max(s, min(int(end * scale), total))
    if s >= e:
        return 0.0
    return float(np.abs(values[s:e]).mean())


def _normalize_activation(raw: float) -> float:
    if raw <= 0.0:
        return 0.0
    return _round_score(100.0 * (1.0 - math.exp(-raw / 2.0)))


def _compute_cognitive_ease(
    *, region_scores: list[float], language_score: float
) -> float:
    if not region_scores:
        return 0.0
    mean = _mean(region_scores)
    variance = _mean([(score - mean) ** 2 for score in region_scores])
    stddev = math.sqrt(variance)
    dispersion_penalty = max(
        0.0,
        min(50.0, stddev * TRIBE_COGNITIVE_EASE_DISPERSION_K),
    )
    language_penalty = max(
        0.0,
        min(25.0, max(0.0, language_score - TRIBE_COGNITIVE_EASE_LANGUAGE_CEILING)),
    )
    return _round_score(100.0 - dispersion_penalty - language_penalty)


def _engagement_recommendations(
    *,
    attention: float,
    emotion: float,
    memory: float,
    visual: float,
    language: float,
) -> list[str]:
    recs: list[str] = []
    if attention < 40.0:
        recs.append(
            "The opening is easy to miss. Make the first line/frame more specific or visually interruptive."
        )
    if emotion < 40.0:
        recs.append(
            "The idea is clear, but it does not create enough stakes, contrast, or payoff."
        )
    if memory < 40.0:
        recs.append(
            "There is not one sticky idea yet. Give the viewer a sharper phrase, number, image, or contrast."
        )
    if visual < 40.0:
        recs.append(
            "The visual field is not doing enough work. Simplify the frame and strengthen the focal subject."
        )
    if language < 40.0:
        recs.append(
            "The wording is too generic or thin. Use more concrete, semantically rich language."
        )
    if not recs:
        recs.append(
            "The TRIBE dimensions are balanced enough to interpret this as a usable signal."
        )
    return recs


def _dominant_tribe_region(activations: Any) -> str:
    best_name = "attention"
    best_score = -1.0
    for name, start, end in TRIBE_VIDEO_REGIONS:
        score = _normalize_activation(_region_mean(activations, start, end))
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def _aggregate_tribe_dimensions(activations: Any) -> dict[str, Any]:
    np = __import__("numpy")
    values = np.asarray(activations, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "overall_score": 0.0,
            "attention_score": 0.0,
            "emotion_score": 0.0,
            "memory_score": 0.0,
            "visual_score": 0.0,
            "language_score": 0.0,
            "cognitive_ease": 0.0,
            "raw_mean_activation": 0.0,
            "dominant_region": "attention",
            "recommendations": [
                "No TRIBE activation rows were returned for this stimulus."
            ],
        }

    visual = _normalize_activation(_region_mean(values, *TRIBE_VISUAL_RANGE))
    language = _normalize_activation(_region_mean(values, *TRIBE_LANGUAGE_RANGE))
    emotion = _normalize_activation(_region_mean(values, *TRIBE_EMOTION_RANGE))
    memory = _normalize_activation(_region_mean(values, *TRIBE_MEMORY_RANGE))
    attention_a = _normalize_activation(_region_mean(values, *TRIBE_ATTENTION_RANGE_A))
    attention_b = _normalize_activation(_region_mean(values, *TRIBE_ATTENTION_RANGE_B))
    attention = _round_score((attention_a + attention_b) / 2.0)
    overall = _round_score(
        attention * 0.25
        + emotion * 0.20
        + memory * 0.20
        + visual * 0.20
        + language * 0.15
    )
    cognitive_ease = _compute_cognitive_ease(
        region_scores=[attention, emotion, memory, visual, language],
        language_score=language,
    )
    return {
        "overall_score": overall,
        "attention_score": attention,
        "emotion_score": emotion,
        "memory_score": memory,
        "visual_score": visual,
        "language_score": language,
        "cognitive_ease": cognitive_ease,
        "raw_mean_activation": round(float(np.abs(values).mean()), 4),
        "dominant_region": _dominant_tribe_region(values),
        "recommendations": _engagement_recommendations(
            attention=attention,
            emotion=emotion,
            memory=memory,
            visual=visual,
            language=language,
        ),
    }


def _flatten_frames(frames: Any) -> Any:
    return frames.mean(axis=0) if getattr(frames, "ndim", 0) == 2 else frames


def _hook_score_from_frames(
    frames: Any, hook_seconds: float = TRIBE_HOOK_SECONDS
) -> float:
    if getattr(frames, "ndim", 0) != 2 or len(frames) == 0:
        return 0.0
    hook_frame_count = max(1, int(round(hook_seconds / TRIBE_FRAME_STEP_SECONDS)))
    hook_vec = _flatten_frames(frames[:hook_frame_count])
    return float(_aggregate_tribe_dimensions(hook_vec)["overall_score"])


def _timeline_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return []
    target = min(TRIBE_MAX_TIMELINE_POINTS, len(ok_rows))
    timeline: list[dict[str, Any]] = []
    for i in range(target):
        idx = int(round(i * (len(ok_rows) - 1) / max(target - 1, 1)))
        row = ok_rows[idx]
        score = row.get("tribe_scores", {})
        timeline.append(
            {
                "timestamp_seconds": round(
                    (float(row.get("start_s", 0.0)) + float(row.get("end_s", 0.0)))
                    / 2.0,
                    2,
                ),
                "overall_score": float(score.get("overall_score", 0.0)),
            }
        )
    return timeline


def _peak_moments_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ranked = sorted(
        ok_rows,
        key=lambda row: float(row.get("tribe_scores", {}).get("overall_score", 0.0)),
        reverse=True,
    )[:TRIBE_MAX_PEAK_MOMENTS]
    moments = [
        {
            "timestamp_seconds": round(
                (float(row.get("start_s", 0.0)) + float(row.get("end_s", 0.0))) / 2.0,
                2,
            ),
            "overall_score": float(
                row.get("tribe_scores", {}).get("overall_score", 0.0)
            ),
            "dominant_region": str(
                row.get("tribe_scores", {}).get("dominant_region", "attention")
            ),
        }
        for row in ranked
    ]
    return sorted(moments, key=lambda item: item["timestamp_seconds"])


async def _score_segments_with_tribe(
    segments: list[dict[str, Any]],
    *,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    np = __import__("numpy")
    from audience_vectors.services.tribe_service import TribeService

    axes = _load_reference_axes()
    service = TribeService()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def score_one(segment: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            result = await service.predict_video(segment["modal_path"])
        if result is None:
            return {**segment, "status": "tribe_failed"}
        if hasattr(result, "frames"):
            frames = np.asarray(result.frames, dtype=np.float32)
        else:
            frames = np.asarray(result["frames"], dtype=np.float32)
        vec = frames.mean(axis=0) if frames.ndim == 2 else frames

        mem_score = float(vec @ axes["mem_direction"])
        audience_axis_score = float(vec @ axes["attention_direction"])
        tribe_scores = _aggregate_tribe_dimensions(vec)
        return {
            **segment,
            "status": "ok",
            "tribe_shape": list(frames.shape),
            "tribe_scores": tribe_scores,
            "tribe_hook_score": _hook_score_from_frames(frames),
            "mem_projection": mem_score,
            "mem_z": (mem_score - axes["mem_mean"]) / axes["mem_std"],
            "mem_percentile": _percentile(axes["mem_projection"], mem_score),
            "attention_projection": audience_axis_score,
            "attention_z": (audience_axis_score - axes["attention_mean"])
            / axes["attention_std"],
            "attention_percentile": _percentile(
                axes["attention_projection"], audience_axis_score
            ),
            "legacy_persona_attention_projection": audience_axis_score,
            "legacy_persona_attention_z": (audience_axis_score - axes["attention_mean"])
            / axes["attention_std"],
            "legacy_persona_attention_percentile": _percentile(
                axes["attention_projection"], audience_axis_score
            ),
        }

    return await asyncio.gather(*(score_one(segment) for segment in segments))


def _score_vector(
    *,
    segment: dict[str, Any],
    frames: Any,
    axes: dict[str, Any],
) -> dict[str, Any]:
    vec = frames.mean(axis=0) if frames.ndim == 2 else frames
    mem_score = float(vec @ axes["mem_direction"])
    audience_axis_score = float(vec @ axes["attention_direction"])
    tribe_scores = _aggregate_tribe_dimensions(vec)
    return {
        **segment,
        "status": "ok",
        "tribe_shape": list(frames.shape),
        "tribe_scores": tribe_scores,
        "tribe_hook_score": _hook_score_from_frames(frames),
        "mem_projection": mem_score,
        "mem_z": (mem_score - axes["mem_mean"]) / axes["mem_std"],
        "mem_percentile": _percentile(axes["mem_projection"], mem_score),
        "attention_projection": audience_axis_score,
        "attention_z": (audience_axis_score - axes["attention_mean"])
        / axes["attention_std"],
        "attention_percentile": _percentile(
            axes["attention_projection"], audience_axis_score
        ),
        "legacy_persona_attention_projection": audience_axis_score,
        "legacy_persona_attention_z": (audience_axis_score - axes["attention_mean"])
        / axes["attention_std"],
        "legacy_persona_attention_percentile": _percentile(
            axes["attention_projection"], audience_axis_score
        ),
    }


async def _score_text_with_tribe(
    *,
    text_modal_path: str,
    segment: dict[str, Any],
) -> dict[str, Any] | None:
    np = __import__("numpy")
    from audience_vectors.services.tribe_service import TribeService

    axes = _load_reference_axes()
    result = await TribeService().predict_text(text_modal_path)
    if result is None:
        return None
    frames = (
        np.asarray(result.frames, dtype=np.float32)
        if hasattr(result, "frames")
        else np.asarray(result["frames"], dtype=np.float32)
    )
    duration = (
        float(result.duration_seconds)
        if hasattr(result, "duration_seconds")
        else float(result.get("duration_seconds", 0.0))
    )
    scored = _score_vector(segment=segment, frames=frames, axes=axes)
    scored["tribe_duration_seconds"] = duration
    scored["scoring_path"] = "native_text"
    return scored


def _extract_visual_frames(source_path: Path, frame_dir: Path) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vf",
            f"fps={VISUAL_SAMPLE_FPS},scale=160:-1",
            str(frame_dir / "frame_%05d.jpg"),
        ]
    )
    return sorted(frame_dir.glob("frame_*.jpg"))


def _rgb_to_hex(rgb: Any) -> str:
    np = __import__("numpy")
    vals = np.clip(np.round(rgb), 0, 255).astype(int)
    return "#" + "".join(f"{int(v):02X}" for v in vals)


def _quantized_palette(rgb: Any, top_n: int = 6) -> list[dict[str, Any]]:
    np = __import__("numpy")
    quant = ((rgb.astype(np.int32) // 32) * 32 + 16).clip(0, 255)
    flat = quant.reshape(-1, 3)
    counts: Counter[tuple[int, int, int]] = Counter(map(tuple, flat.tolist()))
    total = sum(counts.values())
    return [
        {
            "hex": _rgb_to_hex(np.asarray(color)),
            "proportion": count / total if total else 0.0,
        }
        for color, count in counts.most_common(top_n)
    ]


def _color_entropy(rgb: Any, bins: int = 8) -> float:
    np = __import__("numpy")
    quant = np.clip((rgb.astype(np.int32) * bins) // 256, 0, bins - 1)
    flat = quant[:, :, 0] * bins * bins + quant[:, :, 1] * bins + quant[:, :, 2]
    counts = np.bincount(flat.reshape(-1), minlength=bins**3).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)) / math.log2(bins**3))


def _visual_metrics(rgb: Any, prev_rgb: Any | None) -> dict[str, float]:
    np = __import__("numpy")
    from scipy import ndimage

    x = rgb.astype(np.float32) / 255.0
    maxc = x.max(axis=2)
    minc = x.min(axis=2)
    saturation = np.zeros_like(maxc)
    np.divide(maxc - minc, maxc, out=saturation, where=maxc > 1e-6)
    gray = 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]
    sx = ndimage.sobel(gray, axis=1, mode="reflect")
    sy = ndimage.sobel(gray, axis=0, mode="reflect")
    grad = np.sqrt(sx**2 + sy**2)
    edge_mask = grad > 0.18
    closed = ndimage.binary_closing(edge_mask, structure=np.ones((3, 3)))
    dilated = ndimage.binary_dilation(closed, iterations=1)
    label_result = cast(tuple[Any, Any], ndimage.label(dilated))
    labels = np.asarray(label_result[0], dtype=np.int32)
    areas = np.bincount(labels.reshape(-1))[1:]
    min_area = max(12, int(rgb.shape[0] * rgb.shape[1] * 0.001))
    max_area = int(rgb.shape[0] * rgb.shape[1] * 0.65)
    component_areas = areas[(areas >= min_area) & (areas <= max_area)]
    motion = 0.0
    if prev_rgb is not None:
        prev = prev_rgb.astype(np.float32) / 255.0
        motion = float(np.abs(x - prev).mean())
    return {
        "color_entropy": _color_entropy(rgb),
        "saturation_mean": float(saturation.mean()),
        "value_std": float(maxc.std()),
        "edge_density": float(edge_mask.mean()),
        "edge_strength": float(grad.mean()),
        "figure_count_proxy": float(len(component_areas)),
        "figure_area_fraction": float(
            component_areas.sum() / (rgb.shape[0] * rgb.shape[1])
        ),
        "luminance_contrast": float(gray.std()),
        "frame_motion_delta": motion,
    }


def _segment_for_time(
    segments: list[dict[str, Any]], time_s: float
) -> dict[str, Any] | None:
    for segment in segments:
        if segment.get("status") != "ok":
            continue
        if float(segment["start_s"]) <= time_s < float(segment["end_s"]):
            return segment
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _hex_to_rgb(color: str) -> tuple[int, int, int] | None:
    color = color.strip().lstrip("#")
    if len(color) != 6:
        return None
    try:
        return (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16),
        )
    except ValueError:
        return None


def _palette_warmth(palette: list[dict[str, Any]]) -> float:
    """Approximate warm-vs-cool color balance from palette swatches."""

    total = 0.0
    weighted = 0.0
    for item in palette:
        rgb = _hex_to_rgb(str(item.get("hex", "")))
        if rgb is None:
            continue
        r, g, b = (channel / 255.0 for channel in rgb)
        weight = float(item.get("proportion", 0.0) or 0.0)
        # Warm colors have relatively more red/yellow than blue.
        warm = _clamp01((r + 0.55 * g - 1.05 * b + 0.55) / 1.65)
        weighted += weight * warm
        total += weight
    return weighted / total if total > 1e-6 else 0.5


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    max_logit = max(logits.values()) if logits else 0.0
    exp_values = {key: math.exp(value - max_logit) for key, value in logits.items()}
    total = sum(exp_values.values()) or 1.0
    return {key: value / total for key, value in exp_values.items()}


def _affect_proxy(row: dict[str, Any]) -> dict[str, Any]:
    """NOVA-inspired affect proxy from media features, not EEG decoding."""

    tribe = row.get("tribe_scores", {})
    attention = _clamp01(float(tribe.get("attention_score", 0.0)) / 100.0)
    emotion = _clamp01(float(tribe.get("emotion_score", 0.0)) / 100.0)
    memory = _clamp01(float(tribe.get("memory_score", 0.0)) / 100.0)
    visual = _clamp01(float(tribe.get("visual_score", 0.0)) / 100.0)
    language = _clamp01(float(tribe.get("language_score", 0.0)) / 100.0)
    ease = _clamp01(float(tribe.get("cognitive_ease", 0.0)) / 100.0)
    motion = _clamp01(float(row.get("frame_motion_delta_mean", 0.0)) / 0.12)
    edge = _clamp01(float(row.get("edge_density_mean", 0.0)) / 0.45)
    density = _clamp01((float(row.get("visual_density_z", 0.0)) + 2.0) / 4.0)
    saturation = _clamp01(float(row.get("saturation_mean_mean", 0.0)))
    contrast = _clamp01(float(row.get("luminance_contrast_mean", 0.0)) / 0.28)
    warmth = _palette_warmth(list(row.get("palette", []) or []))

    arousal = _clamp01(
        0.33 * attention
        + 0.30 * emotion
        + 0.17 * motion
        + 0.12 * contrast
        + 0.08 * density
    )
    valence = _clamp01(
        0.28 * warmth
        + 0.22 * ease
        + 0.18 * visual
        + 0.13 * language
        + 0.11 * saturation
        + 0.08 * memory
        - 0.14 * edge
        - 0.10 * density
    )

    logits = {
        "happy": 1.35 * valence + 0.45 * arousal + 0.35 * warmth + 0.20 * visual,
        "anger": 1.05 * arousal + 0.55 * edge + 0.35 * density - 0.50 * valence,
        "fear": 1.15 * arousal + 0.55 * motion + 0.35 * density - 0.35 * valence,
        "sadness": 0.85 * (1.0 - arousal)
        + 0.45 * memory
        + 0.25 * language
        - 0.65 * valence,
        "disgust": 0.55 * emotion + 0.45 * density + 0.40 * edge - 0.40 * valence,
        "neutral": 0.95 * ease + 0.40 * (1.0 - emotion) + 0.30 * (1.0 - attention),
    }
    probabilities = _softmax(logits)
    top = max(probabilities.items(), key=lambda item: item[1])
    sorted_probs = sorted(probabilities.values(), reverse=True)
    confidence = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

    return {
        "label": top[0],
        "confidence": round(confidence, 4),
        "scores": {
            key: _round_score(probabilities.get(key, 0.0) * 100.0)
            for key in AFFECT_CLASSES
        },
        "arousal_proxy": _round_score(arousal * 100.0),
        "valence_proxy": _round_score(valence * 100.0),
        "palette_warmth": round(warmth, 4),
        "method": "NOVA-inspired media proxy from TRIBE dimensions plus palette/motion/density; not EEG PSD decoding.",
    }


def _mean_affect_proxy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [row.get("affect_proxy", {}) for row in rows if row.get("affect_proxy")]
    if not profiles:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "scores": {key: 0.0 for key in AFFECT_CLASSES},
            "arousal_proxy": 0.0,
            "valence_proxy": 0.0,
            "method": "No affect proxy available.",
            "note": "Affect proxy is NOVA-inspired but not EEG decoding.",
        }
    mean_scores = {
        key: _round_score(
            _mean(
                [float(profile.get("scores", {}).get(key, 0.0)) for profile in profiles]
            )
        )
        for key in AFFECT_CLASSES
    }
    top = max(mean_scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(mean_scores.values(), reverse=True)
    confidence = (
        (sorted_scores[0] - sorted_scores[1]) / 100.0 if len(sorted_scores) > 1 else 0.0
    )
    return {
        "label": top[0],
        "confidence": round(confidence, 4),
        "scores": mean_scores,
        "arousal_proxy": _round_score(
            _mean([float(profile.get("arousal_proxy", 0.0)) for profile in profiles])
        ),
        "valence_proxy": _round_score(
            _mean([float(profile.get("valence_proxy", 0.0)) for profile in profiles])
        ),
        "method": "NOVA-inspired media proxy from TRIBE dimensions plus palette/motion/density; not EEG PSD decoding.",
        "note": "Use this for creative diagnosis. Real NOVA-style validation would require EEG PSD features and emotion labels.",
    }


def _affect_read(profile: dict[str, Any]) -> str:
    label = str(profile.get("label", "neutral"))
    arousal = float(profile.get("arousal_proxy", 0.0))
    valence = float(profile.get("valence_proxy", 0.0))
    prefix = {
        "happy": "Warm positive affect",
        "anger": "High-arousal conflict affect",
        "fear": "High-arousal uncertainty affect",
        "sadness": "Low-arousal reflective affect",
        "disgust": "Aversion or friction affect",
        "neutral": "Low-specificity affect",
    }.get(label, "Mixed affect")
    return (
        f"{prefix}: proxy label {label}, arousal {arousal:.0f}/100, "
        f"valence {valence:.0f}/100. This is media-derived, not EEG."
    )


def _add_visual_analysis(
    *,
    source_path: Path,
    segments: list[dict[str, Any]],
    frame_dir: Path,
) -> list[dict[str, Any]]:
    np = __import__("numpy")
    from PIL import Image

    frame_paths = _extract_visual_frames(source_path, frame_dir)
    frame_rows: list[dict[str, Any]] = []
    prev_rgb: Any | None = None
    for idx, frame_path in enumerate(frame_paths):
        rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        row = {
            "time_s": idx / VISUAL_SAMPLE_FPS,
            "palette": _quantized_palette(rgb),
            **_visual_metrics(rgb, prev_rgb),
        }
        frame_rows.append(row)
        prev_rgb = rgb

    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame_row in frame_rows:
        segment = _segment_for_time(segments, float(frame_row["time_s"]))
        if segment is not None:
            by_segment[segment["sample_id"]].append(frame_row)

    metric_names = [
        "color_entropy",
        "saturation_mean",
        "value_std",
        "edge_density",
        "edge_strength",
        "figure_count_proxy",
        "figure_area_fraction",
        "luminance_contrast",
        "frame_motion_delta",
    ]
    scored = [segment for segment in segments if segment.get("status") == "ok"]
    raw_rows: list[dict[str, Any]] = []
    for segment in scored:
        frames = by_segment.get(segment["sample_id"], [])
        if not frames:
            raw_rows.append(segment)
            continue
        palette_counter: Counter[str] = Counter()
        for frame in frames:
            for item in frame["palette"]:
                palette_counter[item["hex"]] += item["proportion"]
        palette_total = sum(palette_counter.values())
        visual = {
            f"{name}_mean": _mean([float(frame[name]) for frame in frames])
            for name in metric_names
        }
        segment = {
            **segment,
            **visual,
            "palette": [
                {
                    "hex": color,
                    "proportion": weight / palette_total if palette_total else 0.0,
                }
                for color, weight in palette_counter.most_common(6)
            ],
        }
        raw_rows.append(segment)

    density_features = [
        "color_entropy_mean",
        "saturation_mean_mean",
        "value_std_mean",
        "edge_density_mean",
        "figure_count_proxy_mean",
        "figure_area_fraction_mean",
        "frame_motion_delta_mean",
    ]
    matrix = np.asarray(
        [
            [float(row.get(feature, 0.0)) for feature in density_features]
            for row in raw_rows
        ],
        dtype=np.float32,
    )
    if len(matrix):
        std = matrix.std(axis=0)
        z = (matrix - matrix.mean(axis=0)) / np.where(std > 1e-6, std, 1.0)
        for row, score in zip(raw_rows, z.mean(axis=1), strict=True):
            row["visual_density_z"] = float(score)
            row["natural_read"] = _natural_read(row)
            row["affect_proxy"] = _affect_proxy(row)
    return raw_rows


def _natural_text_read(mem: float, attention: float) -> str:
    if mem >= 0.85 and attention >= 0.85:
        return "Text hook: the language lands as both attention-pulling and encoding-friendly."
    if mem >= 0.85:
        return "Memorable text beat: likely distinctive wording, concrete imagery, or a crisp claim."
    if attention >= 0.85:
        return "Attention-pulling text beat: strong pull, but not necessarily a stable memory object."
    if mem < 0.35 and attention < 0.35:
        return "Flat text beat: the copy is not giving the axes much to grab onto."
    return (
        "Middle-band text beat: usable, but probably needs sharper imagery or stakes."
    )


def _natural_static_image_read(
    *,
    mem: float,
    attention: float,
    density: float,
) -> str | None:
    if mem >= 0.85 and attention >= 0.85:
        return "Static hook: the image has enough organized visual structure to score high without motion."
    if density > 0.75 and mem < 0.60:
        return "Busy static frame: visual density is present, but it may not resolve cleanly."
    return None


def _natural_video_read(
    *,
    mem: float,
    attention: float,
    density: float,
    motion: float,
    edge: float,
) -> str:
    if mem >= 0.85 and attention >= 0.85:
        return (
            "Hook zone: the segment is both high-pull and high-encoding. "
            "The visual structure is likely landing as a clean beat."
        )
    if attention >= 0.85 and mem < 0.70:
        return (
            "Attention spike: motion/change is pulling the viewer forward, "
            "but the segment may not form a crisp memory object."
        )
    if mem >= 0.85:
        return (
            "Memory-heavy beat: this looks distinctive or categorizable even if "
            "it is not the strongest attention spike."
        )
    if mem < 0.35 and attention < 0.35:
        return (
            "Lull zone: neither the BMD memorability axis nor the raw TRIBE "
            "attention dimension sees much pull here."
        )
    if density > 0.75 and mem < 0.60:
        return (
            "Dense but not necessarily useful: there is visual complexity, but "
            "it may be clutter rather than a clean semantic beat."
        )
    if motion > 0.07:
        return "Motion-led beat: likely driven by screen change or cut energy."
    if edge > 0.45:
        return (
            "Shape-rich beat: lots of edges/figures, but score depends on organization."
        )
    return "Middle band: not an obvious hook or trough."


def _natural_read(row: dict[str, Any]) -> str:
    mem = float(row.get("mem_percentile", 0.0))
    tribe_scores = row.get("tribe_scores", {})
    attention = float(tribe_scores.get("attention_score", 0.0)) / 100.0
    density = float(row.get("visual_density_z", 0.0))
    motion = float(row.get("frame_motion_delta_mean", 0.0))
    edge = float(row.get("edge_density_mean", 0.0))
    input_kind = str(row.get("input_kind", "video"))
    if input_kind == "text":
        return _natural_text_read(mem, attention)
    if input_kind == "image" and motion <= 1e-6:
        static_read = _natural_static_image_read(
            mem=mem,
            attention=attention,
            density=density,
        )
        if static_read is not None:
            return static_read
    return _natural_video_read(
        mem=mem,
        attention=attention,
        density=density,
        motion=motion,
        edge=edge,
    )


def _spearman(xs: list[float], ys: list[float]) -> float:
    from scipy.stats import spearmanr

    if len(xs) < 3:
        return 0.0
    result = cast(tuple[Any, Any], spearmanr(xs, ys))
    value = float(result[0])
    return 0.0 if math.isnan(value) else value


def _mean_tribe_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = [
        "overall_score",
        "attention_score",
        "emotion_score",
        "memory_score",
        "visual_score",
        "language_score",
        "cognitive_ease",
        "raw_mean_activation",
    ]
    scores = [row.get("tribe_scores", {}) for row in rows if row.get("tribe_scores")]
    if not scores:
        return _aggregate_tribe_dimensions([])
    out: dict[str, Any] = {
        name: round(_mean([float(score.get(name, 0.0)) for score in scores]), 4)
        if name == "raw_mean_activation"
        else _round_score(_mean([float(score.get(name, 0.0)) for score in scores]))
        for name in dimensions
    }
    dominant_counts = Counter(
        str(score.get("dominant_region", "attention")) for score in scores
    )
    out["dominant_region"] = dominant_counts.most_common(1)[0][0]
    out["recommendations"] = _engagement_recommendations(
        attention=float(out["attention_score"]),
        emotion=float(out["emotion_score"]),
        memory=float(out["memory_score"]),
        visual=float(out["visual_score"]),
        language=float(out["language_score"]),
    )
    return out


def _score_read(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Pass"
    if score >= 40:
        return "Revise"
    return "Fail"


def _threshold_checks(
    *,
    raw: dict[str, Any],
    input_kind: str,
    duration_s: float,
    peak_moments: list[dict[str, Any]],
    hook_score: float | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {
            "metric": "Overall",
            "score": float(raw.get("overall_score", 0.0)),
            "floor": 60.0,
            "passed": float(raw.get("overall_score", 0.0)) >= 60.0,
            "meaning": "General creative signal; below 60 means revise before publishing.",
        }
    ]
    dimension_meanings = {
        "attention_score": "Below 40 usually means the opener or focal interrupt is too easy to miss.",
        "emotion_score": "Below 40 usually means weak stakes, contrast, or payoff.",
        "memory_score": "Below 40 usually means there is no single sticky idea yet.",
        "visual_score": "Below 40 usually means the frame is not doing enough work.",
        "language_score": "Below 40 usually means the wording is generic or semantically thin.",
    }
    for metric, meaning in dimension_meanings.items():
        score = float(raw.get(metric, 0.0))
        if score < 40.0:
            checks.append(
                {
                    "metric": metric.replace("_score", "").replace("_", " ").title(),
                    "score": score,
                    "floor": 40.0,
                    "passed": False,
                    "meaning": meaning,
                }
            )
    if input_kind == "image":
        for metric in ("attention_score", "visual_score"):
            score = float(raw.get(metric, 0.0))
            floor = 60.0 if metric == "attention_score" else 55.0
            checks.append(
                {
                    "metric": metric.replace("_score", "").replace("_", " ").title(),
                    "score": score,
                    "floor": floor,
                    "passed": score >= floor,
                    "meaning": "Static creative needs fast attention and a readable visual field.",
                }
            )
    if input_kind == "video":
        hook = float(hook_score or 0.0)
        checks.append(
            {
                "metric": "Hook",
                "score": hook,
                "floor": 60.0,
                "passed": hook >= 60.0,
                "meaning": "Video final gate: the first two seconds should clear 60.",
            }
        )
        first_peak_limit = duration_s * 0.40 if duration_s > 0 else 0.0
        early_peak = any(
            float(moment.get("timestamp_seconds", 0.0)) <= first_peak_limit
            for moment in peak_moments
        )
        checks.append(
            {
                "metric": "Early peak",
                "score": 100.0 if early_peak else 0.0,
                "floor": 1.0,
                "passed": early_peak,
                "meaning": "At least one peak moment should land in the first 40% of the video.",
            }
        )
        if hook < 50.0:
            checks.append(
                {
                    "metric": "Opening revise",
                    "score": hook,
                    "floor": 50.0,
                    "passed": False,
                    "meaning": "The first two seconds are weak enough that the opener should be fixed first.",
                }
            )
    return checks


def _engagement_explanation(
    *,
    raw: dict[str, Any],
    input_kind: str,
    threshold_checks: list[dict[str, Any]],
    top_mem: dict[str, Any],
    top_attention: dict[str, Any],
) -> dict[str, Any]:
    dims = {
        "attention": float(raw.get("attention_score", 0.0)),
        "emotion": float(raw.get("emotion_score", 0.0)),
        "memory": float(raw.get("memory_score", 0.0)),
        "visual": float(raw.get("visual_score", 0.0)),
        "language": float(raw.get("language_score", 0.0)),
        "cognitive ease": float(raw.get("cognitive_ease", 0.0)),
    }
    ranked = sorted(dims.items(), key=lambda item: item[1], reverse=True)
    strongest = [f"{name} ({score:.0f}/100)" for name, score in ranked[:2]]
    weakest = [f"{name} ({score:.0f}/100)" for name, score in ranked[-2:]]
    failing = [check for check in threshold_checks if not check["passed"]]
    overall = float(raw.get("overall_score", 0.0))
    if overall < 40.0 or any(check["metric"] == "Opening revise" for check in failing):
        verdict = "revise"
    elif failing:
        verdict = "compare" if overall >= 60.0 else "revise"
    else:
        verdict = "ready"

    if verdict == "ready":
        summary = (
            f"Ready to test: TRIBE overall is {overall:.0f}/100, led by "
            f"{strongest[0]} and {strongest[1]}."
        )
    elif verdict == "compare":
        summary = (
            f"Workable but not clean: TRIBE overall is {overall:.0f}/100. "
            f"Improve {weakest[0]} before high-stakes use."
        )
    else:
        summary = (
            f"Revise before publishing: TRIBE overall is {overall:.0f}/100, "
            f"with weakest signal in {weakest[0]}."
        )

    next_edits = list(raw.get("recommendations", []))[:3]
    if input_kind == "video" and any(
        check["metric"] == "Hook" and not check["passed"] for check in threshold_checks
    ):
        next_edits.insert(
            0, "Fix the first two seconds before tuning the rest of the clip."
        )
    if not next_edits:
        next_edits = ["Use variant testing if the asset is strategically important."]

    return {
        "verdict": verdict,
        "plain_english_summary": summary,
        "strongest_signals": strongest,
        "weakest_signals": weakest,
        "threshold_checks": threshold_checks,
        "next_edits": next_edits,
        "raw_tribe_scores": raw,
        "legacy_axes": {
            "bmd_memorability": {
                "top_timestamp": top_mem["timestamp"],
                "top_percentile": top_mem["mem_percentile"],
            },
            "audience_axis_attention": {
                "top_timestamp": top_attention["timestamp"],
                "top_percentile": top_attention["attention_percentile"],
                "note": "Legacy synthetic-persona/audience axis; not the raw TRIBE attention score.",
            },
        },
    }


def _summarize(rows: list[dict[str, Any]], input_kind: str = "video") -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {"error": "No segments scored."}
    for row in ok_rows:
        if "affect_proxy" not in row:
            row["affect_proxy"] = _affect_proxy(row)
    top_mem = max(ok_rows, key=lambda row: row["mem_projection"])
    top_audience_axis = max(ok_rows, key=lambda row: row["attention_projection"])
    top_attention = max(
        ok_rows,
        key=lambda row: float(row.get("tribe_scores", {}).get("attention_score", 0.0)),
    )
    low_rows = sorted(
        ok_rows,
        key=lambda row: row["mem_percentile"]
        + (float(row.get("tribe_scores", {}).get("attention_score", 0.0)) / 100.0),
    )[:3]
    motion_corr = _spearman(
        [float(row.get("frame_motion_delta_mean", 0.0)) for row in ok_rows],
        [
            float(row.get("tribe_scores", {}).get("attention_score", 0.0))
            for row in ok_rows
        ],
    )
    edge_corr = _spearman(
        [float(row.get("edge_density_mean", 0.0)) for row in ok_rows],
        [float(row["mem_percentile"]) for row in ok_rows],
    )
    raw_tribe = _mean_tribe_scores(ok_rows)
    affect_profile = _mean_affect_proxy(ok_rows)
    duration_s = max(float(row.get("end_s", 0.0)) for row in ok_rows)
    peak_moments = _peak_moments_from_rows(ok_rows)
    timeline = _timeline_from_rows(ok_rows)
    hook_score = (
        float(ok_rows[0].get("tribe_hook_score", raw_tribe.get("overall_score", 0.0)))
        if input_kind == "video"
        else None
    )
    if input_kind == "video":
        raw_tribe = {
            **raw_tribe,
            "hook_score": _round_score(float(hook_score or 0.0)),
            "peak_moments": peak_moments,
            "timeline": timeline,
        }
    threshold_checks = _threshold_checks(
        raw=raw_tribe,
        input_kind=input_kind,
        duration_s=duration_s,
        peak_moments=peak_moments,
        hook_score=hook_score,
    )
    explanation = _engagement_explanation(
        raw=raw_tribe,
        input_kind=input_kind,
        threshold_checks=threshold_checks,
        top_mem=top_mem,
        top_attention=top_audience_axis,
    )
    return {
        "n_scored_segments": len(ok_rows),
        "mean_mem_percentile": _mean([float(row["mem_percentile"]) for row in ok_rows]),
        "mean_attention_percentile": _mean(
            [float(row["attention_percentile"]) for row in ok_rows]
        ),
        "mean_tribe_attention_score": float(raw_tribe.get("attention_score", 0.0)),
        "mean_tribe_overall_score": float(raw_tribe.get("overall_score", 0.0)),
        "verdict": explanation["verdict"],
        "plain_english_summary": explanation["plain_english_summary"],
        "strongest_signals": explanation["strongest_signals"],
        "weakest_signals": explanation["weakest_signals"],
        "threshold_checks": explanation["threshold_checks"],
        "next_edits": explanation["next_edits"],
        "raw_tribe_scores": raw_tribe,
        "affect_profile": affect_profile,
        "affect_commentary": _affect_read(affect_profile),
        "engagement_explanation": explanation,
        "breakout_score": None,
        "breakout_note": (
            "Creative-only uploads do not include platform metrics. Breakout scoring needs "
            "views, likes, comments, shares, saves, author baseline, and post age."
        ),
        "top_mem": {
            "timestamp": top_mem["timestamp"],
            "percentile": top_mem["mem_percentile"],
            "z": top_mem["mem_z"],
            "read": top_mem.get("natural_read", ""),
        },
        "top_attention": {
            "timestamp": top_attention["timestamp"],
            "score": float(
                top_attention.get("tribe_scores", {}).get("attention_score", 0.0)
            ),
            "read": top_attention.get("natural_read", ""),
        },
        "top_audience_axis": {
            "timestamp": top_audience_axis["timestamp"],
            "percentile": top_audience_axis["attention_percentile"],
            "z": top_audience_axis["attention_z"],
            "read": top_audience_axis.get("natural_read", ""),
            "note": "Legacy synthetic-persona/audience axis; not raw TRIBE attention.",
        },
        "lowest_windows": [
            {
                "timestamp": row["timestamp"],
                "mem_percentile": row["mem_percentile"],
                "tribe_attention_score": float(
                    row.get("tribe_scores", {}).get("attention_score", 0.0)
                ),
                "audience_axis_percentile": row["attention_percentile"],
                "read": row.get("natural_read", ""),
            }
            for row in low_rows
        ],
        "correlations": {
            "motion_vs_attention_spearman": motion_corr,
            "edge_density_vs_memorability_spearman": edge_corr,
        },
        "commentary": _overall_commentary(ok_rows, motion_corr, edge_corr, input_kind),
    }


def _overall_commentary(
    rows: list[dict[str, Any]],
    motion_corr: float,
    edge_corr: float,
    input_kind: str,
) -> str:
    top_mem = max(rows, key=lambda row: row["mem_projection"])
    top_attention = max(
        rows,
        key=lambda row: float(row.get("tribe_scores", {}).get("attention_score", 0.0)),
    )
    top_attention_score = float(
        top_attention.get("tribe_scores", {}).get("attention_score", 0.0)
    )
    if input_kind == "text":
        return (
            f"The text stimulus sits at {top_mem['mem_percentile']:.0%} memorability "
            f"on the BMD axis, with raw TRIBE attention at {top_attention_score:.0f}/100. "
            "Read the score breakdown first; palette and density fields come from a "
            "rendered text card so visual commentary stays comparable."
        )
    if input_kind == "image":
        return (
            f"The static image scores highest around {top_mem['timestamp']} for memorability "
            f"and around {top_attention['timestamp']} for raw TRIBE attention. Because the "
            "image is scored as a still visual clip, motion is intentionally near zero; "
            "edges, contrast, color spread, and figure organization carry most of the read."
        )
    if top_mem["timestamp"] == top_attention["timestamp"]:
        lead = (
            f"The main hook is {top_mem['timestamp']}: BMD memorability and raw TRIBE attention "
            "peak in the same window, so the clip has a unified spike."
        )
    else:
        lead = (
            f"The memorability peak is {top_mem['timestamp']}, while raw TRIBE attention "
            f"peak is {top_attention['timestamp']}; the clip separates pull from encoding."
        )
    motion = (
        "Motion/change appears strongly tied to attention."
        if motion_corr > 0.45
        else "Attention is not explained by motion alone."
    )
    edge = (
        "Edge/shape density appears related to memorability."
        if edge_corr > 0.35
        else "Raw edge density is not enough to explain memorability."
    )
    return (
        f"{lead} {motion} {edge} The practical rule is to look for structured "
        "density: screen change, contrast, and visual complexity work best when "
        "they resolve into a clear semantic beat instead of clutter."
    )


async def _analyze_video_source(
    *,
    source_path: Path,
    job_id: str,
    filename: str,
    source_type: str,
    segment_seconds: float,
    tmp_path: Path,
    source_url: str | None = None,
    input_kind: str = "video",
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    segments = _segment_video(
        source_path=source_path,
        job_id=job_id,
        segment_seconds=segment_seconds,
        out_dir=tmp_path / "segments",
    )
    for segment in segments:
        segment["input_kind"] = input_kind
    _upload_segments_to_volume(segments)
    scored = await _score_segments_with_tribe(segments)
    visual_rows = _add_visual_analysis(
        source_path=source_path,
        segments=scored,
        frame_dir=tmp_path / "frames",
    )
    notes = [
        "Memorability is BMD-human-label derived.",
        "Attention, emotion, memory, visual, language, and cognitive ease are raw TRIBE-derived engagement dimensions.",
        "Affect labels are NOVA-inspired media proxies from TRIBE plus visual statistics, not EEG PSD decoding.",
        "The legacy audience axis is retained in JSON but is not used as the primary attention score.",
        "Visual figure counts are edge-component proxies, not object detections.",
    ]
    if input_kind == "image":
        notes.append("Image inputs are scored as static 6-second visual clips.")
    if input_kind == "text":
        notes.append("This text input fell back to a rendered-card visual score.")
    notes.extend(extra_notes or [])
    return {
        "job_id": job_id,
        "filename": filename,
        "source_type": source_type,
        "source_url": source_url,
        "input_kind": input_kind,
        "duration_s": _ffprobe_duration(source_path),
        "segment_seconds": segment_seconds,
        "n_segments": len(segments),
        "summary": _summarize(visual_rows, input_kind),
        "segments": visual_rows,
        "notes": notes,
    }


async def _analyze_image_source(
    *,
    image_path: Path,
    job_id: str,
    filename: str,
    source_type: str,
    segment_seconds: float,
    tmp_path: Path,
    source_url: str | None = None,
) -> dict[str, Any]:
    card_path = tmp_path / "image_card.png"
    video_path = tmp_path / "image_stimulus.mp4"
    _render_image_card(image_path, card_path)
    _still_image_to_video(card_path, video_path)
    return await _analyze_video_source(
        source_path=video_path,
        job_id=job_id,
        filename=filename,
        source_type=source_type,
        segment_seconds=STATIC_STIMULUS_SECONDS,
        tmp_path=tmp_path,
        source_url=source_url,
        input_kind="image",
    )


def _text_payload(
    *,
    job_id: str,
    filename: str,
    source_type: str,
    source_url: str | None,
    segment_seconds: float,
    visual_rows: list[dict[str, Any]],
    text: str,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "filename": filename,
        "source_type": source_type,
        "source_url": source_url,
        "input_kind": "text",
        "duration_s": STATIC_STIMULUS_SECONDS,
        "segment_seconds": segment_seconds,
        "n_segments": len(visual_rows),
        "text_characters": len(text),
        "summary": _summarize(visual_rows, "text"),
        "segments": visual_rows,
        "notes": notes,
    }


async def _analyze_text_source(
    *,
    text: str,
    job_id: str,
    filename: str,
    source_type: str,
    segment_seconds: float,
    tmp_path: Path,
    source_url: str | None = None,
) -> dict[str, Any]:
    text_path = tmp_path / "source.txt"
    card_path = tmp_path / "text_card.png"
    video_path = tmp_path / "text_stimulus.mp4"
    _write_text_stimulus(text, text_path)
    _render_text_card(text, card_path, filename)
    _still_image_to_video(card_path, video_path)

    remote_volume_path = f"/video-analyzer/{job_id}/source.txt"
    text_modal_path = _upload_file_to_volume(text_path, remote_volume_path)
    segment = {
        "sample_id": f"text_{job_id}_seg_0000",
        "local_path": str(video_path),
        "modal_path": text_modal_path,
        "remote_volume_path": remote_volume_path,
        "start_s": 0.0,
        "end_s": STATIC_STIMULUS_SECONDS,
        "duration_s": STATIC_STIMULUS_SECONDS,
        "timestamp": f"00:00-{_timestamp(STATIC_STIMULUS_SECONDS)}",
        "input_kind": "text",
    }
    notes = [
        "Memorability is BMD-human-label derived.",
        "Attention, emotion, memory, visual, language, and cognitive ease are raw TRIBE-derived engagement dimensions.",
        "Affect labels are NOVA-inspired media proxies from TRIBE plus visual statistics, not EEG PSD decoding.",
        "The legacy audience axis is retained in JSON but is not used as the primary attention score.",
        "Text inputs use TRIBE's native text pathway when available.",
        "Palette and density commentary come from a rendered text card.",
    ]
    try:
        scored = await asyncio.wait_for(
            _score_text_with_tribe(
                text_modal_path=text_modal_path,
                segment=segment,
            ),
            timeout=TEXT_NATIVE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        scored = None
        notes.append(
            "Native text scoring timed out and fell back to rendered-card scoring."
        )
    except Exception as exc:  # noqa: BLE001
        scored = None
        notes.append(f"Native text scoring fell back to visual-card scoring: {exc}")

    if scored is None:
        return await _analyze_video_source(
            source_path=video_path,
            job_id=job_id,
            filename=filename,
            source_type=source_type,
            segment_seconds=STATIC_STIMULUS_SECONDS,
            tmp_path=tmp_path,
            source_url=source_url,
            input_kind="text",
            extra_notes=notes[3:],
        )

    visual_rows = _add_visual_analysis(
        source_path=video_path,
        segments=[scored],
        frame_dir=tmp_path / "frames",
    )
    return _text_payload(
        job_id=job_id,
        filename=filename,
        source_type=source_type,
        source_url=source_url,
        segment_seconds=segment_seconds,
        visual_rows=visual_rows,
        text=text,
        notes=notes,
    )


async def _save_upload_to_tmp(file: Any, tmp_path: Path) -> tuple[Path, str, str]:
    suffix = Path(file.filename).suffix or ".mp4"
    source_path = tmp_path / f"source{suffix}"
    size = 0
    with source_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ValueError("File exceeds the 250 MB MVP upload limit.")
            out.write(chunk)
    return (
        source_path,
        str(file.filename),
        _classify_media(source_path, file.content_type),
    )


async def _analyze_media_source(
    *,
    source_path: Path,
    media_kind: str,
    job_id: str,
    filename: str,
    source_type: str,
    segment_seconds: float,
    tmp_path: Path,
    source_url: str | None = None,
) -> dict[str, Any]:
    if media_kind == "text":
        return await _analyze_text_source(
            text=_read_text_source(source_path),
            job_id=job_id,
            filename=filename,
            source_type=source_type,
            segment_seconds=segment_seconds,
            tmp_path=tmp_path,
            source_url=source_url,
        )
    if media_kind == "image":
        return await _analyze_image_source(
            image_path=source_path,
            job_id=job_id,
            filename=filename,
            source_type=source_type,
            segment_seconds=segment_seconds,
            tmp_path=tmp_path,
            source_url=source_url,
        )
    return await _analyze_video_source(
        source_path=source_path,
        job_id=job_id,
        filename=filename,
        source_type=source_type,
        segment_seconds=segment_seconds,
        tmp_path=tmp_path,
        source_url=source_url,
    )


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audience Vector Media Analyzer</title>
  <meta name="description" content="Upload video, image, or text inputs; paste public URLs; score raw TRIBE engagement dimensions, BMD memorability, NOVA-inspired affect proxies, and product-friendly threshold checks." />
  <meta name="theme-color" content="#ece5d5" />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://jawaun--video-analyzer.modal.run/" />
  <meta property="og:title" content="Audience Vector Media Analyzer" />
  <meta property="og:description" content="A TRIBE-powered instrument for scoring video, image, and text memorability, attention, affect proxies, and density structure." />
  <meta property="og:image" content="https://jawaun--video-analyzer.modal.run/og-image.png?v=20260529a" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta property="og:image:type" content="image/png" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="Audience Vector Media Analyzer" />
  <meta name="twitter:description" content="Score video, image, and text memorability, attention, and density structure." />
  <meta name="twitter:image" content="https://jawaun--video-analyzer.modal.run/og-image.png?v=20260529a" />
  <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
  <link rel="icon" href="/favicon.ico" sizes="any" />
  <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --canvas: #ece5d5;
      --canvas-2: #e4dcc8;
      --surface: #f5efe3;
      --surface-2: #fbf7ed;
      --ink: #1a1410;
      --ink-2: #3a322a;
      --muted: #7a6e5c;
      --subtle: #a89d87;
      --hair: #d9cfb8;
      --coral: #f15539;
      --signal: #176f68;
      --slate: #24333b;
      --gold: #8a6b3e;
      --deep: #17120e;
      --oxide: #9a523c;
      --mist: #d0d8cc;
      --cream-shadow: rgba(255, 255, 255, .76);
      --earth-shadow: rgba(82, 66, 42, .18);
      --serif: "Cormorant Garamond", "Times New Roman", serif;
      --sans: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: hidden;
      background:
        linear-gradient(90deg, rgba(26,20,16,.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(26,20,16,.03) 1px, transparent 1px),
        linear-gradient(118deg, transparent 0 62%, rgba(23,111,104,.065) 62% 74%, transparent 74%),
        linear-gradient(25deg, rgba(241,85,57,.07), transparent 38%),
        var(--canvas);
      background-size: 42px 42px, 42px 42px, auto, auto, auto;
      color: var(--ink);
      font: 14px/1.52 var(--sans);
      letter-spacing: -0.02em;
      -webkit-font-smoothing: antialiased;
    }
    button, input, select, textarea { font: inherit; letter-spacing: inherit; }
    button { border: 0; }
    p { color: var(--muted); margin: 0; }
    small { color: var(--muted); font-size: 11px; letter-spacing: .02em; }
    .app {
      width: min(1560px, calc(100% - 40px));
      margin: 0 auto;
      padding: 22px 0 32px;
      overflow-x: clip;
    }
    .sitebar {
      position: sticky;
      top: 0;
      z-index: 30;
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 22px;
      padding: 16px 0 18px;
      margin-bottom: 22px;
      border-bottom: 1px solid var(--hair);
      background:
        linear-gradient(180deg, rgba(251,247,237,.88), rgba(236,229,213,.78)),
        rgba(236,229,213,.88);
      backdrop-filter: blur(18px);
      box-shadow: 0 10px 26px rgba(82,66,42,.07);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 13px;
      min-width: 0;
    }
    .seal {
      width: 36px;
      height: 36px;
      display: inline-grid;
      place-items: center;
      border-radius: 999px;
      color: var(--gold);
      background: var(--surface);
      box-shadow: 6px 6px 13px var(--earth-shadow), -6px -6px 13px var(--cream-shadow);
      flex: 0 0 auto;
    }
    .seal svg { width: 22px; height: 22px; }
    .wordmark {
      font-family: var(--serif);
      font-size: 23px;
      font-weight: 500;
      line-height: .95;
      letter-spacing: .06em;
      white-space: nowrap;
    }
    .tagline {
      margin-top: 3px;
      color: var(--muted);
      font-size: 9px;
      letter-spacing: .24em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .nav {
      display: flex;
      gap: 26px;
      color: var(--ink-2);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: .22em;
      text-transform: uppercase;
    }
    .nav span { opacity: .72; }
    .utility {
      justify-self: end;
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 31px;
      padding: 7px 11px;
      border: 1px solid rgba(217,207,184,.82);
      border-radius: 999px;
      background: rgba(251,247,237,.72);
      color: var(--ink-2);
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
      box-shadow: inset 1px 1px 2px rgba(255,255,255,.78), inset -1px -1px 2px rgba(82,66,42,.08);
    }
    .chip.hot { color: #fff; background: var(--ink); border-color: var(--ink); }
    .masthead {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 470px;
      gap: 24px;
      align-items: stretch;
      margin-bottom: 24px;
    }
    .masthead > *, .workspace > *, .panel { min-width: 0; }
    .intro {
      position: relative;
      min-height: 300px;
      padding: 34px 38px;
      border: 1px solid var(--hair);
      min-width: 0;
      overflow: hidden;
      background:
        linear-gradient(90deg, rgba(23,111,104,.08) 0 2px, transparent 2px 100%),
        linear-gradient(135deg, rgba(245,239,227,.94), rgba(228,220,200,.72)),
        var(--surface);
      background-size: 74px 100%, auto, auto;
      box-shadow: 18px 18px 36px var(--earth-shadow), -14px -14px 32px var(--cream-shadow);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .intro::before {
      content: "";
      position: absolute;
      inset: 14px;
      pointer-events: none;
      border: 1px solid rgba(217,207,184,.74);
    }
    .intro::after {
      content: "";
      position: absolute;
      right: 28px;
      top: 28px;
      width: 92px;
      height: calc(100% - 56px);
      pointer-events: none;
      opacity: .46;
      background:
        linear-gradient(90deg, transparent 0 48%, var(--coral) 48% 52%, transparent 52%),
        repeating-linear-gradient(180deg, transparent 0 17px, rgba(26,20,16,.18) 17px 18px);
    }
    .intro > * { position: relative; z-index: 1; }
    .eyebrow {
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .28em;
      text-transform: uppercase;
    }
    h1 {
      max-width: 18ch;
      margin: 20px 0 16px;
      font-family: var(--serif);
      font-size: 68px;
      font-weight: 400;
      line-height: .94;
      letter-spacing: -0.02em;
    }
    .intro-copy {
      max-width: 62ch;
      color: var(--ink-2);
      font-size: 15px;
      line-height: 1.64;
      overflow-wrap: break-word;
    }
    .hero-modes {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      max-width: 620px;
      margin-top: 24px;
    }
    .hero-mode {
      min-height: 58px;
      padding: 10px 12px;
      border: 1px solid rgba(217,207,184,.8);
      background: rgba(251,247,237,.46);
      box-shadow: inset 2px 2px 5px rgba(82,66,42,.08), inset -2px -2px 5px rgba(255,255,255,.58);
    }
    .hero-mode b {
      display: block;
      font-size: 12px;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .hero-mode span {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 650;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .quick-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 28px;
    }
    .stat {
      padding-top: 13px;
      border-top: 1px solid var(--hair);
    }
    .stat b {
      display: block;
      font-size: 18px;
      letter-spacing: -0.04em;
    }
    .stat span {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 650;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    .specimen {
      position: relative;
      min-height: 300px;
      padding: 18px 18px 70px;
      overflow: hidden;
      border: 1px solid var(--hair);
      background:
        linear-gradient(180deg, rgba(23,18,14,.96), rgba(36,51,59,.92)),
        var(--deep);
      box-shadow: inset 9px 9px 24px rgba(0,0,0,.22), inset -10px -10px 26px rgba(255,255,255,.05), 13px 13px 28px rgba(82,66,42,.15);
    }
    .specimen::before {
      content: "MEDIA SIGNAL FIELD";
      position: absolute;
      left: 18px;
      right: 18px;
      bottom: 41px;
      color: rgba(245,239,227,.72);
      font-size: 9px;
      font-weight: 780;
      letter-spacing: .24em;
    }
    .specimen::after {
      content: "";
      position: absolute;
      left: 18px;
      right: 18px;
      bottom: 24px;
      height: 8px;
      background:
        linear-gradient(90deg, var(--coral) 0 16%, var(--gold) 16% 31%, var(--signal) 31% 58%, var(--mist) 58% 74%, rgba(245,239,227,.22) 74%),
        rgba(245,239,227,.12);
      box-shadow: 0 0 0 1px rgba(245,239,227,.16);
    }
    .specimen-grid {
      height: 100%;
      min-height: 264px;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .frame-tile {
      position: relative;
      overflow: hidden;
      min-width: 0;
      background: var(--surface);
      box-shadow: 7px 7px 16px rgba(0,0,0,.22), -7px -7px 16px rgba(255,255,255,.04);
      transition: transform .28s ease, filter .28s ease;
    }
    .frame-tile:hover {
      transform: translateY(-2px);
      filter: saturate(1.08) contrast(1.04);
    }
    .frame-tile::after {
      content: attr(data-label);
      position: absolute;
      left: 12px;
      bottom: 10px;
      color: rgba(255,255,255,.82);
      font-size: 9px;
      font-weight: 700;
      letter-spacing: .2em;
      text-transform: uppercase;
    }
    .frame-a {
      background:
        linear-gradient(180deg, rgba(20,16,10,.02), rgba(20,16,10,.44)),
        repeating-linear-gradient(45deg, rgba(36,51,59,.42) 0 5px, transparent 5px 14px),
        linear-gradient(135deg, #d6c7a2, #8b7548 60%, #3d3220);
    }
    .frame-b {
      background:
        linear-gradient(180deg, rgba(20,16,10,.08), rgba(20,16,10,.28)),
        linear-gradient(90deg, transparent 0 64%, rgba(241,85,57,.68) 64% 82%, transparent 82%),
        linear-gradient(135deg, #e8dbbf, #b89766 56%, #5e4724);
    }
    .frame-c {
      background:
        linear-gradient(180deg, rgba(20,16,10,.05), rgba(20,16,10,.5)),
        repeating-linear-gradient(-35deg, rgba(23,111,104,.34) 0 4px, transparent 4px 12px),
        linear-gradient(135deg, #4a3724, #14100b);
    }
    .frame-d {
      background:
        repeating-linear-gradient(90deg, transparent 0 34px, rgba(241,85,57,.38) 34px 47px, transparent 47px 68px),
        repeating-linear-gradient(0deg, transparent 0 42px, rgba(138,107,62,.42) 42px 55px, transparent 55px 84px),
        linear-gradient(135deg, #f2ead6, #c8b79a);
      background-size: 54px 54px, 62px 62px, auto;
    }
    .workspace {
      display: grid;
      grid-template-columns: 355px minmax(0, 1fr) 340px;
      gap: 18px;
      align-items: start;
    }
    .panel {
      border: 1px solid rgba(217,207,184,.92);
      background: var(--surface);
      box-shadow: 13px 13px 26px var(--earth-shadow), -11px -11px 24px var(--cream-shadow);
      transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease;
      animation: liftIn .44s ease both;
    }
    .panel:hover {
      transform: translateY(-1px);
      box-shadow: 17px 17px 34px rgba(82,66,42,.19), -13px -13px 28px rgba(255,255,255,.7);
    }
    .panel-inner { padding: 18px; }
    .panel-title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      padding: 17px 18px;
      border-bottom: 1px solid var(--hair);
    }
    .panel-title h2, .panel-title h3 {
      margin: 0;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .2em;
      text-transform: uppercase;
    }
    .panel-title span {
      color: var(--muted);
      font-size: 10px;
      font-weight: 650;
      letter-spacing: .16em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    label, .field-label {
      display: block;
      margin: 16px 0 7px;
      color: var(--ink-2);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid rgba(217,207,184,.62);
      border-radius: 0;
      background: var(--surface-2);
      color: var(--ink);
      padding: 12px 12px;
      box-shadow: inset 4px 4px 9px rgba(82,66,42,.12), inset -4px -4px 9px rgba(255,255,255,.75);
    }
    input[type="file"] { color: var(--muted); }
    .mode-deck {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }
    .mode-card {
      min-height: 74px;
      padding: 11px 9px;
      border: 1px solid rgba(217,207,184,.86);
      background: rgba(251,247,237,.5);
      box-shadow: inset 2px 2px 5px rgba(82,66,42,.08), inset -2px -2px 5px rgba(255,255,255,.62);
    }
    .mode-card i {
      display: block;
      width: 21px;
      height: 21px;
      margin-bottom: 9px;
      border: 1px solid var(--muted);
      background: var(--surface-2);
      box-shadow: 2px 2px 4px rgba(82,66,42,.12), -2px -2px 4px rgba(255,255,255,.56);
    }
    .mode-card:nth-child(1) i { background: linear-gradient(90deg, var(--slate) 0 28%, var(--surface-2) 28% 42%, var(--slate) 42% 70%, var(--surface-2) 70%); }
    .mode-card:nth-child(2) i { background: linear-gradient(135deg, var(--coral) 0 46%, var(--surface-2) 46% 62%, var(--signal) 62%); }
    .mode-card:nth-child(3) i { background: repeating-linear-gradient(180deg, var(--ink) 0 2px, transparent 2px 6px), var(--surface-2); }
    .mode-card b {
      display: block;
      font-size: 11px;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .mode-card small {
      display: block;
      margin-top: 2px;
      font-size: 9px;
      line-height: 1.25;
      text-transform: uppercase;
    }
    .file-native {
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }
    .upload-shell {
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 12px;
      align-items: center;
      min-height: 96px;
      margin: 0;
      padding: 15px;
      cursor: pointer;
      border: 1px solid rgba(217,207,184,.8);
      background:
        linear-gradient(135deg, rgba(251,247,237,.72), rgba(228,220,200,.44)),
        var(--surface-2);
      box-shadow: inset 5px 5px 12px rgba(82,66,42,.1), inset -5px -5px 12px rgba(255,255,255,.72);
      transition: transform .2s ease, border-color .2s ease, background .2s ease;
    }
    .upload-shell:hover {
      transform: translateY(-1px);
      border-color: rgba(241,85,57,.42);
    }
    .upload-mark {
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      background: var(--deep);
      color: #f7eddc;
      font-size: 24px;
      font-weight: 400;
      box-shadow: 5px 5px 10px rgba(82,66,42,.2), -4px -4px 10px rgba(255,255,255,.58);
    }
    .upload-shell b {
      display: block;
      color: var(--ink);
      font-size: 13px;
      letter-spacing: .03em;
      text-transform: none;
    }
    .upload-shell small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.35;
      letter-spacing: .1em;
      text-transform: uppercase;
    }
    textarea {
      min-height: 122px;
      resize: vertical;
    }
    input:focus, select:focus, textarea:focus, button:focus-visible {
      outline: 2px solid rgba(241,85,57,.42);
      outline-offset: 3px;
    }
    .run {
      position: relative;
      width: 100%;
      min-height: 50px;
      margin-top: 16px;
      cursor: pointer;
      background: var(--ink);
      color: #f7eddc;
      font-size: 11px;
      font-weight: 750;
      letter-spacing: .22em;
      text-transform: uppercase;
      box-shadow: 7px 7px 15px rgba(82,66,42,.22), -6px -6px 14px rgba(255,255,255,.62);
      transition: transform .2s ease, background .2s ease, color .2s ease;
    }
    .run::after {
      content: "";
      position: absolute;
      left: 0;
      bottom: 0;
      width: 100%;
      height: 3px;
      background: linear-gradient(90deg, var(--coral), var(--signal), var(--gold));
      transform: scaleX(.38);
      transform-origin: left center;
      transition: transform .25s ease;
    }
    .run:hover { transform: translateY(-1px); background: var(--coral); }
    .run:hover::after { transform: scaleX(1); }
    .run:disabled { opacity: .52; cursor: wait; transform: none; }
    .limits {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 9px;
      margin-top: 14px;
    }
    .limit {
      min-height: 62px;
      padding: 10px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.48);
    }
    .limit b { display: block; font-size: 17px; }
    .limit span {
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .queue {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }
    .batch {
      border: 1px solid rgba(34,28,22,.18);
      background: rgba(251,247,237,.4);
      box-shadow: 0 10px 24px rgba(47,38,24,.06);
    }
    .batch.done {
      border-color: rgba(47,143,131,.32);
    }
    .batch.failed {
      border-color: rgba(156,47,38,.32);
    }
    .batch-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 9px 10px;
      border-bottom: 1px solid var(--hair);
      background:
        linear-gradient(90deg, rgba(27,22,18,.95), rgba(27,22,18,.86)),
        var(--deep);
      color: var(--paper);
    }
    .batch-head strong {
      font-size: 10px;
      font-weight: 760;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    .batch-meta {
      color: rgba(251,247,237,.72);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
      text-align: right;
    }
    .batch-jobs {
      display: grid;
      gap: 8px;
      padding: 9px;
    }
    .batch-selector {
      display: grid;
      gap: 10px;
      padding: 0 9px 10px;
    }
    .selector-card {
      border: 1px solid var(--hair);
      background:
        linear-gradient(145deg, rgba(251,247,237,.66), rgba(224,216,195,.48));
      box-shadow: inset 4px 4px 11px rgba(82,66,42,.08), inset -4px -4px 11px rgba(255,255,255,.48);
    }
    .selector-top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
      padding: 12px;
      border-bottom: 1px solid var(--hair);
    }
    .selector-top b,
    .selector-empty b {
      display: block;
      color: var(--ink);
      font-size: 11px;
      font-weight: 780;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .selector-top strong {
      display: block;
      margin-top: 5px;
      color: var(--ink);
      font-family: var(--serif);
      font-size: 22px;
      font-weight: 400;
      line-height: 1;
      overflow-wrap: anywhere;
    }
    .selector-top span,
    .selector-empty span {
      display: block;
      margin-top: 5px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
    }
    .selector-score {
      min-width: 70px;
      padding: 8px 9px;
      border: 1px solid rgba(34,28,22,.2);
      background: rgba(251,247,237,.54);
      color: var(--ink);
      text-align: right;
      box-shadow: 5px 5px 13px rgba(82,66,42,.12), -5px -5px 13px rgba(255,255,255,.52);
    }
    .selector-score small {
      display: block;
      color: var(--muted);
      font-size: 9px;
      font-weight: 760;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .selector-score em {
      display: block;
      color: var(--ink);
      font-style: normal;
      font-size: 22px;
      font-weight: 760;
    }
    .selector-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--hair);
    }
    .selector-actions button {
      min-height: 32px;
      cursor: pointer;
      border: 1px solid rgba(34,28,22,.24);
      background: rgba(251,247,237,.52);
      color: var(--ink);
      font-size: 9px;
      font-weight: 780;
      letter-spacing: .16em;
      text-transform: uppercase;
      box-shadow: 4px 4px 10px rgba(82,66,42,.1), -4px -4px 10px rgba(255,255,255,.5);
      transition: transform .2s ease, background .2s ease, color .2s ease;
    }
    .selector-actions button:hover {
      transform: translateY(-1px);
      background: var(--ink);
      color: #f7eddc;
    }
    .selector-ranking {
      display: grid;
      gap: 1px;
      background: var(--hair);
    }
    .selector-row {
      display: grid;
      grid-template-columns: 30px minmax(0, 1fr) 52px;
      gap: 8px;
      align-items: center;
      padding: 9px 10px;
      background: rgba(251,247,237,.58);
    }
    .selector-row:first-child {
      background: rgba(241,85,57,.08);
    }
    .selector-rank {
      color: var(--muted);
      font-size: 10px;
      font-weight: 780;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    .selector-name {
      min-width: 0;
      color: var(--ink);
      font-size: 12px;
      font-weight: 680;
      overflow-wrap: anywhere;
    }
    .selector-name small {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 500;
      line-height: 1.35;
    }
    .selector-empty {
      padding: 11px 12px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.42);
    }
    .history-panel {
      display: grid;
      gap: 9px;
      margin-top: 14px;
    }
    .history-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 10px;
      font-weight: 760;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    .history-head button,
    .history-actions button {
      min-height: 28px;
      cursor: pointer;
      border: 1px solid rgba(34,28,22,.2);
      background: rgba(251,247,237,.46);
      color: var(--ink);
      font-size: 9px;
      font-weight: 780;
      letter-spacing: .14em;
      text-transform: uppercase;
      transition: transform .2s ease, background .2s ease, color .2s ease;
    }
    .history-head button:hover,
    .history-actions button:hover {
      transform: translateY(-1px);
      background: var(--ink);
      color: #f7eddc;
    }
    .history-list {
      display: grid;
      gap: 8px;
    }
    .history-card {
      padding: 10px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.42);
      box-shadow: inset 3px 3px 8px rgba(82,66,42,.07), inset -3px -3px 8px rgba(255,255,255,.45);
    }
    .history-card strong {
      display: block;
      color: var(--ink);
      font-size: 12px;
      font-weight: 680;
      overflow-wrap: anywhere;
    }
    .history-card small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.35;
    }
    .history-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 9px;
    }
    .job {
      position: relative;
      padding: 11px 11px 11px 30px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.48);
      overflow: hidden;
    }
    .job::before {
      content: "";
      position: absolute;
      left: 11px;
      top: 16px;
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--coral);
      box-shadow: 0 0 0 0 rgba(241,85,57,.42);
      animation: pulse 1.5s ease-in-out infinite;
    }
    .job.done::before {
      background: var(--signal);
      animation: none;
    }
    .job.failed::before {
      background: #9c2f26;
      animation: none;
    }
    .job strong {
      display: block;
      color: var(--ink);
      font-size: 12px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .status {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(241,85,57,.38); }
      70% { box-shadow: 0 0 0 9px rgba(241,85,57,0); }
      100% { box-shadow: 0 0 0 0 rgba(241,85,57,0); }
    }
    .results {
      display: grid;
      gap: 18px;
      min-width: 0;
    }
    .empty {
      min-height: 520px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 20px;
      padding: 0;
      overflow: hidden;
      background: linear-gradient(180deg, var(--surface), rgba(245,239,227,.72));
    }
    .empty-stage {
      position: relative;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      min-height: 320px;
      padding: 18px;
      background:
        linear-gradient(180deg, rgba(23,18,14,.95), rgba(36,51,59,.88)),
        var(--deep);
    }
    .empty-stage::before {
      content: "";
      position: absolute;
      left: 18px;
      right: 18px;
      top: 50%;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(241,85,57,.72), transparent);
      animation: scan 3.8s ease-in-out infinite;
    }
    .empty-frame {
      position: relative;
      overflow: hidden;
      min-height: 160px;
      border: 1px solid rgba(245,239,227,.1);
    }
    .empty-frame::after {
      content: attr(data-label);
      position: absolute;
      left: 18px;
      bottom: 16px;
      color: rgba(255,255,255,.78);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: .22em;
      text-transform: uppercase;
    }
    @keyframes scan {
      0%, 100% { transform: translateY(-86px); opacity: .3; }
      50% { transform: translateY(86px); opacity: .9; }
    }
    .empty-copy {
      padding: 22px 24px 24px;
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      border-top: 1px solid var(--hair);
    }
    .empty-copy h2 {
      margin: 0 0 8px;
      font-family: var(--serif);
      font-size: 32px;
      font-weight: 400;
      line-height: 1;
      letter-spacing: -0.02em;
    }
    .empty-copy p { max-width: 58ch; }
    .result {
      overflow: hidden;
      animation: liftIn .38s ease both;
    }
    @keyframes liftIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .result-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: start;
      padding: 20px;
      border-bottom: 1px solid var(--hair);
      background: rgba(251,247,237,.42);
    }
    .source-kicker {
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: .2em;
      text-transform: uppercase;
    }
    .result h2 {
      margin: 0 0 10px;
      font-family: var(--serif);
      font-size: clamp(28px, 3.4vw, 46px);
      font-weight: 400;
      line-height: 1;
      overflow-wrap: anywhere;
    }
    .download {
      min-width: 132px;
      min-height: 38px;
      cursor: pointer;
      border: 1px solid var(--ink);
      background: transparent;
      color: var(--ink);
      font-size: 10px;
      font-weight: 760;
      letter-spacing: .18em;
      text-transform: uppercase;
      transition: background .2s ease, color .2s ease, transform .2s ease;
    }
    .download:hover {
      transform: translateY(-1px);
      background: var(--ink);
      color: #f7eddc;
    }
    .numbers {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 0;
      border-bottom: 1px solid var(--hair);
      background: var(--surface);
    }
    .num {
      min-height: 96px;
      padding: 16px;
      border-right: 1px solid var(--hair);
    }
    .num:last-child { border-right: 0; }
    .num span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    .num b {
      display: block;
      margin-top: 12px;
      font-size: 24px;
      letter-spacing: -0.05em;
    }
    .num small {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }
    .breakdown {
      display: grid;
      gap: 14px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--hair);
      background:
        linear-gradient(135deg, rgba(251,247,237,.62), rgba(228,220,200,.36));
    }
    .section-label {
      color: var(--muted);
      font-size: 10px;
      font-weight: 780;
      letter-spacing: .2em;
      text-transform: uppercase;
    }
    .dimension-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .dimension {
      padding: 12px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.56);
      box-shadow: inset 3px 3px 8px rgba(82,66,42,.08), inset -3px -3px 8px rgba(255,255,255,.5);
    }
    .dimension strong {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
      font-size: 12px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .dimension strong span {
      font-size: 16px;
      letter-spacing: -0.03em;
      text-transform: none;
    }
    .dimension .bar {
      margin-top: 10px;
      height: 7px;
    }
    .dimension small {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }
    .checks {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 9px;
    }
    .check {
      position: relative;
      padding: 10px 12px 10px 28px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.48);
      color: var(--ink-2);
      font-size: 11px;
      line-height: 1.4;
    }
    .check::before {
      content: "";
      position: absolute;
      left: 11px;
      top: 14px;
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--signal);
    }
    .check.fail::before { background: var(--coral); }
    .check b {
      display: block;
      color: var(--ink);
      font-size: 10px;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .timeline {
      display: grid;
      gap: 0;
      background: rgba(251,247,237,.28);
    }
    .row {
      display: grid;
      grid-template-columns: 94px minmax(180px, .9fr) minmax(220px, 1.25fr);
      gap: 14px;
      align-items: center;
      padding: 15px 18px;
      border-bottom: 1px solid var(--hair);
      transition: background .2s ease;
    }
    .row:hover { background: rgba(255,255,255,.34); }
    .row:last-child { border-bottom: 0; }
    .time {
      display: grid;
      gap: 4px;
      color: var(--ink);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: .02em;
    }
    .time span {
      color: var(--muted);
      font-size: 9px;
      font-weight: 750;
      letter-spacing: .2em;
      text-transform: uppercase;
    }
    .bars {
      display: grid;
      gap: 9px;
    }
    .metric {
      display: grid;
      grid-template-columns: 42px 1fr 66px;
      gap: 9px;
      align-items: center;
    }
    .metric-label {
      color: var(--ink-2);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    .bar {
      height: 9px;
      background: rgba(26,20,16,.08);
      box-shadow: inset 2px 2px 4px rgba(82,66,42,.14), inset -2px -2px 4px rgba(255,255,255,.55);
      overflow: hidden;
    }
    .bar i {
      display: block;
      width: calc(var(--v) * 1%);
      height: 100%;
      background: var(--slate);
      transform-origin: left center;
      animation: fillBar .7s cubic-bezier(.2,.8,.2,1) both;
    }
    .bar.att i { background: var(--signal); }
    @keyframes fillBar {
      from { transform: scaleX(0); }
      to { transform: scaleX(1); }
    }
    .read {
      color: var(--ink-2);
      font-size: 12px;
      line-height: 1.55;
    }
    .palette {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 9px;
    }
    .swatch {
      width: 28px;
      height: 18px;
      border: 1px solid rgba(26,20,16,.18);
      background: var(--c);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);
    }
    .score-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 9px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 23px;
      padding: 4px 8px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.55);
      color: var(--muted);
      font-size: 10px;
      font-weight: 650;
      white-space: nowrap;
    }
    .inspector {
      background:
        linear-gradient(180deg, rgba(245,239,227,.96), rgba(228,220,200,.78));
    }
    .side-list {
      display: grid;
      gap: 12px;
    }
    .callout {
      padding: 14px;
      border: 1px solid var(--hair);
      background: rgba(251,247,237,.48);
      box-shadow: inset 3px 3px 8px rgba(82,66,42,.08), inset -3px -3px 8px rgba(255,255,255,.54);
      transition: transform .2s ease, border-color .2s ease;
    }
    .callout:hover {
      transform: translateX(2px);
      border-color: rgba(23,111,104,.36);
    }
    .callout b {
      display: block;
      margin-bottom: 6px;
      color: var(--ink);
      font-size: 11px;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    .mini-matrix {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 16px;
    }
    .matrix-cell {
      min-height: 94px;
      padding: 12px;
      border: 1px solid var(--hair);
      background: var(--canvas-2);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .matrix-cell:nth-child(1), .matrix-cell:nth-child(2) {
      background: var(--deep);
      color: #f7eddc;
      border-color: var(--deep);
    }
    .matrix-cell:nth-child(1) span, .matrix-cell:nth-child(2) span {
      color: rgba(247,237,220,.62);
    }
    .matrix-cell strong {
      font-size: 20px;
      letter-spacing: -0.05em;
    }
    .matrix-cell span {
      color: var(--muted);
      font-size: 9px;
      font-weight: 750;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    @media (max-width: 1240px) {
      .masthead, .workspace { grid-template-columns: 1fr; }
      .workspace { gap: 16px; }
      .specimen { min-height: 260px; }
    }
    @media (max-width: 820px) {
      .app {
        width: calc(100vw - 40px);
        max-width: calc(100vw - 40px);
        padding-top: 12px;
      }
      .sitebar { grid-template-columns: 1fr; gap: 12px; position: relative; }
      .nav { order: 3; overflow-x: auto; padding-bottom: 2px; }
      .utility { justify-self: start; }
      .panel-title span { display: none; }
      .masthead, .workspace {
        width: 100%;
        max-width: 100%;
        gap: 14px;
        overflow: hidden;
      }
      .intro, .specimen, .panel, .results, .workspace > * {
        width: 100%;
        max-width: 100%;
      }
      .intro { min-height: auto; padding: 25px 20px; overflow: hidden; }
      .intro::after { display: none; }
      .intro-copy { width: 100%; max-width: 34ch; }
      h1 { font-size: 42px; }
      .quick-stats, .numbers, .empty-stage { grid-template-columns: 1fr 1fr; }
      .dimension-grid, .checks { grid-template-columns: 1fr 1fr; }
      .row { grid-template-columns: 1fr; gap: 10px; }
      .result-head { grid-template-columns: 1fr; }
      .download { width: 100%; }
    }
    @media (max-width: 520px) {
      .quick-stats, .numbers, .limits, .mini-matrix, .specimen-grid, .empty-stage, .dimension-grid, .checks { grid-template-columns: 1fr; }
      .selector-top, .selector-row { grid-template-columns: 1fr; }
      .selector-score { text-align: left; }
      .num { border-right: 0; border-bottom: 1px solid var(--hair); }
      .num:last-child { border-bottom: 0; }
    }
    @media (max-width: 420px) {
      .specimen { display: none; }
      .intro { padding: 24px 20px; }
      h1 { font-size: 39px; }
    }
  </style>
</head>
<body>
  <main class="app">
    <header class="sitebar">
      <div class="brand">
        <span class="seal" aria-hidden="true">
          <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.35">
            <circle cx="32" cy="32" r="28"/>
            <path d="M32 10 L37 27 L54 32 L37 37 L32 54 L27 37 L10 32 L27 27 Z"/>
          </svg>
        </span>
        <div>
          <div class="wordmark">Audience Vector</div>
          <div class="tagline">Media scoring instrument</div>
        </div>
      </div>
      <nav class="nav" aria-label="Interface sections">
        <span>Intake</span>
        <span>Timeline</span>
        <span>Readout</span>
      </nav>
      <div class="utility">
        <span class="chip">TRIBE v2</span>
        <span class="chip hot">Live</span>
      </div>
    </header>

    <section class="masthead">
      <div class="intro">
        <div>
          <div class="eyebrow">Memorability Lab</div>
          <h1>Score the shape of media before it lands.</h1>
          <p class="intro-copy">Upload video, image, or text files; paste public URLs; or drop raw copy. The app exposes raw TRIBE engagement dimensions, BMD memorability, NOVA-inspired affect proxies, product thresholds, and plain-English reads of language, palette, motion, edge, and density structure.</p>
          <div class="hero-modes" aria-label="Supported media modes">
            <div class="hero-mode"><b>Video</b><span>Temporal windows</span></div>
            <div class="hero-mode"><b>Image</b><span>Static clips</span></div>
            <div class="hero-mode"><b>Text</b><span>Native TRIBE path</span></div>
          </div>
        </div>
        <div class="quick-stats" aria-label="Analyzer capabilities">
          <div class="stat"><b>30s</b><span>Max window</span></div>
          <div class="stat"><b>3 axes</b><span>Memory + attention + affect</span></div>
          <div class="stat"><b>JSON</b><span>Exportable readout</span></div>
        </div>
      </div>
      <aside class="specimen" aria-label="Visual motif preview">
        <div class="specimen-grid">
          <div class="frame-tile frame-a" data-label="edge map"></div>
          <div class="frame-tile frame-b" data-label="hook"></div>
          <div class="frame-tile frame-c" data-label="signal"></div>
          <div class="frame-tile frame-d" data-label="palette"></div>
        </div>
      </aside>
    </section>

    <section class="workspace">
      <aside class="panel">
        <div class="panel-title">
          <h2>Intake</h2>
          <span>Batch ready</span>
        </div>
        <div class="panel-inner">
        <form id="form">
          <div class="mode-deck" aria-label="Supported media inputs">
            <div class="mode-card"><i aria-hidden="true"></i><b>Video</b><small>segment</small></div>
            <div class="mode-card"><i aria-hidden="true"></i><b>Image</b><small>still clip</small></div>
            <div class="mode-card"><i aria-hidden="true"></i><b>Text</b><small>language</small></div>
          </div>
          <div class="field-label">Media files</div>
          <input class="file-native" id="files" name="files" type="file" accept="video/*,image/*,.txt,.md,.markdown,.csv,.json,.html,.htm,text/plain,text/markdown" multiple />
          <label class="upload-shell" for="files">
            <span class="upload-mark" aria-hidden="true">+</span>
            <span><b>Choose media files</b><small id="fileSummary">MP4, MOV, PNG, JPG, TXT, MD</small></span>
          </label>
          <label for="urls">Video, image, or text URLs</label>
          <textarea id="urls" name="urls" rows="4" placeholder="https://www.youtube.com/watch?v=...&#10;https://example.com/image.png&#10;https://example.com/article.html"></textarea>
          <label for="texts">Text / copy</label>
          <textarea id="texts" name="texts" rows="5" placeholder="Paste scripts, titles, hooks, prompts, or copy. Use a blank line or --- to split multiple text blocks."></textarea>
          <label for="segment">Window length</label>
          <select id="segment" name="segment_seconds">
            <option value="10" selected>10 seconds</option>
            <option value="5">5 seconds</option>
            <option value="15">15 seconds</option>
            <option value="30">30 seconds</option>
          </select>
          <button class="run" id="run" type="submit">Analyze Media</button>
        </form>
        <div class="limits">
          <div class="limit"><b>6 min</b><span>Per video</span></div>
          <div class="limit"><b>250 MB</b><span>MVP cap</span></div>
        </div>
        <div class="queue" id="queue"></div>
        <div class="history-panel" id="historyPanel"></div>
        </div>
      </aside>
      <section class="results" id="results">
        <div class="panel empty">
          <div class="panel-title">
            <h2>Analysis Stage</h2>
            <span>Awaiting media</span>
          </div>
          <div class="empty-stage" aria-hidden="true">
            <div class="empty-frame frame-a" data-label="motion"></div>
            <div class="empty-frame frame-b" data-label="attention"></div>
            <div class="empty-frame frame-c" data-label="memory"></div>
            <div class="empty-frame frame-d" data-label="density"></div>
          </div>
          <div class="empty-copy">
            <div>
              <h2>Drop media into the instrument.</h2>
              <p>The first result will replace this specimen board with a scored timeline, palette swatches, natural-language reads, and downloadable analysis JSON.</p>
            </div>
            <span class="chip">No paper updates</span>
          </div>
        </div>
      </section>
      <aside class="panel inspector">
        <div class="panel-title">
          <h2>Readout</h2>
          <span>Score grammar</span>
        </div>
        <div class="panel-inner">
        <div class="side-list">
          <p class="callout"><b>Score breakdown</b>Raw TRIBE dimensions summarize predicted attention, emotion, memory, visual, language, and cognitive-ease signal.</p>
          <p class="callout"><b>Affect proxy</b>NOVA-inspired happy, anger, fear, sadness, disgust, and neutral scores come from TRIBE plus media statistics. They are not EEG PSD decoding.</p>
          <p class="callout"><b>BMD memorability</b>Projection onto the human-label memorability direction. Keep it separate from TRIBE's raw memory dimension.</p>
          <p class="callout"><b>Audience axis</b>The old persona-derived attention axis is retained as context in JSON, but raw TRIBE attention is the product-facing attention score.</p>
          <p class="callout"><b>Modalities</b>Video is segmented directly. Images become static visual clips. Text uses TRIBE text scoring when available, with a rendered card for comparable visual commentary.</p>
          <p class="callout"><b>Breakout</b>Platform breakout is not a magic creative score. It requires views, likes, comments, shares, saves, age, and author baselines.</p>
        </div>
        <div class="mini-matrix" aria-label="Component legend">
          <div class="matrix-cell"><span>TRIBE</span><strong>Raw</strong></div>
          <div class="matrix-cell"><span>Gate</span><strong>Pass</strong></div>
          <div class="matrix-cell"><span>Proxy</span><strong>Motion</strong></div>
          <div class="matrix-cell"><span>Proxy</span><strong>Affect</strong></div>
        </div>
        </div>
      </aside>
    </section>
  </main>
<script>
const form = document.getElementById('form');
const queue = document.getElementById('queue');
const historyPanel = document.getElementById('historyPanel');
const results = document.getElementById('results');
const runButton = document.getElementById('run');
const filesInput = document.getElementById('files');
const fileSummary = document.getElementById('fileSummary');
let batchSeq = 0;
let activeJobs = 0;
let runFeedbackTimer = null;
const HISTORY_KEY = 'audience-vector-history-v1';
const MAX_HISTORY = 12;

const pct = v => `${Math.round((v || 0) * 100)}%`;
const scorePct = v => `${Math.round(Number(v || 0))}%`;
const z = v => `${v >= 0 ? '+' : ''}${Number(v || 0).toFixed(2)}`;
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
}[char]));
const metric = value => Number(value || 0).toFixed(2);
const scoreRead = value => {
  const score = Number(value || 0);
  if (score >= 80) return 'Strong';
  if (score >= 60) return 'Pass';
  if (score >= 40) return 'Revise';
  return 'Fail';
};
const dimensionMeaning = key => ({
  overall_score: 'Composite creative signal',
  attention_score: 'Immediate pull / salience',
  emotion_score: 'Stakes, contrast, payoff',
  memory_score: 'Sticky idea encoding',
  visual_score: 'Visual field strength',
  language_score: 'Semantic richness',
  cognitive_ease: 'Ease of parsing'
}[key] || 'TRIBE dimension');
const dimensionLabel = key => ({
  overall_score: 'Overall',
  attention_score: 'Attention',
  emotion_score: 'Emotion',
  memory_score: 'Memory',
  visual_score: 'Visual',
  language_score: 'Language',
  cognitive_ease: 'Cognitive ease'
}[key] || key);
function dimensionCard(key, value) {
  const score = Number(value || 0);
  return `
    <div class="dimension">
      <strong>${escapeHtml(dimensionLabel(key))}<span>${scorePct(score)}</span></strong>
      <div class="bar"><i style="--v:${Math.round(score)}"></i></div>
      <small>${escapeHtml(scoreRead(score))} — ${escapeHtml(dimensionMeaning(key))}</small>
    </div>
  `;
}
const affectLabel = key => ({
  happy: 'Happy',
  anger: 'Anger',
  fear: 'Fear',
  sadness: 'Sadness',
  disgust: 'Disgust',
  neutral: 'Neutral'
}[key] || key);
const affectMeaning = key => ({
  happy: 'Warmth, reward, positive payoff',
  anger: 'Conflict, pressure, high-friction intensity',
  fear: 'Uncertainty, threat, suspense, alarm',
  sadness: 'Reflective, low-arousal loss or tenderness',
  disgust: 'Aversion, friction, rejection, unease',
  neutral: 'Low-specificity or balanced affect'
}[key] || 'Affect proxy');
function renderAffect(summary) {
  const profile = summary.affect_profile || {};
  const scores = profile.scores || {};
  const keys = ['happy', 'anger', 'fear', 'sadness', 'disgust', 'neutral'];
  const cards = keys.map(key => `
    <div class="dimension">
      <strong>${escapeHtml(affectLabel(key))}<span>${scorePct(scores[key])}</span></strong>
      <div class="bar"><i style="--v:${Math.round(Number(scores[key] || 0))}"></i></div>
      <small>${escapeHtml(affectMeaning(key))}</small>
    </div>
  `).join('');
  return `
    <div class="breakdown">
      <div class="section-label">Affect proxy</div>
      <div class="score-meta">
        <span class="pill">Label ${escapeHtml(profile.label || 'neutral')}</span>
        <span class="pill">Arousal ${scorePct(profile.arousal_proxy)}</span>
        <span class="pill">Valence ${scorePct(profile.valence_proxy)}</span>
        <span class="pill">NOVA-inspired, not EEG</span>
      </div>
      <p class="read">${escapeHtml(summary.affect_commentary || profile.note || '')}</p>
      <div class="dimension-grid">${cards}</div>
    </div>
  `;
}
function renderBreakdown(summary) {
  const raw = summary.raw_tribe_scores || {};
  const affect = summary.affect_profile || {};
  const keys = ['overall_score', 'attention_score', 'emotion_score', 'memory_score', 'visual_score', 'language_score', 'cognitive_ease'];
  const cards = keys.map(key => dimensionCard(key, raw[key])).join('');
  const checks = (summary.threshold_checks || []).map(check => {
    const read = check.metric === 'Early peak'
      ? (check.passed ? 'present' : 'missing')
      : `${scorePct(check.score)} / floor ${scorePct(check.floor)}`;
    return `
      <div class="check ${check.passed ? 'pass' : 'fail'}">
        <b>${escapeHtml(check.metric)} ${escapeHtml(read)}</b>
        ${escapeHtml(check.meaning || '')}
      </div>
    `;
  }).join('');
  const next = (summary.next_edits || []).slice(0, 3).map(edit => `<span class="pill">${escapeHtml(edit)}</span>`).join('');
  return `
    <div class="breakdown">
      <div class="section-label">Raw TRIBE dimensions</div>
      <div class="dimension-grid">${cards}</div>
      ${checks ? `<div class="section-label">Threshold checks</div><div class="checks">${checks}</div>` : ''}
      ${next ? `<div class="section-label">Next edits</div><div class="score-meta">${next}</div>` : ''}
    </div>
  `;
}

function sourceName(data) {
  const name = String(data.filename || data.source_url || data.source_type || 'candidate');
  return name.length > 110 ? `${name.slice(0, 107)}...` : name;
}

function selectorValue(data) {
  const summary = data.summary || {};
  const raw = summary.raw_tribe_scores || {};
  const mem = Number(summary.mean_mem_percentile || 0) * 100;
  const attention = Number(raw.attention_score || 0);
  const overall = Number(raw.overall_score || 0);
  const ease = Number(raw.cognitive_ease || 0);
  const hookOrMem = raw.hook_score === undefined ? mem : Number(raw.hook_score || 0);
  const failedChecks = (summary.threshold_checks || []).filter(check => check && check.passed === false).length;
  const earlyPeakBonus = (summary.threshold_checks || []).some(check => check.metric === 'Early peak' && check.passed) ? 2 : 0;
  const score = (0.34 * overall) + (0.24 * mem) + (0.18 * attention) + (0.14 * hookOrMem) + (0.10 * ease) + earlyPeakBonus - (failedChecks * 4);
  return Math.max(0, Math.min(100, score));
}

function rankedCandidates(batch) {
  return (batch.results || [])
    .map((data, index) => ({data, index, selector: selectorValue(data)}))
    .sort((a, b) => b.selector - a.selector);
}

function candidateLine(item) {
  const summary = item.data.summary || {};
  const raw = summary.raw_tribe_scores || {};
  const affect = summary.affect_profile || {};
  const mem = Number(summary.mean_mem_percentile || 0) * 100;
  const hook = raw.hook_score === undefined ? mem : Number(raw.hook_score || 0);
  return [
    `${escapeHtml(summary.verdict || 'inconclusive')}`,
    `TRIBE ${scorePct(raw.overall_score)}`,
    `BMD ${scorePct(mem)}`,
    `ATT ${scorePct(raw.attention_score)}`,
    `Hook/BMD ${scorePct(hook)}`,
    `Affect ${escapeHtml(affect.label || 'neutral')}`
  ].join(' · ');
}

function selectorReportMarkdown(batch, ranked = rankedCandidates(batch)) {
  const buckets = activeLearningBuckets(ranked);
  const lines = [
    `# Audience Vector Selector Run ${batch.id}`,
    '',
    `Completed: ${batch.done}/${batch.total}`,
    `Failed: ${batch.failed}`,
    '',
    'Selector formula: 34% TRIBE overall, 24% BMD memorability, 18% raw TRIBE attention, 14% hook or BMD early signal, 10% cognitive ease, with small penalties for failed threshold gates.',
    '',
    '| Rank | Candidate | Selector | Verdict | TRIBE | BMD | Attention | Affect |',
    '|---:|---|---:|---|---:|---:|---:|---|'
  ];
  ranked.forEach((item, index) => {
    const data = item.data;
    const summary = data.summary || {};
    const raw = summary.raw_tribe_scores || {};
    const affect = summary.affect_profile || {};
    const mem = Number(summary.mean_mem_percentile || 0) * 100;
    lines.push(`| ${index + 1} | ${sourceName(data).replaceAll('|', '\\|')} | ${Math.round(item.selector)} | ${summary.verdict || 'inconclusive'} | ${Math.round(Number(raw.overall_score || 0))} | ${Math.round(mem)} | ${Math.round(Number(raw.attention_score || 0))} | ${affect.label || 'neutral'} |`);
  });
  if (ranked[0]) {
    const best = ranked[0].data;
    const bestSummary = best.summary || {};
    lines.push('', '## Winner Read', '', bestSummary.plain_english_summary || bestSummary.commentary || 'No plain-English summary returned.');
    const edits = (bestSummary.next_edits || []).slice(0, 5);
    if (edits.length) {
      lines.push('', '## Next Edits', '');
      edits.forEach(edit => lines.push(`- ${edit}`));
    }
  }
  if (buckets.winners.length || buckets.ambiguous.length || buckets.failures.length) {
    lines.push('', '## Active-Learning Buckets', '');
    if (buckets.winners.length) {
      lines.push('### Exploit / ship candidates', '');
      buckets.winners.forEach(item => lines.push(`- ${sourceName(item.data)} — selector ${Math.round(item.selector)}, ${plainCandidateLine(item)}`));
    }
    if (buckets.ambiguous.length) {
      lines.push('', '### Ambiguous candidates to label', '');
      buckets.ambiguous.forEach(item => lines.push(`- ${sourceName(item.data)} — selector ${Math.round(item.selector)}, ${plainCandidateLine(item)}`));
    }
    if (buckets.failures.length) {
      lines.push('', '### Failure cases to study', '');
      buckets.failures.forEach(item => lines.push(`- ${sourceName(item.data)} — selector ${Math.round(item.selector)}, ${plainCandidateLine(item)}`));
    }
  }
  lines.push('', 'Note: this is a product selector report over predictive scores. It is not a brain scan or a claim of causal neural measurement.');
  return lines.join('\\n');
}

function plainCandidateLine(item) {
  const summary = item.data.summary || {};
  const raw = summary.raw_tribe_scores || {};
  const affect = summary.affect_profile || {};
  const mem = Number(summary.mean_mem_percentile || 0) * 100;
  return [
    `verdict ${summary.verdict || 'inconclusive'}`,
    `TRIBE ${Math.round(Number(raw.overall_score || 0))}`,
    `BMD ${Math.round(mem)}`,
    `attention ${Math.round(Number(raw.attention_score || 0))}`,
    `affect ${affect.label || 'neutral'}`
  ].join(', ');
}

function activeLearningBuckets(ranked) {
  const top = ranked[0]?.selector || 0;
  const winners = ranked.filter(item => item.selector >= 70 || (top >= 55 && top - item.selector <= 3)).slice(0, 3);
  const ambiguous = ranked
    .filter(item => item.selector >= 40 && item.selector < 70 && top - item.selector <= 10)
    .filter(item => !winners.includes(item))
    .slice(0, 4);
  const failures = ranked
    .filter(item => {
      const verdict = (item.data.summary || {}).verdict || '';
      return item.selector < 40 || verdict === 'revise';
    })
    .slice(-4)
    .reverse();
  return {winners, ambiguous, failures};
}

function activeLearningMarkdown(batch, ranked = rankedCandidates(batch)) {
  const buckets = activeLearningBuckets(ranked);
  const lines = [
    `# Active-Learning Export - Run ${batch.id}`,
    '',
    'Use this to decide what to generate next, what to label with humans, and what failure modes to inspect.',
    ''
  ];
  const sections = [
    ['Exploit / ship candidates', buckets.winners, 'High selector score or near-top candidate. Use these as product-selected outputs or positive examples.'],
    ['Ambiguous candidates to label', buckets.ambiguous, 'Close enough to matter. Send these to humans or rerun with stronger baselines.'],
    ['Failure cases to study', buckets.failures, 'Weak or revise-labeled candidates. Use these to find drift, weak hooks, or proxy failures.']
  ];
  sections.forEach(([title, items, note]) => {
    lines.push(`## ${title}`, '', note, '');
    if (!items.length) {
      lines.push('- None in this run.', '');
      return;
    }
    items.forEach(item => {
      const summary = item.data.summary || {};
      const edits = (summary.next_edits || []).slice(0, 2).join('; ');
      lines.push(`- ${sourceName(item.data)} — selector ${Math.round(item.selector)}; ${plainCandidateLine(item)}${edits ? `; next edits: ${edits}` : ''}`);
    });
    lines.push('');
  });
  return lines.join('\\n');
}

function compactResult(data) {
  const summary = data.summary || {};
  const raw = summary.raw_tribe_scores || {};
  const affect = summary.affect_profile || {};
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    saved_at: new Date().toISOString(),
    filename: data.filename || 'analysis',
    source_type: data.source_type || 'unknown',
    duration_s: data.duration_s || 0,
    n_segments: data.n_segments || 0,
    verdict: summary.verdict || 'inconclusive',
    summary: summary.plain_english_summary || summary.commentary || '',
    overall_score: Number(raw.overall_score || 0),
    attention_score: Number(raw.attention_score || 0),
    memory_score: Number(raw.memory_score || 0),
    bmd_memorability: Number(summary.mean_mem_percentile || 0) * 100,
    hook_score: raw.hook_score === undefined ? null : Number(raw.hook_score || 0),
    affect_label: affect.label || 'neutral',
    next_edits: (summary.next_edits || []).slice(0, 4),
    top_windows: (data.segments || []).slice(0, 3).map(row => ({
      timestamp: row.timestamp,
      mem_percentile: row.mem_percentile,
      natural_read: row.natural_read
    }))
  };
}

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    return [];
  }
}

function saveHistoryEntry(data) {
  const next = [compactResult(data), ...loadHistory()].slice(0, MAX_HISTORY);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
    renderHistory();
  } catch (err) {
    console.warn('Could not save local history', err);
  }
}

function historyEntryMarkdown(entry) {
  const lines = [
    `# Audience Vector Snapshot - ${entry.filename}`,
    '',
    `Saved: ${entry.saved_at}`,
    `Source: ${entry.source_type}`,
    `Verdict: ${entry.verdict}`,
    `Overall: ${Math.round(entry.overall_score)} / 100`,
    `Attention: ${Math.round(entry.attention_score)} / 100`,
    `BMD memorability: ${Math.round(entry.bmd_memorability)} / 100`,
    `Affect: ${entry.affect_label}`,
    '',
    '## Read',
    '',
    entry.summary || 'No summary returned.'
  ];
  if (entry.next_edits && entry.next_edits.length) {
    lines.push('', '## Next Edits', '');
    entry.next_edits.forEach(edit => lines.push(`- ${edit}`));
  }
  if (entry.top_windows && entry.top_windows.length) {
    lines.push('', '## First Windows', '');
    entry.top_windows.forEach(row => lines.push(`- ${row.timestamp}: BMD ${pct(row.mem_percentile)} — ${row.natural_read || ''}`));
  }
  return lines.join('\\n');
}

function renderHistory() {
  if (!historyPanel) return;
  const history = loadHistory();
  if (!history.length) {
    historyPanel.innerHTML = `
      <div class="history-head"><span>Local history</span></div>
      <div class="selector-empty">
        <b>No saved runs yet</b>
        <span>Completed analyses are remembered locally in this browser for quick snapshot exports.</span>
      </div>
    `;
    return;
  }
  const cards = history.map(entry => `
    <div class="history-card" data-id="${escapeHtml(entry.id)}">
      <strong>${escapeHtml(entry.filename)}</strong>
      <small>${escapeHtml(entry.verdict)} · TRIBE ${scorePct(entry.overall_score)} · BMD ${scorePct(entry.bmd_memorability)} · Affect ${escapeHtml(entry.affect_label || 'neutral')}</small>
      <div class="history-actions">
        <button type="button" data-action="copy">Copy</button>
        <button type="button" data-action="download">Download</button>
      </div>
    </div>
  `).join('');
  historyPanel.innerHTML = `
    <div class="history-head">
      <span>Local history</span>
      <button type="button" data-action="clear-history">Clear</button>
    </div>
    <div class="history-list">${cards}</div>
  `;
  historyPanel.querySelector('[data-action="clear-history"]').addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });
  historyPanel.querySelectorAll('.history-card').forEach(card => {
    const entry = history.find(item => item.id === card.dataset.id);
    if (!entry) return;
    card.querySelector('[data-action="copy"]').addEventListener('click', event => copyReport(event.currentTarget, historyEntryMarkdown(entry)));
    card.querySelector('[data-action="download"]').addEventListener('click', () => downloadText(`${String(entry.filename || 'snapshot').replace(/[^a-z0-9_.-]+/gi, '_')}.snapshot.md`, historyEntryMarkdown(entry), 'text/markdown'));
  });
}

function downloadText(filename, text, type = 'text/plain') {
  const blob = new Blob([text], {type});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function copyText(button, text, fallbackFilename = 'audience-vector-report.md') {
  const original = button.textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied';
  } catch (err) {
    downloadText(fallbackFilename, text, 'text/markdown');
    button.textContent = 'Downloaded';
  }
  setTimeout(() => { button.textContent = original; }, 900);
}

async function copyReport(button, text) {
  copyText(button, text, 'audience-vector-selector-report.md');
}

function sharePayload(batch, ranked = rankedCandidates(batch)) {
  return {
    schema: 'audience-vector-shared-run-v1',
    id: batch.id,
    saved_at: new Date().toISOString(),
    total: batch.total,
    done: batch.done,
    failed: batch.failed,
    selector_formula: '34% TRIBE overall, 24% BMD memorability, 18% raw TRIBE attention, 14% hook or BMD early signal, 10% cognitive ease, with penalties for failed gates.',
    ranked: ranked.map((item, index) => ({
      rank: index + 1,
      selector_score: item.selector,
      payload: item.data
    }))
  };
}

async function saveShareRun(batch, button) {
  const original = button.textContent;
  button.textContent = 'Saving...';
  try {
    const response = await fetch('/api/runs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(sharePayload(batch))
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.error || 'share failed');
    const url = `${window.location.origin}${window.location.pathname}?run=${encodeURIComponent(payload.id)}`;
    await navigator.clipboard.writeText(url);
    button.textContent = 'Link copied';
  } catch (err) {
    console.warn('Could not create share link', err);
    button.textContent = 'Share failed';
  }
  setTimeout(() => { button.textContent = original; }, 1100);
}

async function hydrateSharedRun() {
  const runId = new URLSearchParams(window.location.search).get('run');
  if (!runId) return;
  const batch = createBatch(0);
  batch.meta.textContent = 'loading shared run';
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    const shared = await response.json();
    if (!response.ok) throw new Error(shared.detail || shared.error || 'shared run failed');
    const rows = Array.isArray(shared.ranked) ? shared.ranked : [];
    batch.total = Number(shared.total || rows.length);
    batch.done = Number(shared.done || rows.length);
    batch.failed = Number(shared.failed || 0);
    batch.results = rows.map(row => row.payload).filter(Boolean);
    batch.jobs.innerHTML = '';
    batch.results.slice().reverse().forEach(result => {
      const job = jobRow(sourceName(result), 'loaded shared result', batch);
      job.el.classList.add('done');
    });
    updateBatch(batch);
    batch.results.slice().reverse().forEach(result => renderResult(result));
  } catch (err) {
    batch.failed = 1;
    batch.total = 1;
    batch.jobs.innerHTML = '';
    const job = jobRow(`shared run ${runId}`, 'failed to load shared run', batch);
    job.el.classList.add('failed');
    updateBatch(batch);
  }
}

function renderBatchSelector(batch) {
  if (!batch.selector) return;
  const ranked = rankedCandidates(batch);
  const complete = batch.done + batch.failed;
  if (!ranked.length) {
    batch.selector.innerHTML = `
      <div class="selector-empty">
        <b>Batch selector</b>
        <span>Waiting for scored candidates. Add two or more files, URLs, or text blocks to get a ranked active-learning slate.</span>
      </div>
    `;
    return;
  }
  const best = ranked[0];
  const enough = ranked.length >= 2;
  const rows = ranked.slice(0, 6).map((item, index) => `
    <div class="selector-row">
      <div class="selector-rank">#${index + 1}</div>
      <div class="selector-name">
        ${escapeHtml(sourceName(item.data))}
        <small>${candidateLine(item)}</small>
      </div>
      <div class="selector-score"><small>Score</small><em>${Math.round(item.selector)}</em></div>
    </div>
  `).join('');
  batch.selector.innerHTML = `
    <div class="selector-card">
      <div class="selector-top">
        <div>
          <b>${enough ? 'Selector winner' : 'Single candidate read'}</b>
          <strong>${escapeHtml(sourceName(best.data))}</strong>
          <span>${escapeHtml(enough ? 'Ranked for generation selection and active-learning review.' : 'Add another candidate to compare variants inside this run.')} ${complete}/${batch.total} complete.</span>
        </div>
        <div class="selector-score"><small>Selector</small><em>${Math.round(best.selector)}</em></div>
      </div>
      <div class="selector-actions">
        <button type="button" class="copy-selector">Copy report</button>
        <button type="button" class="share-selector">Share link</button>
        <button type="button" class="download-selector">Download report</button>
        <button type="button" class="active-selector">Active-learning</button>
        <button type="button" class="download-json">Download JSON</button>
      </div>
      <div class="selector-ranking">${rows}</div>
    </div>
  `;
  const report = selectorReportMarkdown(batch, ranked);
  const activeReport = activeLearningMarkdown(batch, ranked);
  batch.selector.querySelector('.copy-selector').addEventListener('click', event => copyReport(event.currentTarget, report));
  batch.selector.querySelector('.share-selector').addEventListener('click', event => saveShareRun(batch, event.currentTarget));
  batch.selector.querySelector('.download-selector').addEventListener('click', () => downloadText(`audience-vector-run-${batch.id}.md`, report, 'text/markdown'));
  batch.selector.querySelector('.active-selector').addEventListener('click', () => downloadText(`audience-vector-active-learning-${batch.id}.md`, activeReport, 'text/markdown'));
  batch.selector.querySelector('.download-json').addEventListener('click', () => downloadText(`audience-vector-run-${batch.id}.json`, JSON.stringify({id: batch.id, total: batch.total, done: batch.done, failed: batch.failed, ranked: ranked.map(item => ({selector_score: item.selector, payload: item.data}))}, null, 2), 'application/json'));
}

filesInput.addEventListener('change', () => {
  const files = [...filesInput.files];
  if (!files.length) {
    fileSummary.textContent = 'MP4, MOV, PNG, JPG, TXT, MD';
    return;
  }
  const count = files.length === 1 ? files[0].name : `${files.length} files selected`;
  fileSummary.textContent = count;
});

function updateRunButton(message) {
  if (runFeedbackTimer) {
    clearTimeout(runFeedbackTimer);
    runFeedbackTimer = null;
  }
  if (message) {
    runButton.textContent = message;
    runFeedbackTimer = setTimeout(() => updateRunButton(), 800);
    return;
  }
  runButton.textContent = activeJobs ? `Analyze Media (${activeJobs} active)` : 'Analyze Media';
}

function createBatch(total) {
  batchSeq += 1;
  const el = document.createElement('div');
  el.className = 'batch';
  el.innerHTML = `
    <div class="batch-head">
      <strong>Run ${batchSeq}</strong>
      <span class="batch-meta">0/${total} complete</span>
    </div>
    <div class="batch-jobs"></div>
    <div class="batch-selector"></div>
  `;
  queue.prepend(el);
  const batch = {
    id: batchSeq,
    el,
    jobs: el.querySelector('.batch-jobs'),
    selector: el.querySelector('.batch-selector'),
    meta: el.querySelector('.batch-meta'),
    total,
    done: 0,
    failed: 0,
    results: []
  };
  renderBatchSelector(batch);
  return batch;
}

function updateBatch(batch) {
  const complete = batch.done + batch.failed;
  batch.meta.textContent = `${complete}/${batch.total} complete${batch.failed ? `, ${batch.failed} failed` : ''}`;
  batch.el.classList.toggle('done', complete === batch.total && batch.failed === 0);
  batch.el.classList.toggle('failed', complete === batch.total && batch.failed > 0);
  renderBatchSelector(batch);
}

function jobRow(name, text, batch) {
  const el = document.createElement('div');
  el.className = 'job';
  el.innerHTML = `<strong>${escapeHtml(name)}</strong><span class="status">${escapeHtml(text)}</span>`;
  batch.jobs.prepend(el);
  return {
    el,
    status: el.querySelector('.status')
  };
}

function finishJob(job, batch, state, text) {
  job.el.classList.add(state);
  job.status.textContent = text;
  if (state === 'done') {
    batch.done += 1;
  } else {
    batch.failed += 1;
  }
  activeJobs = Math.max(0, activeJobs - 1);
  updateBatch(batch);
  updateRunButton();
}

async function submitAnalysisJob(label, progressText, fillBody, batch, segmentSeconds) {
  const job = jobRow(label, progressText, batch);
  const body = new FormData();
  fillBody(body);
  body.append('segment_seconds', segmentSeconds);
  try {
    const response = await fetch('/api/analyze', {method: 'POST', body});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.error || 'analysis failed');
    batch.results.push(payload);
    finishJob(job, batch, 'done', 'done');
    renderResult(payload);
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    finishJob(job, batch, 'failed', `failed: ${message}`);
  }
}

function clearQueuedInputs() {
  filesInput.value = '';
  fileSummary.textContent = 'MP4, MOV, PNG, JPG, TXT, MD';
  document.getElementById('urls').value = '';
  document.getElementById('texts').value = '';
}

function renderResult(data) {
  const summary = data.summary || {};
  const raw = summary.raw_tribe_scores || {};
  const affect = summary.affect_profile || {};
  const notes = (data.notes || []).slice(0, 3).map(note => `<span class="pill">${escapeHtml(note)}</span>`).join('');
  const article = document.createElement('article');
  article.className = 'panel result';
  const rows = (data.segments || []).map(row => {
    const rowTribe = row.tribe_scores || {};
    const rowAffect = row.affect_proxy || {};
    const tribeAttention = Number(rowTribe.attention_score || 0);
    const tribeOverall = Number(rowTribe.overall_score || 0);
    return `
    <div class="row">
      <div class="time"><span>Window</span>${escapeHtml(row.timestamp)}</div>
      <div class="bars">
        <div class="metric">
          <span class="metric-label">BMD</span>
          <div class="bar"><i style="--v:${Math.round(row.mem_percentile * 100)}"></i></div>
          <small>${pct(row.mem_percentile)} / ${z(row.mem_z)}</small>
        </div>
        <div class="metric">
          <span class="metric-label">Att</span>
          <div class="bar att"><i style="--v:${Math.round(tribeAttention)}"></i></div>
          <small>${scorePct(tribeAttention)} TRIBE</small>
        </div>
      </div>
      <div class="read">
        ${escapeHtml(row.natural_read || '')}
        <div class="score-meta">
          <span class="pill">TRIBE overall ${scorePct(tribeOverall)}</span>
          <span class="pill">Density ${z(row.visual_density_z)}</span>
          <span class="pill">Motion ${metric(row.frame_motion_delta_mean)}</span>
          <span class="pill">Edges ${metric(row.edge_density_mean)}</span>
          <span class="pill">Affect ${escapeHtml(rowAffect.label || 'neutral')}</span>
          <span class="pill">Audience axis ${pct(row.legacy_persona_attention_percentile ?? row.attention_percentile)}</span>
        </div>
        <div class="palette">${(row.palette || []).slice(0,6).map(c => `<span class="swatch" title="${escapeHtml(c.hex)}" style="--c:${escapeHtml(c.hex)}"></span>`).join('')}</div>
      </div>
    </div>
  `}).join('');
  article.innerHTML = `
    <div class="result-head">
      <div>
        <div class="source-kicker">${escapeHtml(data.source_type || 'video')} x ${Math.round(data.duration_s || 0)}s x ${data.n_segments || 0} windows</div>
        <h2>${escapeHtml(data.filename)}</h2>
        <p>${escapeHtml(summary.plain_english_summary || summary.commentary || '')}</p>
        <div class="score-meta">${notes}</div>
      </div>
      <button type="button" class="download">Download JSON</button>
    </div>
    <div class="numbers">
      <div class="num"><span>Verdict</span><b>${escapeHtml(summary.verdict || 'inconclusive')}</b><small>${escapeHtml(scoreRead(raw.overall_score))}</small></div>
      <div class="num"><span>Overall</span><b>${scorePct(raw.overall_score)}</b><small>TRIBE composite</small></div>
      <div class="num"><span>Attention</span><b>${scorePct(raw.attention_score)}</b><small>Raw TRIBE, not persona-derived</small></div>
      <div class="num"><span>Affect</span><b>${escapeHtml(affect.label || 'neutral')}</b><small>${scorePct(affect.arousal_proxy)} arousal</small></div>
      <div class="num"><span>${raw.hook_score === undefined ? 'BMD mem' : 'Hook'}</span><b>${raw.hook_score === undefined ? pct(summary.mean_mem_percentile) : scorePct(raw.hook_score)}</b><small>${raw.hook_score === undefined ? 'Human-label axis' : 'First 2 seconds'}</small></div>
    </div>
    ${renderBreakdown(summary)}
    ${renderAffect(summary)}
    <div class="timeline">${rows}</div>
  `;
  article.querySelector('.download').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${String(data.filename || 'video').replace(/[^a-z0-9_.-]+/gi, '_')}.analysis.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
  if (results.querySelector('.empty')) results.innerHTML = '';
  results.prepend(article);
  saveHistoryEntry(data);
}

renderHistory();
hydrateSharedRun();

form.addEventListener('submit', async event => {
  event.preventDefault();
  const files = [...filesInput.files];
  const urls = document.getElementById('urls').value
    .split(/\\n+/)
    .map(v => v.trim())
    .filter(Boolean);
  const texts = document.getElementById('texts').value
    .split(/\\n\\s*---\\s*\\n|\\n\\s*\\n+/)
    .map(v => v.trim())
    .filter(Boolean);
  if (!files.length && !urls.length && !texts.length) return;
  const total = files.length + urls.length + texts.length;
  const segmentSeconds = document.getElementById('segment').value;
  const batch = createBatch(total);
  activeJobs += total;
  updateBatch(batch);
  updateRunButton('Added to queue');
  clearQueuedInputs();

  const fileJobs = files.map(file => submitAnalysisJob(
    file.name,
    'uploading and analyzing...',
    body => body.append('file', file),
    batch,
    segmentSeconds
  ));
  const urlJobs = urls.map(url => submitAnalysisJob(
    url,
    'resolving media and analyzing...',
    body => body.append('url', url),
    batch,
    segmentSeconds
  ));
  const textJobs = texts.map((text, index) => {
    const label = text.length > 64 ? `${text.slice(0, 64)}...` : text;
    return submitAnalysisJob(
      `text ${index + 1}: ${label}`,
      'scoring copy...',
      body => body.append('text', text),
      batch,
      segmentSeconds
    );
  });
  Promise.all([...fileJobs, ...urlJobs, ...textJobs]).then(() => updateBatch(batch));
});
</script>
</body>
</html>"""


@app.function(
    image=base_image,
    timeout=60 * 60,
    secrets=env_secrets,
    volumes={str(ANALYZER_RUNS_MOUNT): analyzer_runs_volume},
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="video-analyzer")
def create_video_analyzer_app() -> Any:  # noqa: C901
    api = FastAPI(title="Audience Vector Video Analyzer")
    image_headers = {"Cache-Control": "public, max-age=86400"}

    @api.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _html()

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/favicon.svg")
    async def favicon_svg() -> Response:
        return Response(
            _favicon_svg(),
            media_type="image/svg+xml",
            headers=image_headers,
        )

    @api.get("/favicon.ico")
    async def favicon_ico() -> Response:
        return Response(
            _favicon_ico(),
            media_type="image/x-icon",
            headers=image_headers,
        )

    @api.get("/apple-touch-icon.png")
    async def apple_touch_icon() -> Response:
        return Response(
            _icon_png(180),
            media_type="image/png",
            headers=image_headers,
        )

    @api.get("/og-image.png")
    async def og_image() -> Response:
        return Response(
            _og_image_png(),
            media_type="image/png",
            headers=image_headers,
        )

    @api.post("/api/runs")
    async def save_run(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        run_id = uuid.uuid4().hex[:16]
        payload = dict(payload)
        payload["id"] = run_id
        payload["saved_at"] = payload.get("saved_at") or datetime.now(UTC).isoformat()
        runs_dir = ANALYZER_RUNS_MOUNT / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_path = runs_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        analyzer_runs_volume.commit()
        return JSONResponse(
            {
                "id": run_id,
                "url": f"{PUBLIC_SITE_URL}/?run={run_id}",
            }
        )

    @api.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        if not re.fullmatch(r"[a-f0-9]{16}", run_id):
            raise HTTPException(status_code=404, detail="Shared run not found.")
        analyzer_runs_volume.reload()
        run_path = ANALYZER_RUNS_MOUNT / "runs" / f"{run_id}.json"
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Shared run not found.")
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Shared run payload is corrupted.",
            ) from exc
        return JSONResponse(payload)

    @api.post("/api/analyze")
    async def analyze(
        file: UploadFile | None = File(default=None),
        url: str = Form(default=""),
        text: str = Form(default=""),
        segment_seconds: float = Form(default=DEFAULT_SEGMENT_SECONDS),
    ) -> JSONResponse:
        job_id = uuid.uuid4().hex[:12]
        requested_url = url.strip()
        requested_text = text.strip()

        with tempfile.TemporaryDirectory(prefix=f"video-analyzer-{job_id}-") as tmp:
            tmp_path = Path(tmp)
            source_url: str | None = None

            try:
                if file is not None and file.filename:
                    source_path, filename, media_kind = await _save_upload_to_tmp(
                        file,
                        tmp_path,
                    )
                    payload = await _analyze_media_source(
                        source_path=source_path,
                        media_kind=media_kind,
                        job_id=job_id,
                        filename=filename,
                        source_type="upload",
                        segment_seconds=segment_seconds,
                        tmp_path=tmp_path,
                    )
                elif requested_url:
                    source_url = requested_url
                    source_path, host, media_kind = _download_url_to_source(
                        requested_url,
                        tmp_path,
                    )
                    payload = await _analyze_media_source(
                        source_path=source_path,
                        media_kind=media_kind,
                        job_id=job_id,
                        filename=host,
                        source_type="url",
                        segment_seconds=segment_seconds,
                        tmp_path=tmp_path,
                        source_url=source_url,
                    )
                elif requested_text:
                    payload = await _analyze_text_source(
                        text=requested_text,
                        job_id=job_id,
                        filename="pasted text",
                        source_type="pasted_text",
                        segment_seconds=segment_seconds,
                        tmp_path=tmp_path,
                    )
                else:
                    raise ValueError(
                        "Provide a media upload, a public URL, or pasted text."
                    )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except subprocess.CalledProcessError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Media processing failed: {exc.stderr[-500:]}",
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {type(exc).__name__}: {exc}",
                ) from exc

        return JSONResponse(payload)

    return api
