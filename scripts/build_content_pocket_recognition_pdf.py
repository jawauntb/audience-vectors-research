"""Build an exportable PDF for the content-pocket recognition-memory draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

INK = "#18202f"
MUTED = "#536174"
GRID = "#d9e2ec"
BG = "#fffdf8"
GREEN = "#2f855a"
BLUE = "#2b6cb0"
GRAY = "#718096"
PALE_GREEN = "#edf7f0"

PAGE_W = 1650
PAGE_H = 2550
MARGIN = 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--out-pdf", required=True, type=Path)
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> Any:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default(size=size)


FONT = {
    "title": load_font(56, bold=True),
    "subtitle": load_font(28),
    "h1": load_font(42, bold=True),
    "h2": load_font(30, bold=True),
    "body": load_font(25),
    "body_bold": load_font(25, bold=True),
    "small": load_font(21),
    "small_bold": load_font(21, bold=True),
    "mono": load_font(22),
}


def new_page() -> Image.Image:
    return Image.new("RGB", (PAGE_W, PAGE_H), BG)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return (int(box[2] - box[0]), int(box[3] - box[1]))


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    font: Any,
    fill: str = INK,
    width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_size(draw, candidate, font)[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in rows}


def draw_footer(draw: ImageDraw.ImageDraw, page: int) -> None:
    draw.line((MARGIN, PAGE_H - 120, PAGE_W - MARGIN, PAGE_H - 120), fill=GRID, width=2)
    draw.text(
        (MARGIN, PAGE_H - 88),
        "Content-pocket recognition-memory validation | aggregate-only export",
        font=FONT["small"],
        fill=MUTED,
    )
    draw.text(
        (PAGE_W - MARGIN - 90, PAGE_H - 88),
        f"{page}",
        font=FONT["small_bold"],
        fill=MUTED,
    )


def cover_page(summary: dict[str, Any]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    draw.rounded_rectangle((MARGIN, 150, PAGE_W - MARGIN, 510), radius=34, fill="#f0f7f3")
    draw.text(
        (MARGIN + 50, 210),
        "Generated-Video Content Pockets Predict",
        font=FONT["title"],
        fill=INK,
    )
    draw.text(
        (MARGIN + 50, 280),
        "Delayed Human Recognition Memory",
        font=FONT["title"],
        fill=INK,
    )
    draw_wrapped(
        draw,
        (
            "Wave 2 Prolific old-vs-lure recognition-memory validation for the "
            "primary orange-flowers and hanging-clothes content-pocket packet."
        ),
        (MARGIN + 50, 370),
        font=FONT["subtitle"],
        fill=MUTED,
        width=PAGE_W - 2 * MARGIN - 100,
        line_gap=10,
    )

    contrast = summary["paired_primary_vs_hard_negative_no_media_errors"]
    primary = by_name(summary["no_media_error_summaries"])["primary_positive"]
    hard = by_name(summary["no_media_error_summaries"])["hard_negative_control"]
    cards = [
        ("Complete Wave 2 payloads", str(summary["complete_matched_participants"]), GRAY),
        ("Primary recognition", pct(primary["accuracy"]), GREEN),
        ("Hard-control recognition", pct(hard["accuracy"]), BLUE),
        ("Paired lift", pp(contrast["mean_difference"]), GREEN),
    ]
    x = MARGIN
    y = 670
    card_w = 320
    for label, value, color in cards:
        draw.rounded_rectangle((x, y, x + card_w, y + 210), radius=22, fill="white", outline=GRID, width=2)
        draw.text((x + 28, y + 32), label, font=FONT["small_bold"], fill=MUTED)
        draw.text((x + 28, y + 92), value, font=FONT["h1"], fill=color)
        x += card_w + 28

    y = 1030
    draw.text((MARGIN, y), "Claim Boundary", font=FONT["h1"], fill=INK)
    y += 72
    claim = (
        "Supported: a narrow delayed old-vs-lure recognition-memory advantage "
        "for the pooled orange/hanging primary packet. Not supported: broad "
        "human memorability, measured-BMD/fMRI grounding, prompt-conditioned "
        "generation control, or general BO/SVD optimization superiority."
    )
    y = draw_wrapped(draw, claim, (MARGIN, y), font=FONT["body"], fill=INK, width=PAGE_W - 2 * MARGIN, line_gap=12)
    y += 52
    caveat = (
        "Important caveats: Wave 1 webhook retention was partial, although the "
        "retained overlap had zero deterministic form mismatches; the original "
        "large-sample final-confirmation plan remains stricter than this Wave 2 "
        "draft result."
    )
    draw_wrapped(draw, caveat, (MARGIN, y), font=FONT["body"], fill=MUTED, width=PAGE_W - 2 * MARGIN, line_gap=12)
    draw_footer(draw, 1)
    return page


def accuracy_page(summary: dict[str, Any]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, 120), "Delayed Recognition Accuracy By Content Arm", font=FONT["h1"], fill=INK)
    draw.text(
        (MARGIN, 175),
        "Wilson 95% intervals; media-error trials excluded.",
        font=FONT["subtitle"],
        fill=MUTED,
    )
    rows_by_name = by_name(summary["no_media_error_summaries"])
    rows = [
        ("Orange flowers", rows_by_name["arm:orange_flowers"], GREEN),
        ("Hanging clothes", rows_by_name["arm:hanging_clothes"], GREEN),
        ("Aerial beach", rows_by_name["arm:aerial_beach"], BLUE),
        ("City street", rows_by_name["arm:city_street"], BLUE),
        ("Storm beach", rows_by_name["arm:storm_beach"], BLUE),
        ("Fillers", rows_by_name["unrelated_filler"], GRAY),
    ]
    plot_x = 560
    plot_y = 370
    plot_w = 820
    row_gap = 205

    def x(value: float) -> int:
        return int(plot_x + (value - 0.5) / 0.5 * plot_w)

    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        tx = x(tick)
        draw.line((tx, plot_y - 55, tx, plot_y + row_gap * (len(rows) - 1) + 60), fill=GRID, width=2)
        draw.text((tx - 24, plot_y + row_gap * (len(rows) - 1) + 90), f"{int(tick * 100)}%", font=FONT["small"], fill=MUTED)
    draw.line((x(0.5), plot_y - 65, x(0.5), plot_y + row_gap * (len(rows) - 1) + 70), fill="#9fb0c4", width=4)
    draw.text((x(0.5) + 10, plot_y - 96), "chance", font=FONT["small"], fill=MUTED)

    for i, (label, row, color) in enumerate(rows):
        y = plot_y + i * row_gap
        draw.text((MARGIN, y - 18), label, font=FONT["body_bold"], fill=INK)
        draw.line((x(row["wilson95_low"]), y, x(row["wilson95_high"]), y), fill=color, width=15)
        draw.ellipse((x(row["accuracy"]) - 20, y - 20, x(row["accuracy"]) + 20, y + 20), fill=color)
        draw.text(
            (min(x(row["accuracy"]) + 35, PAGE_W - MARGIN - 230), y - 20),
            f"{pct(row['accuracy'])} ({row['correct']}/{row['n']})",
            font=FONT["small_bold"],
            fill=INK,
        )

    draw_footer(draw, 2)
    return page


def pooled_contrast_page(summary: dict[str, Any]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, 120), "Primary Pockets Exceed Hard Controls", font=FONT["h1"], fill=INK)
    draw.text((MARGIN, 175), "Participant-level paired contrast after media-error exclusion.", font=FONT["subtitle"], fill=MUTED)
    rows = by_name(summary["no_media_error_summaries"])
    primary = rows["primary_positive"]
    hard = rows["hard_negative_control"]
    contrast = summary["paired_primary_vs_hard_negative_no_media_errors"]

    base_y = 1420
    chart_x = MARGIN + 100
    bar_w = 270
    gap = 210
    scale_h = 900

    for tick in [0.5, 0.7, 0.9]:
        y = int(base_y - tick * scale_h)
        draw.line((chart_x - 60, y, chart_x + 2 * bar_w + gap + 70, y), fill=GRID, width=2)
        draw.text((chart_x - 120, y - 18), f"{int(tick * 100)}%", font=FONT["small"], fill=MUTED)

    for i, (label, row, color) in enumerate(
        [("Primary positives", primary, GREEN), ("Hard controls", hard, BLUE)]
    ):
        x = chart_x + i * (bar_w + gap)
        h = int(row["accuracy"] * scale_h)
        draw.rounded_rectangle((x, base_y - h, x + bar_w, base_y), radius=24, fill=color)
        draw.text((x + 48, base_y - h - 62), pct(row["accuracy"]), font=FONT["h1"], fill=INK)
        draw.text((x + 10, base_y + 45), label, font=FONT["body_bold"], fill=INK)

    callout = (930, 550, PAGE_W - MARGIN, 1120)
    draw.rounded_rectangle(callout, radius=34, fill=PALE_GREEN, outline="#8ee0aa", width=3)
    draw.text((callout[0] + 54, callout[1] + 70), "Paired lift", font=FONT["h1"], fill=INK)
    draw.text((callout[0] + 54, callout[1] + 160), pp(contrast["mean_difference"]), font=load_font(68, bold=True), fill=GREEN)
    lines = [
        f"Bootstrap 95% CI: {pp(contrast['bootstrap95_low'])} to {pp(contrast['bootstrap95_high'])}",
        f"Sign-flip p = {contrast['sign_flip_permutation_p']:.4g}",
        f"Paired participants: {contrast['participants']}",
    ]
    y = callout[1] + 290
    for line in lines:
        draw.text((callout[0] + 54, y), line, font=FONT["body"], fill=MUTED)
        y += 62

    draw_footer(draw, 3)
    return page


def individual_contrast_page(summary: dict[str, Any]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, 120), "Pocket-Specific Lift Vs Hard Controls", font=FONT["h1"], fill=INK)
    draw_wrapped(
        draw,
        "The packet-level claim is stronger than the individual orange-flowers claim.",
        (MARGIN, 180),
        font=FONT["subtitle"],
        fill=MUTED,
        width=PAGE_W - 2 * MARGIN,
    )
    contrasts = summary["arm_contrasts_no_media_errors"]
    rows = [
        ("Orange flowers", contrasts["orange_flowers"]),
        ("Hanging clothes", contrasts["hanging_clothes"]),
    ]
    x0 = 470
    y0 = 560
    plot_w = 850
    row_gap = 330
    min_v = -0.05
    max_v = 0.25

    def x(value: float) -> int:
        return int(x0 + (value - min_v) / (max_v - min_v) * plot_w)

    for tick in [-0.05, 0.0, 0.1, 0.2, 0.25]:
        tx = x(tick)
        draw.line((tx, y0 - 115, tx, y0 + row_gap + 105), fill=GRID, width=2)
        draw.text((tx - 36, y0 + row_gap + 140), pp(tick), font=FONT["small"], fill=MUTED)
    draw.line((x(0), y0 - 125, x(0), y0 + row_gap + 115), fill="#9fb0c4", width=4)

    for i, (label, row) in enumerate(rows):
        y = y0 + i * row_gap
        draw.text((MARGIN, y - 24), label, font=FONT["body_bold"], fill=INK)
        draw.line((x(row["bootstrap95_low"]), y, x(row["bootstrap95_high"]), y), fill=GREEN, width=16)
        draw.ellipse((x(row["mean_difference"]) - 22, y - 22, x(row["mean_difference"]) + 22, y + 22), fill=GREEN)
        draw.text(
            (x(row["bootstrap95_high"]) + 35, y - 26),
            f"{pp(row['mean_difference'])}, p = {row['sign_flip_permutation_p']:.4g}",
            font=FONT["body_bold"],
            fill=INK,
        )

    y = 1390
    draw.text((MARGIN, y), "Interpretation", font=FONT["h1"], fill=INK)
    y += 72
    draw_wrapped(
        draw,
        (
            "Hanging clothes clears the individual hard-control contrast. Orange "
            "flowers is high in absolute recognition but has a weaker standalone "
            "contrast in this Wave 2 sample. The honest claim is therefore the "
            "pooled primary packet, with pocket-specific nuance preserved."
        ),
        (MARGIN, y),
        font=FONT["body"],
        fill=INK,
        width=PAGE_W - 2 * MARGIN,
        line_gap=12,
    )
    draw_footer(draw, 4)
    return page


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    pages = [
        cover_page(summary),
        accuracy_page(summary),
        pooled_contrast_page(summary),
        individual_contrast_page(summary),
    ]
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        args.out_pdf,
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )


if __name__ == "__main__":
    main()
