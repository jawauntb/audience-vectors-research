"""Summarize Wan2.2 base-or-gated-best-of-N selector results.

Inputs are the base/single-LoRA TRIBE report, the LoRA best-of-N TRIBE report,
and the CLIP preservation composite report. Outputs a compact selector report
plus optional midframe QC sheets.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def scores_by_label(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["label"]): float(row["v_mem_projection"])
        for row in report["scores"]
    }


def rows_by_label(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in report["rows"]}


def best_bon_label(seed: str, bon_scores: dict[str, float]) -> str:
    prefix = f"{seed}_m"
    labels = [label for label in bon_scores if label.startswith(prefix)]
    if not labels:
        raise KeyError(f"no best-of-N labels found for {seed}")
    return max(labels, key=lambda label: bon_scores[label])


def policy_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n_improved": sum(value > 1e-9 for value in values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize(
    *,
    single_report: dict[str, Any],
    bon_report: dict[str, Any],
    composite_report: dict[str, Any],
) -> dict[str, Any]:
    single_scores = scores_by_label(single_report)
    bon_scores = scores_by_label(bon_report)
    composite_rows = rows_by_label(composite_report)
    seeds = sorted(single_report["by_seed"])

    rows: list[dict[str, Any]] = []
    deltas = {
        "single_lora": [],
        "raw_best_of_n": [],
        "gated_best_of_n": [],
        "base_or_single_lora": [],
        "base_or_raw_best_of_n": [],
        "base_or_gated_best_of_n": [],
    }

    for seed in seeds:
        base_label = f"{seed}_base"
        single_label = f"{seed}_lora"
        raw_label = best_bon_label(seed, bon_scores)
        gated_label = composite_report["by_seed"][seed]["gated_best_label"]
        if gated_label is None:
            gated_label = raw_label

        base = single_scores[base_label]
        single = single_scores[single_label]
        raw = bon_scores[raw_label]
        gated = bon_scores[gated_label]
        product_score = max(base, gated)
        product_label = gated_label if gated > base else base_label
        product_variant = "gated_lora_best_of_n" if gated > base else "base"

        single_delta = single - base
        raw_delta = raw - base
        gated_delta = gated - base
        base_or_single = max(base, single) - base
        base_or_raw = max(base, raw) - base
        base_or_gated = product_score - base

        deltas["single_lora"].append(single_delta)
        deltas["raw_best_of_n"].append(raw_delta)
        deltas["gated_best_of_n"].append(gated_delta)
        deltas["base_or_single_lora"].append(base_or_single)
        deltas["base_or_raw_best_of_n"].append(base_or_raw)
        deltas["base_or_gated_best_of_n"].append(base_or_gated)

        gated_row = composite_rows[gated_label]
        rows.append(
            {
                "seed": seed,
                "base_label": base_label,
                "single_label": single_label,
                "raw_best_label": raw_label,
                "gated_best_label": gated_label,
                "product_label": product_label,
                "product_variant": product_variant,
                "base_score": base,
                "single_score": single,
                "raw_best_score": raw,
                "gated_best_score": gated,
                "product_score": product_score,
                "single_delta": single_delta,
                "raw_best_delta": raw_delta,
                "gated_best_delta": gated_delta,
                "base_or_single_lift": base_or_single,
                "base_or_raw_lift": base_or_raw,
                "base_or_gated_lift": base_or_gated,
                "gated_seed_image_cosine": gated_row["seed_image_cosine"],
                "gated_prompt_clip_cosine": gated_row["prompt_clip_cosine"],
                "gated_passes_preservation_gate": gated_row[
                    "passes_preservation_gate"
                ],
            }
        )

    return {
        "summary": {
            "n_seeds": len(seeds),
            "policies": {
                name: policy_stats(values) for name, values in deltas.items()
            },
            "composite_summary": composite_report["summary"],
            "bon_summary": bon_report["summary"],
        },
        "selections": rows,
    }


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    summary = report["summary"]
    policy_rows = [
        ("Single LoRA", "single_lora"),
        ("Raw best-of-N", "raw_best_of_n"),
        ("Gated best-of-N", "gated_best_of_n"),
        ("Base or single LoRA", "base_or_single_lora"),
        ("Base or raw best-of-N", "base_or_raw_best_of_n"),
        ("Base or gated best-of-N", "base_or_gated_best_of_n"),
    ]

    lines = [
        "# Wan2.2 Product-Style Selector",
        "",
        "Policy family: compare base clips to LoRA candidates, optionally keep the base when LoRA hurts the TRIBE/BMD projection.",
        "",
        "## Summary",
        "",
        f"- Seeds: **{summary['n_seeds']}**",
        f"- Preservation gate pass rate: **{summary['composite_summary']['gate_pass_rate']:.3f}**",
        "",
        "| selection rule | improved seeds | mean | median | min/max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in policy_rows:
        stats = summary["policies"][key]
        lines.append(
            f"| {label} | {stats['n_improved']}/{summary['n_seeds']} | "
            f"{stats['mean']:+.4f} | {stats['median']:+.4f} | "
            f"{stats['min']:+.4f} / {stats['max']:+.4f} |"
        )

    lines += [
        "",
        "## Per Seed",
        "",
        "| seed | base | single | raw best | gated best | product |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["selections"]:
        lines.append(
            f"| `{row['seed']}` | {row['base_score']:.4f} | "
            f"{row['single_score']:.4f} ({row['single_delta']:+.4f}) | "
            f"{row['raw_best_score']:.4f} ({row['raw_best_delta']:+.4f}) | "
            f"{row['gated_best_score']:.4f} ({row['gated_best_delta']:+.4f}) | "
            f"`{row['product_label']}` ({row['base_or_gated_lift']:+.4f}) |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def midframe(path: Path) -> Image.Image:
    frames = [np.asarray(frame) for frame in iio.imiter(path)]
    if not frames:
        raise ValueError(f"no frames in {path}")
    return Image.fromarray(frames[len(frames) // 2]).convert("RGB")


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (width, height), "white")
    out.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return out


def load_fonts() -> tuple[Any, Any]:
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def tile(
    *,
    path: Path,
    title: str,
    subtitle: str,
    size: tuple[int, int],
    small_font: Any,
) -> Image.Image:
    width, height = size
    text_height = 50
    out = Image.new("RGB", size, "white")
    out.paste(fit_image(midframe(path), width, height - text_height), (0, text_height))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(190, 190, 190), width=1)
    draw.text((8, 5), title[:42], font=small_font, fill=(20, 20, 20))
    draw.text((8, 26), subtitle[:56], font=small_font, fill=(70, 70, 70))
    return out


def write_contact_sheet(
    *,
    items: list[tuple[Path, str, str]],
    cols: int,
    out_path: Path,
    title: str,
) -> None:
    title_font, small_font = load_fonts()
    tile_size = (320, 230)
    top_height = 42
    n_rows = (len(items) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * tile_size[0], top_height + n_rows * tile_size[1]),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=title_font, fill=(10, 10, 10))
    for idx, (path, item_title, subtitle) in enumerate(items):
        image = tile(
            path=path,
            title=item_title,
            subtitle=subtitle,
            size=tile_size,
            small_font=small_font,
        )
        sheet.paste(
            image,
            ((idx % cols) * tile_size[0], top_height + (idx // cols) * tile_size[1]),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_qc_sheets(
    *,
    report: dict[str, Any],
    single_generated_dir: Path,
    bon_generated_dir: Path,
    out_prefix: Path,
) -> None:
    comparison_items: list[tuple[Path, str, str]] = []
    product_items: list[tuple[Path, str, str]] = []
    for row in report["selections"]:
        base_path = single_generated_dir / f"{row['base_label']}.mp4"
        single_path = single_generated_dir / f"{row['single_label']}.mp4"
        raw_path = bon_generated_dir / f"{row['raw_best_label']}.mp4"
        gated_path = bon_generated_dir / f"{row['gated_best_label']}.mp4"
        product_path = (
            gated_path if row["product_variant"] != "base" else base_path
        )
        comparison_items.extend(
            [
                (base_path, f"{row['seed']} base", f"v={row['base_score']:+.2f}"),
                (
                    single_path,
                    f"{row['seed']} single",
                    f"d={row['single_delta']:+.2f}",
                ),
                (
                    raw_path,
                    f"{row['seed']} raw best",
                    f"d={row['raw_best_delta']:+.2f}",
                ),
                (
                    gated_path,
                    f"{row['seed']} gated best",
                    f"d={row['gated_best_delta']:+.2f}",
                ),
            ]
        )
        product_items.append(
            (
                product_path,
                f"{row['seed']} / {row['product_variant']}",
                f"lift={row['base_or_gated_lift']:+.2f} score={row['product_score']:+.2f}",
            )
        )
    write_contact_sheet(
        items=comparison_items,
        cols=4,
        out_path=out_prefix.with_name(f"{out_prefix.name}_comparison.jpg"),
        title="Base vs single LoRA vs raw best-of-N vs gated best-of-N",
    )
    write_contact_sheet(
        items=product_items,
        cols=4,
        out_path=out_prefix.with_name(f"{out_prefix.name}_product_selected.jpg"),
        title="Current product selector: base or gated best-of-N",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-report", type=Path, required=True)
    parser.add_argument("--bon-report", type=Path, required=True)
    parser.add_argument("--composite-report", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--single-generated-dir", type=Path)
    parser.add_argument("--bon-generated-dir", type=Path)
    parser.add_argument("--qc-prefix", type=Path)
    args = parser.parse_args()

    report = summarize(
        single_report=load_json(args.single_report),
        bon_report=load_json(args.bon_report),
        composite_report=load_json(args.composite_report),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[selector] wrote {args.out_json}", flush=True)
    print(f"[selector] wrote {args.out_md}", flush=True)

    if args.qc_prefix is not None:
        if args.single_generated_dir is None or args.bon_generated_dir is None:
            raise ValueError("QC sheets require both generated dirs")
        write_qc_sheets(
            report=report,
            single_generated_dir=args.single_generated_dir,
            bon_generated_dir=args.bon_generated_dir,
            out_prefix=args.qc_prefix,
        )
        print(f"[selector] wrote QC sheets at prefix {args.qc_prefix}", flush=True)


if __name__ == "__main__":
    main()
