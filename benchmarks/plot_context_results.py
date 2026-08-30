#!/usr/bin/env python3
"""Render dependency-free SVGs from immutable FreeToken benchmark JSON.

The context plot consumes the append-only cross-model JSONL registry.  The
weight A/B plot consumes the compact Task-7 summary.  SVG is intentional: the
benchmark environment does not need matplotlib, and ImageMagick can make a PNG
copy for chat/reporting without changing the plotted data.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Iterable


PALETTE = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2")
GRID = "#d6dbe3"
INK = "#172033"
MUTED = "#596579"
BACKGROUND = "#ffffff"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        missing = {
            "model",
            "quantization",
            "actual_context_tokens",
            "prefill_tps",
            "decode_tps",
        } - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no benchmark rows")
    return rows


def _text(x: float, y: float, value: object, *, size: int = 14, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="DejaVu Sans, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{INK}">{html.escape(str(value))}</text>'
    )


def _series_name(row: dict) -> str:
    return f"{row['model']} · {row['quantization']}"


def _format_context(value: float) -> str:
    if value >= 1024:
        return f"{value / 1024:.0f}K"
    return str(int(value))


def context_ticks(values: Iterable[float]) -> list[float]:
    """Merge actual token counts that represent the same nominal context tier."""
    groups: list[list[float]] = []
    for value in sorted({float(item) for item in values}):
        if groups and value / (sum(groups[-1]) / len(groups[-1])) <= 1.05:
            groups[-1].append(value)
        else:
            groups.append([value])
    ticks = [sum(group) / len(group) for group in groups]
    if len(ticks) > 6:
        indices = sorted(set(round(i * (len(ticks) - 1) / 5) for i in range(6)))
        ticks = [ticks[index] for index in indices]
    return ticks


def _line_panel(
    rows: list[dict],
    *,
    metric: str,
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
    colors: dict[str, str],
) -> str:
    left, right, top, bottom = 78.0, 22.0, 42.0, 60.0
    px0, px1 = x + left, x + width - right
    py0, py1 = y + top, y + height - bottom
    values = [float(row[metric]) for row in rows]
    contexts = [float(row["actual_context_tokens"]) for row in rows]
    log_min = math.log10(min(contexts))
    log_max = math.log10(max(contexts))
    if math.isclose(log_min, log_max):
        log_min -= 0.15
        log_max += 0.15
    else:
        pad = (log_max - log_min) * 0.04
        log_min -= pad
        log_max += pad
    ymax = max(values) * 1.12

    def sx(value: float) -> float:
        return px0 + (math.log10(value) - log_min) / (log_max - log_min) * (px1 - px0)

    def sy(value: float) -> float:
        return py1 - value / ymax * (py1 - py0)

    out = [
        f'<g aria-label="{html.escape(title)}">',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{BACKGROUND}" stroke="{GRID}"/>',
        _text(x + 16, y + 27, title, size=18, weight=700),
    ]
    for step in range(6):
        value = ymax * step / 5
        yy = sy(value)
        out.append(f'<line x1="{px0:.1f}" y1="{yy:.1f}" x2="{px1:.1f}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(_text(px0 - 10, yy + 5, f"{value:.0f}", size=12, anchor="end"))

    ticks = context_ticks(contexts)
    for value in ticks:
        xx = sx(value)
        out.append(f'<line x1="{xx:.1f}" y1="{py1:.1f}" x2="{xx:.1f}" y2="{py1 + 6:.1f}" stroke="{INK}"/>')
        out.append(_text(xx, py1 + 24, _format_context(value), size=12, anchor="middle"))
    out.append(_text((px0 + px1) / 2, y + height - 12, "Actual context tokens (log scale)", size=13, anchor="middle"))
    out.append(_text(x + 17, (py0 + py1) / 2, "tok/s", size=13, anchor="middle"))

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_series_name(row), []).append(row)
    for name, series_rows in grouped.items():
        color = colors[name]
        ordered = sorted(series_rows, key=lambda row: float(row["actual_context_tokens"]))
        if len({float(row["actual_context_tokens"]) for row in ordered}) > 1:
            points = " ".join(f"{sx(float(row['actual_context_tokens'])):.1f},{sy(float(row[metric])):.1f}" for row in ordered)
            out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-opacity="0.55"/>')
        for row in ordered:
            xx = sx(float(row["actual_context_tokens"]))
            yy = sy(float(row[metric]))
            detail = (
                f"{name}; context={row['actual_context_tokens']}; {metric}={float(row[metric]):.3f}; "
                f"profile={row.get('runtime_profile', 'unknown')}"
            )
            out.append(
                f'<circle data-point="{html.escape(detail)}" cx="{xx:.1f}" cy="{yy:.1f}" r="6" '
                f'fill="{color}" stroke="white" stroke-width="1.5"><title>{html.escape(detail)}</title></circle>'
            )
    out.append("</g>")
    return "\n".join(out)


def render_context_svg(rows: Iterable[dict]) -> str:
    rows = list(rows)
    if not rows:
        raise ValueError("no context rows")
    series = sorted({_series_name(row) for row in rows})
    colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(series)}
    width, height = 1400, 820
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        _text(44, 44, "FreeToken end-to-end context-speed ledger", size=26, weight=700),
        _text(44, 70, f"Every valid registry row is plotted · {len(rows)} measurements", size=14),
    ]
    legend_x = 44.0
    for name in series:
        out.append(f'<circle cx="{legend_x:.1f}" cy="100" r="6" fill="{colors[name]}"/>')
        out.append(_text(legend_x + 12, 105, name, size=13))
        legend_x += max(210, len(name) * 7.1 + 34)
    out.append(_line_panel(rows, metric="prefill_tps", title="Prefill, tok/s", x=35, y=125, width=655, height=650, colors=colors))
    out.append(_line_panel(rows, metric="decode_tps", title="Decode, tok/s", x=710, y=125, width=655, height=650, colors=colors))
    out.append("</svg>")
    return "\n".join(out)


def _bar_panel(runs: list[dict], *, metric: str, title: str, x: float, y: float, width: float, height: float) -> str:
    left, right, top, bottom = 62.0, 18.0, 42.0, 82.0
    px0, px1 = x + left, x + width - right
    py0, py1 = y + top, y + height - bottom
    values = [float(run[metric]) for run in runs]
    ymax = max(values) * 1.16 if max(values) else 1.0
    slot = (px1 - px0) / len(runs)
    bar_width = slot * 0.58
    out = [
        f'<g aria-label="{html.escape(title)}">',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{BACKGROUND}" stroke="{GRID}"/>',
        _text(x + 16, y + 27, title, size=17, weight=700),
    ]
    for step in range(5):
        value = ymax * step / 4
        yy = py1 - value / ymax * (py1 - py0)
        out.append(f'<line x1="{px0:.1f}" y1="{yy:.1f}" x2="{px1:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
    for index, (run, value) in enumerate(zip(runs, values)):
        xx = px0 + slot * index + (slot - bar_width) / 2
        yy = py1 - value / ymax * (py1 - py0)
        color = PALETTE[index % len(PALETTE)]
        detail = f"{run['label']}; {title}={value:.3f}; {run['slots']} slots"
        out.append(
            f'<rect data-point="{html.escape(detail)}" x="{xx:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
            f'height="{py1 - yy:.1f}" fill="{color}"><title>{html.escape(detail)}</title></rect>'
        )
        suffix = "%" if metric == "decode_miss_rate_percent" else ""
        decimals = 1 if value >= 100 else 2
        out.append(_text(xx + bar_width / 2, yy - 8, f"{value:.{decimals}f}{suffix}", size=13, anchor="middle", weight=700))
        out.append(_text(xx + bar_width / 2, py1 + 22, run["label"], size=12, anchor="middle", weight=700))
        out.append(_text(xx + bar_width / 2, py1 + 42, f"{run['slots']} slots", size=11, anchor="middle"))
    out.append("</g>")
    return "\n".join(out)


def render_weight_ab_svg(summary: dict) -> str:
    runs = []
    for source in summary["runs"]:
        run = dict(source)
        run["decode_miss_rate_percent"] = float(run["decode_miss_rate"]) * 100
        run["decode_transfer_mb_per_output_token"] = float(run["decode_transfer_bytes_per_output_token"]) / 1_000_000
        runs.append(run)
    width, height = 1400, 900
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        _text(38, 43, "Ornith 1.5 35B · matched 16K weight/cache A/B", size=25, weight=700),
        _text(38, 68, "Same prompt, INT8 KV, prefill=1024, greedy decode · lower is better in bottom row", size=14),
        _bar_panel(runs, metric="prefill_tps", title="Prefill, tok/s", x=30, y=95, width=665, height=370),
        _bar_panel(runs, metric="decode_tps", title="Decode, tok/s", x=705, y=95, width=665, height=370),
        _bar_panel(runs, metric="decode_miss_rate_percent", title="Cache miss, %", x=30, y=485, width=665, height=370),
        _bar_panel(runs, metric="decode_transfer_mb_per_output_token", title="Transfer, MB/output token", x=705, y=485, width=665, height=370),
        "</svg>",
    ]
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--registry", type=Path, help="append-only model-context-speed JSONL")
    source.add_argument("--weight-ab-summary", type=Path, help="Task-7 compact summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="output SVG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.registry:
        svg = render_context_svg(load_jsonl(args.registry))
    else:
        summary = json.loads(args.weight_ab_summary.read_text(encoding="utf-8"))
        svg = render_weight_ab_svg(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
