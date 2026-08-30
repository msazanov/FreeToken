#!/usr/bin/env python3
"""Render dependency-free SVGs from immutable FreeToken benchmark JSON.

The objective comparison plot consumes the append-only cross-model JSONL
registry plus an explicit cohort manifest.  The weight A/B plot consumes the
compact Task-7 summary.  SVG is intentional: the benchmark environment does
not need matplotlib, and ImageMagick can make a PNG copy for chat/reporting
without changing the plotted data.
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

BASELINE_SERIES_BY_IDENTITY = {
    ("Ornith 1.5 35b", "Q4_K_M"): "Ornith 1.5 35B · Q4_K_M",
    ("Qwen3.8 Flash Next REAP-256", "Q3_K_XL"): "Qwen3.8 REAP-256 · Q3_K_XL",
}
BASELINE_CATEGORY_BY_REQUESTED_CONTEXT = {
    1024: "1K",
    16384: "16K",
    65536: "64K",
}


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


def _artifact_key(value: object) -> str:
    artifact = str(value).replace("\\", "/")
    marker = "benchmarks/results/"
    if marker in artifact:
        return artifact.split(marker, 1)[1]
    return artifact.removeprefix("./")


def classify_comparison_rows(rows: Iterable[dict], manifest: dict) -> dict:
    """Join immutable ledger rows to explicitly declared comparison cohorts."""
    by_artifact: dict[str, dict] = {}
    for row in rows:
        key = _artifact_key(row["artifact"])
        if key in by_artifact:
            raise ValueError(f"duplicate ledger artifact: {key}")
        by_artifact[key] = row
    assigned: set[str] = set()
    cohorts: dict[str, list[dict]] = {}
    for cohort_name, entries in manifest.get("cohorts", {}).items():
        cohorts[cohort_name] = []
        for entry in entries:
            item = dict(entry)
            key = _artifact_key(entry["artifact"])
            if key in assigned:
                raise ValueError(f"benchmark row assigned more than once: {key}")
            row = by_artifact[key]
            if cohort_name == "model_context":
                identity = (row["model"], row["quantization"])
                expected_series = BASELINE_SERIES_BY_IDENTITY.get(identity)
                if expected_series != entry.get("series"):
                    raise ValueError(
                        f"baseline series mismatch for {key}: "
                        f"expected {expected_series!r}, got {entry.get('series')!r}"
                    )
                expected_category = BASELINE_CATEGORY_BY_REQUESTED_CONTEXT.get(
                    int(row["requested_context_tokens"])
                )
                if expected_category != entry.get("category"):
                    raise ValueError(
                        f"baseline category mismatch for {key}: "
                        f"expected {expected_category!r}, got {entry.get('category')!r}"
                    )
            item["row"] = row
            assigned.add(key)
            cohorts[cohort_name].append(item)
    excluded: list[dict] = []
    for entry in manifest.get("excluded", []):
        item = dict(entry)
        key = _artifact_key(entry["artifact"])
        if key in assigned:
            raise ValueError(f"benchmark row assigned more than once: {key}")
        item["row"] = by_artifact[key]
        assigned.add(key)
        excluded.append(item)
    unclassified = sorted(set(by_artifact) - assigned)
    if unclassified:
        raise ValueError(f"unclassified benchmark rows: {', '.join(unclassified)}")
    return {"cohorts": cohorts, "excluded": excluded}


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


def _scatter_marker(
    *,
    x: float,
    y: float,
    color: str,
    shape: str,
    size: float = 7.0,
    hollow: bool = False,
) -> str:
    fill = BACKGROUND if hollow else color
    stroke_width = 2.5 if hollow else 1.5
    if shape == "circle":
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="{fill}" '
            f'stroke="{color}" stroke-width="{stroke_width}"/>'
        )
    if shape == "square":
        side = size * 1.75
        return (
            f'<rect x="{x - side / 2:.1f}" y="{y - side / 2:.1f}" width="{side:.1f}" '
            f'height="{side:.1f}" fill="{fill}" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
    if shape == "diamond":
        points = f"{x:.1f},{y - size:.1f} {x + size:.1f},{y:.1f} {x:.1f},{y + size:.1f} {x - size:.1f},{y:.1f}"
        return f'<polygon points="{points}" fill="{fill}" stroke="{color}" stroke-width="{stroke_width}"/>'
    if shape == "triangle":
        points = f"{x:.1f},{y - size:.1f} {x + size:.1f},{y + size:.1f} {x - size:.1f},{y + size:.1f}"
        return f'<polygon points="{points}" fill="{fill}" stroke="{color}" stroke-width="{stroke_width}"/>'
    if shape == "cross":
        return (
            f'<line x1="{x - size:.1f}" y1="{y - size:.1f}" x2="{x + size:.1f}" y2="{y + size:.1f}" stroke="{color}" stroke-width="2.5"/>'
            f'<line x1="{x - size:.1f}" y1="{y + size:.1f}" x2="{x + size:.1f}" y2="{y - size:.1f}" stroke="{color}" stroke-width="2.5"/>'
        )
    raise ValueError(f"unknown marker shape: {shape}")


def _scatter_legend_item(
    *,
    x: float,
    y: float,
    color: str,
    shape: str,
    label: str,
    line: bool = False,
    hollow: bool = False,
) -> str:
    marker_x = x + 16
    parts = []
    if line:
        parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 32:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
    parts.append(
        _scatter_marker(
            x=marker_x,
            y=y,
            color=color,
            shape=shape,
            size=6,
            hollow=hollow,
        )
    )
    parts.append(_text(x + 42, y + 5, label, size=12))
    return "".join(parts)


def render_comparison_svg(rows: Iterable[dict], manifest: dict) -> str:
    rows = list(rows)
    classified = classify_comparison_rows(rows, manifest)
    cohorts = classified["cohorts"]
    model_context = cohorts["model_context"]
    tq3_ab = cohorts["tq3_weight_cache_ab"]
    prefill_sweep = cohorts["prefill_block_sweep"]
    excluded = classified["excluded"]

    ornith_base = "#1d4ed8"
    ornith_q4_repeat = "#2563eb"
    ornith_tq3_fixed = "#60a5fa"
    ornith_tq3_auto = "#0ea5e9"
    ornith_prefill = "#93c5fd"
    ornith_prefill_palette = (
        "#bfdbfe",
        "#93c5fd",
        "#60a5fa",
        "#7dd3fc",
        "#38bdf8",
        "#0ea5e9",
        "#0284c7",
        "#0369a1",
    )
    ornith_prefill_shapes = (
        "square",
        "diamond",
        "triangle",
        "circle",
        "square",
        "diamond",
        "triangle",
        "circle",
    )
    ornith_unmatched = "#1e3a8a"
    qwen_base = "#059669"

    points: list[dict] = []
    baseline_lines: dict[str, list[dict]] = {
        "Ornith Q4_K_M · базовая линия": [],
        "Qwen REAP Q3_K_XL · базовая линия": [],
    }
    for entry in model_context:
        is_ornith = entry["row"]["model"].startswith("Ornith")
        label = (
            "Ornith Q4_K_M · базовая линия"
            if is_ornith
            else "Qwen REAP Q3_K_XL · базовая линия"
        )
        point = {
            "row": entry["row"],
            "family": "ornith" if is_ornith else "qwen",
            "config": label,
            "color": ornith_base if is_ornith else qwen_base,
            "shape": "circle" if is_ornith else "square",
            "hollow": False,
            "annotation": f"{float(entry['row']['decode_tps']):.2f}",
        }
        points.append(point)
        baseline_lines[label].append(point)

    for entry in tq3_ab:
        variant = entry["variant"]
        if variant == "Q4_K_M fixed":
            config = "Ornith Q4_K_M · fixed 1429"
            color, shape, hollow = ornith_q4_repeat, "circle", True
        elif variant == "TQ3_4S fixed":
            config = "Ornith TQ3_4S · fixed 1429"
            color, shape, hollow = ornith_tq3_fixed, "diamond", False
        else:
            config = "Ornith TQ3_4S · auto 2633"
            color, shape, hollow = ornith_tq3_auto, "triangle", False
        points.append(
            {
                "row": entry["row"],
                "family": "ornith",
                "config": config,
                "color": color,
                "shape": shape,
                "hollow": hollow,
                "annotation": None,
                "repeat": entry["repeat"],
            }
        )

    for index, entry in enumerate(prefill_sweep):
        points.append(
            {
                "row": entry["row"],
                "family": "ornith",
                "config": "Ornith Q4_K_M · prefill-block sweep",
                "color": ornith_prefill_palette[index],
                "shape": ornith_prefill_shapes[index],
                "hollow": False,
                "annotation": None,
                "prefill_block": entry["category"],
            }
        )

    for entry in excluded:
        points.append(
            {
                "row": entry["row"],
                "family": "ornith",
                "config": "Ornith Q4_K_M · unmatched short run",
                "color": ornith_unmatched,
                "shape": "cross",
                "hollow": True,
                "annotation": None,
                "reason": entry["reason"],
            }
        )

    max_decode = max(float(point["row"]["decode_tps"]) for point in points)
    x_max = math.ceil(max_decode * 1.08 / 5) * 5
    width, height = 1500, 1050
    px0, px1 = 145.0, 1440.0
    py0, py1 = 190.0, 790.0
    log_min, log_max = 9.5, 16.5

    def sx(value: float) -> float:
        return px0 + value / x_max * (px1 - px0)

    def sy(value: float) -> float:
        return py1 - (math.log2(value) - log_min) / (log_max - log_min) * (py1 - py0)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        _text(42, 43, "Скорость генерации в зависимости от размера контекста", size=27, weight=700),
        _text(42, 71, f"RTX 2070 · FreeToken · {len(points)} сохранённый end-to-end прогон · точка = один сохранённый end-to-end прогон", size=14),
        f'<g data-x-metric="decode_tps" data-x-scale="linear" data-y-metric="actual_context_tokens" data-y-scale="log2">',
    ]

    legend = [
        (42, 108, ornith_base, "circle", "Ornith Q4_K_M · базовая линия", True, False),
        (405, 108, qwen_base, "square", "Qwen REAP Q3_K_XL · базовая линия", True, False),
        (790, 108, ornith_q4_repeat, "circle", "Ornith Q4 · fixed 1429", False, True),
        (1090, 108, ornith_tq3_fixed, "diamond", "Ornith TQ3_4S · fixed 1429", False, False),
        (42, 142, ornith_tq3_auto, "triangle", "Ornith TQ3_4S · auto 2633", False, False),
        (350, 142, ornith_prefill, "square", "Ornith Q4_K_M · prefill-block sweep", False, False),
        (735, 142, ornith_unmatched, "cross", "Ornith Q4 · несопоставимый short-run", False, True),
    ]
    for lx, ly, color, shape, label, has_line, hollow in legend:
        out.append(
            _scatter_legend_item(
                x=lx,
                y=ly,
                color=color,
                shape=shape,
                label=label,
                line=has_line,
                hollow=hollow,
            )
        )

    for tick in range(0, int(x_max) + 1, 5):
        xx = sx(float(tick))
        out.append(
            f'<line x1="{xx:.1f}" y1="{py0:.1f}" x2="{xx:.1f}" y2="{py1:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(_text(xx, py1 + 25, tick, size=12, anchor="middle"))
    for exponent in range(10, 17):
        context = 2**exponent
        yy = sy(float(context))
        out.append(
            f'<line x1="{px0:.1f}" y1="{yy:.1f}" x2="{px1:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(_text(px0 - 12, yy + 5, f"{context // 1024}K", size=12, anchor="end"))
    out.append(
        f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{px1 - px0:.1f}" height="{py1 - py0:.1f}" '
        f'fill="none" stroke="{INK}" stroke-width="1.2"/>'
    )

    for label, line_points in baseline_lines.items():
        ordered = sorted(line_points, key=lambda point: float(point["row"]["actual_context_tokens"]))
        coords = " ".join(
            f"{sx(float(point['row']['decode_tps'])):.1f},{sy(float(point['row']['actual_context_tokens'])):.1f}"
            for point in ordered
        )
        color = ordered[0]["color"]
        artifacts = "|".join(
            _artifact_key(point["row"]["artifact"]) for point in ordered
        )
        out.append(
            f'<polyline data-series-line="{html.escape(label)}" '
            f'data-series-artifacts="{html.escape(artifacts)}" points="{coords}" fill="none" '
            f'stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    for point in points:
        row = point["row"]
        xx = sx(float(row["decode_tps"]))
        yy = sy(float(row["actual_context_tokens"]))
        extras = []
        if "repeat" in point:
            extras.append(f"repeat={point['repeat']}")
        if "prefill_block" in point:
            extras.append(f"prefill={point['prefill_block']}")
        detail = (
            f"{point['config']}; context={row['actual_context_tokens']}; "
            f"decode={float(row['decode_tps']):.6f} tok/s; output={row['completion_tokens']}"
        )
        if extras:
            detail += "; " + "; ".join(extras)
        artifact = _artifact_key(row["artifact"])
        out.append(
            f'<g data-measurement="{html.escape(detail)}" '
            f'data-artifact="{html.escape(artifact)}" '
            f'data-context-tokens="{int(row["actual_context_tokens"])}" '
            f'data-decode-tps="{repr(float(row["decode_tps"]))}" '
            f'data-marker-shape="{point["shape"]}" '
            f'data-marker-color="{point["color"]}" '
            f'data-model-family="{point["family"]}">'
            + _scatter_marker(
                x=xx,
                y=yy,
                color=point["color"],
                shape=point["shape"],
                size=7.5,
                hollow=point["hollow"],
            )
            + f'<title>{html.escape(detail)}</title></g>'
        )
        if point["annotation"]:
            anchor = "start" if point["family"] == "qwen" else "middle"
            label_x = xx + 11 if point["family"] == "qwen" else xx
            label_y = yy - 11
            out.append(_text(label_x, label_y, point["annotation"], size=11, anchor=anchor, weight=700))

    out.extend(
        [
            _text((px0 + px1) / 2, 835, "Скорость генерации decode, ток/с — линейная шкала", size=14, anchor="middle", weight=700),
            (
                f'<text data-axis="context" transform="rotate(-90 34 {(py0 + py1) / 2:.1f})" '
                f'x="34" y="{(py0 + py1) / 2:.1f}" text-anchor="middle" '
                f'font-family="DejaVu Sans, sans-serif" font-size="14" font-weight="700" fill="{INK}">'
                "Фактический контекст, токенов — log2, каждый шаг ×2</text>"
            ),
            _text(42, 875, "1K feature-повторы: Q4 fixed 25.26/24.69 · TQ3 fixed 28.49/27.55 · TQ3 auto 35.06/33.43 ток/с.", size=12),
            _text(42, 900, "16K prefill-block sweep: p1024=19.52 · p1280=26.09 · p1536=17.60 · p1792=18.46 · p2048=17.63 ток/с", size=12),
            _text(42, 922, "продолжение: p2560=28.06 · p3072=19.66 · p4096=18.01. Крест: отдельный short-run p1024=21.48, output=108.", size=12),
            _text(42, 955, "Линия соединяет только одинаковую конфигурацию на 1K/16K/64K. Одноточечные модификации не образуют выдуманную зависимость от контекста.", size=12),
            _text(42, 978, "Синие оттенки — Ornith и её модификации; зелёный — Qwen. Decode throughput не является оценкой качества ответа.", size=12),
            _text(42, 1010, "Источник: benchmarks/results/model-context-speed.jsonl · классификация: benchmarks/comparison_cohorts.json", size=12),
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(out)


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
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=Path(__file__).with_name("comparison_cohorts.json"),
        help="explicit cohort/exclusion manifest used with --registry",
    )
    parser.add_argument("--output", type=Path, required=True, help="output SVG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.registry:
        manifest = json.loads(args.comparison_manifest.read_text(encoding="utf-8"))
        svg = render_comparison_svg(load_jsonl(args.registry), manifest)
    else:
        summary = json.loads(args.weight_ab_summary.read_text(encoding="utf-8"))
        svg = render_weight_ab_svg(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
