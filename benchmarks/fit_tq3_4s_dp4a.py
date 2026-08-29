#!/usr/bin/env python3
"""Reproduce the SM75 int8 table used to approximate TQ3_4S centroids.

Turing DP4A needs integer lanes with one common scale.  TQ3_4S instead stores
eight asymmetric Lloyd-Max centroids.  This script exhaustively enumerates every
rounded int8 table reachable between scale 0.01 and 0.03, refits the common scale
by weighted least squares, and selects minimum error.  Bin weights are the
standard-normal probability masses implied by the midpoint boundaries of the
published Lloyd-Max centroids.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CENTROIDS = (
    -1.996684,
    -1.291398,
    -0.740341,
    -0.247508,
    0.230106,
    0.725222,
    1.277503,
    1.988943,
)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def gaussian_bin_weights() -> tuple[float, ...]:
    boundaries = [-math.inf]
    boundaries.extend((left + right) / 2.0 for left, right in zip(CENTROIDS, CENTROIDS[1:]))
    boundaries.append(math.inf)
    return tuple(
        normal_cdf(boundaries[index + 1]) - normal_cdf(boundaries[index])
        for index in range(len(CENTROIDS))
    )


def packed_i8_hex(values: tuple[int, ...]) -> str:
    word = sum((value & 0xFF) << (8 * index) for index, value in enumerate(values))
    return f"0x{word:08X}"


def reachable_tables(min_scale: float, max_scale: float) -> set[tuple[int, ...]]:
    """Enumerate constant-rounding intervals instead of relying on a grid search."""
    breakpoints = {min_scale, max_scale}
    for centroid in CENTROIDS:
        magnitude = abs(centroid)
        for integer in range(128):
            boundary = magnitude / (integer + 0.5)
            if min_scale < boundary < max_scale:
                breakpoints.add(boundary)

    ordered = sorted(breakpoints)
    scales = [
        (left + right) / 2.0 for left, right in zip(ordered, ordered[1:])
    ]
    tables = set()
    for scale in scales:
        tables.add(
            tuple(max(-127, min(127, round(centroid / scale))) for centroid in CENTROIDS)
        )
    return tables


def fit_codebook(min_scale: float = 0.01, max_scale: float = 0.03) -> dict[str, object]:
    weights = gaussian_bin_weights()
    best: tuple[float, tuple[int, ...], float, float] | None = None
    tables = reachable_tables(min_scale, max_scale)
    for levels in tables:
        denominator = sum(weight * level * level for weight, level in zip(weights, levels))
        if denominator == 0:
            continue
        scale = sum(
            weight * level * centroid
            for weight, level, centroid in zip(weights, levels, CENTROIDS)
        ) / denominator
        errors = tuple(
            level * scale - centroid for level, centroid in zip(levels, CENTROIDS)
        )
        weighted_rmse = math.sqrt(
            sum(weight * error * error for weight, error in zip(weights, errors))
        )
        candidate = (weighted_rmse, levels, scale, max(map(abs, errors)))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("scale interval produced no non-zero candidate")

    weighted_rmse, levels, scale, max_abs = best
    return {
        "centroids": list(CENTROIDS),
        "gaussian_bin_weights": list(weights),
        "scale_search_interval": [min_scale, max_scale],
        "reachable_tables_evaluated": len(tables),
        "levels": list(levels),
        "scale": scale,
        "weighted_rmse": weighted_rmse,
        "max_centroid_abs_error": max_abs,
        "levels_lo_hex": packed_i8_hex(levels[:4]),
        "levels_hi_hex": packed_i8_hex(levels[4:]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON artifact; never overwritten")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = fit_codebook()
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite immutable artifact: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
