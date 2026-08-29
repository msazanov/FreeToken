#!/usr/bin/env python3
"""Reproducible SM75 microbenchmark for FreeToken's Ornith TQ3_4S MMVQ path.

This measures two real Ornith expert-matrix geometries at batch size one.  It
compares the packed DP4A MMVQ kernel with the correctness-first fallback that
materializes the exact type-46 matrix and then calls ``torch.mm``.  It is a
kernel benchmark, not an end-to-end model throughput or quality benchmark.

Every CUDA-event sample is retained in the output JSON together with source,
GPU, seed, codebook, numerical-error, and thermal provenance.

Run from the repository root::

    MAX_JOBS=12 PYTHONPATH=python:. python benchmarks/tq3_4s_kernel_bench.py \
      --output benchmarks/results/ornith35-tq3-sm75-kernel/task4-mmvq.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

ORNITH_EXPERT_SHAPES = {
    "expert_gate_up": (512, 2048),
    "expert_down": (2048, 512),
}
TQ3_4S_LEVELS = (-113, -73, -42, -14, 13, 41, 72, 112)
TQ3_4S_DP4A_SCALE = 0.017704291602768495
SOURCE_INPUTS = (
    "benchmarks/tq3_4s_kernel_bench.py",
    "python/freetoken/kernel/csrc/gguf/ggml-common.h",
    "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu",
    "python/freetoken/kernel/csrc/gguf/mmvq.cuh",
    "python/freetoken/kernel/csrc/gguf/vecdotq.cuh",
    "python/freetoken/models/gguf/dequant.py",
)


def percentile(samples: list[float], quantile: float) -> float:
    if not samples:
        raise ValueError("cannot summarize an empty sample list")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_samples(samples: list[float]) -> dict[str, object]:
    """Summarize timings without discarding their original measurement order."""
    return {
        "count": len(samples),
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 0.50),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": percentile(samples, 0.95),
        "max_ms": max(samples),
        "raw_ms": samples,
    }


def _run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def git_provenance() -> dict[str, object]:
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", "."],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    status = _run_text(["git", "status", "--short"])
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "branch": _run_text(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def source_provenance() -> dict[str, str]:
    """Content hashes cover untracked sources that ``git diff`` cannot encode."""
    return {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_INPUTS
    }


def nvidia_smi_snapshot() -> dict[str, object]:
    fields = (
        "name,uuid,temperature.gpu,power.draw,utilization.gpu,"
        "clocks.current.sm,memory.used,memory.free"
    )
    raw = _run_text(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    keys = fields.split(",")
    values = [item.strip() for item in raw.split(",")]
    if len(values) != len(keys):
        return {"raw": raw}
    return dict(zip(keys, values, strict=True))


def make_packed_matrix(rows: int, columns: int, seed: int):
    import torch

    if columns % 32:
        raise ValueError("TQ3_4S columns must be divisible by 32")
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0,
        256,
        (rows, columns // 32 * 16),
        dtype=torch.uint8,
        generator=generator,
    )
    blocks = packed.reshape(-1, 16)
    scale_bank = torch.tensor([0x20, 0x5F, 0x80, 0xFF], dtype=torch.uint8)
    block_index = torch.arange(blocks.shape[0]).view(-1, 1)
    lane_index = torch.arange(4).view(1, -1)
    blocks[:, :4] = scale_bank[(block_index + lane_index) % 4]
    return packed, generator


def operation_order(iteration: int) -> tuple[str, str]:
    operations = ("packed_mmvq", "exact_cuda_dequant_plus_torch_mm")
    return operations if iteration % 2 == 0 else operations[::-1]


def cuda_event_interleaved_samples(
    operations: dict[str, Callable[[], object]], *, warmup: int, iterations: int
) -> dict[str, list[float]]:
    import torch

    expected = set(operation_order(0))
    if set(operations) != expected:
        raise ValueError(f"expected benchmark operations {sorted(expected)}")
    results: dict[str, object] = {}
    for iteration in range(warmup):
        for name in operation_order(iteration):
            results[name] = operations[name]()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in expected}
    for iteration in range(iterations):
        for name in operation_order(iteration):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            results[name] = operations[name]()
            stop.record()
            stop.synchronize()
            samples[name].append(float(start.elapsed_time(stop)))
    # Keep the last result of both operations live through synchronization.
    if set(results) != expected:
        raise RuntimeError("benchmark operation did not execute")
    return samples


def quality_metrics(actual, reference) -> dict[str, float]:
    import torch

    actual_f32 = actual.float().cpu()
    reference_f32 = reference.float().cpu()
    error = actual_f32 - reference_f32
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference_f32)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), reference_f32.flatten(), dim=0
    )
    return {
        "cosine": float(cosine.item()),
        "relative_l2": float(relative_l2.item()),
        "max_abs": float(error.abs().max().item()),
        "mean_abs": float(error.abs().mean().item()),
    }


def quality_acceptance(
    quality: dict[str, float], *, min_cosine: float, max_relative_l2: float
) -> dict[str, object]:
    return {
        "passed": quality["cosine"] >= min_cosine
        and quality["relative_l2"] <= max_relative_l2,
        "min_cosine": min_cosine,
        "max_relative_l2": max_relative_l2,
    }


def run_case(
    *,
    name: str,
    rows: int,
    columns: int,
    dtype,
    seed: int,
    warmup: int,
    iterations: int,
    min_cosine: float,
    max_relative_l2: float,
) -> dict[str, object]:
    import torch

    from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_vec_a8
    from freetoken.models.gguf.dequant import GGML_TQ3_4S, dequant_tq3_4s

    packed_cpu, generator = make_packed_matrix(rows, columns, seed)
    activation_cpu = (
        torch.randn((1, columns), generator=generator, dtype=torch.float32) * 0.5
    ).to(dtype)
    packed = packed_cpu.cuda()
    activation = activation_cpu.cuda()

    def mmvq():
        return ggml_mul_mat_vec_a8(packed, activation, GGML_TQ3_4S, rows)

    def exact_materialize_then_mm():
        dense = ggml_dequantize(packed, GGML_TQ3_4S, rows, columns, dtype)
        return activation @ dense.t()

    exact_cpu = dequant_tq3_4s(packed_cpu.flatten(), torch.float32).reshape(rows, columns)
    reference = activation_cpu.float() @ exact_cpu.t()
    actual = mmvq()
    torch.cuda.synchronize()

    quality = quality_metrics(actual, reference)
    samples = cuda_event_interleaved_samples(
        {
            "packed_mmvq": mmvq,
            "exact_cuda_dequant_plus_torch_mm": exact_materialize_then_mm,
        },
        warmup=warmup,
        iterations=iterations,
    )
    mmvq_summary = summarize_samples(samples["packed_mmvq"])
    fallback_summary = summarize_samples(samples["exact_cuda_dequant_plus_torch_mm"])

    return {
        "name": name,
        "dtype": str(dtype).removeprefix("torch."),
        "batch_size": 1,
        "rows_out": rows,
        "columns_in": columns,
        "values": rows * columns,
        "packed_bytes": packed_cpu.numel(),
        "stored_bits_per_weight": packed_cpu.numel() * 8 / (rows * columns),
        "seed": seed,
        "quality_vs_exact_fp32_cpu": quality,
        "acceptance": quality_acceptance(
            quality,
            min_cosine=min_cosine,
            max_relative_l2=max_relative_l2,
        ),
        "timing": {
            "packed_mmvq": mmvq_summary,
            "exact_cuda_dequant_plus_torch_mm": fallback_summary,
            "p50_speedup": float(fallback_summary["p50_ms"])
            / float(mmvq_summary["p50_ms"]),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True, type=Path, help="immutable JSON artifact")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must both be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        raise SystemExit(f"this target benchmark requires exact SM75, got {capability}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable artifact: {args.output}")

    started = nvidia_smi_snapshot()
    cases: list[dict[str, object]] = []
    for shape_index, (name, (rows, columns)) in enumerate(ORNITH_EXPERT_SHAPES.items()):
        for dtype_index, dtype in enumerate((torch.float16, torch.bfloat16)):
            case_seed = args.seed + shape_index * 100 + dtype_index
            print(f"[tq3] {name} {rows}x{columns} {dtype} seed={case_seed}", flush=True)
            cases.append(
                run_case(
                    name=name,
                    rows=rows,
                    columns=columns,
                    dtype=dtype,
                    seed=case_seed,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    min_cosine=args.min_cosine,
                    max_relative_l2=args.max_relative_l2,
                )
            )
            timing = cases[-1]["timing"]
            quality = cases[-1]["quality_vs_exact_fp32_cpu"]
            print(
                f"[tq3] p50={timing['packed_mmvq']['p50_ms']:.6f} ms "
                f"speedup={timing['p50_speedup']:.3f}x "
                f"rel_l2={quality['relative_l2']:.6f}",
                flush=True,
            )

    artifact = {
        "schema_version": 1,
        "kind": "tq3_4s_sm75_kernel_microbenchmark",
        "scope": "packed batch-1 expert MMVQ versus exact materialize-plus-MM",
        "not_measured": [
            "model tokens/second",
            "prefill throughput",
            "fused MoE routing/accumulation",
            "expert cache hit rate",
            "end-to-end model quality",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "parameters": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "base_seed": args.seed,
            "cuda_event_per_sample": True,
            "measurement_order": "alternating packed/fallback, reversed each iteration",
            "min_cosine": args.min_cosine,
            "max_relative_l2": args.max_relative_l2,
        },
        "codebook": {
            "int8_levels": TQ3_4S_LEVELS,
            "scale": TQ3_4S_DP4A_SCALE,
            "fit": "weighted least squares against exact TQ3_4S Lloyd-Max centroids",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "hardware": {
            "cuda_device": torch.cuda.get_device_name(),
            "compute_capability": list(capability),
            "before": started,
            "after": nvidia_smi_snapshot(),
        },
        "git": git_provenance(),
        "source_sha256": source_provenance(),
        "cases": cases,
    }
    artifact["status"] = (
        "success" if all(case["acceptance"]["passed"] for case in cases) else "quality_failure"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"[tq3] wrote {args.output}", flush=True)
    return 0 if artifact["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
