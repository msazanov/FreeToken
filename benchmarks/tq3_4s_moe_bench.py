#!/usr/bin/env python3
"""Resident-slot top-8 TQ3_4S MoE microbenchmark on the target SM75 GPU.

The geometry is Ornith 1.5 35B's routed SwiGLU: H=2048, I=512, top-8.  All
eight selected expert slots are already resident in VRAM, so this isolates the
packed gate/up + activation + down + routing accumulation path.  It deliberately
does not measure cache misses, PCIe, NVMe, prefill, or model tokens/second.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tq3_4s_kernel_bench import (  # noqa: E402
    git_provenance,
    make_packed_matrix,
    nvidia_smi_snapshot,
    quality_acceptance,
    quality_metrics,
    summarize_samples,
)


ORNITH_MOE_GEOMETRY = {
    "hidden": 2048,
    "intermediate": 512,
    "top_k": 8,
    "resident_slots": 8,
    "tokens": 1,
}
SOURCE_INPUTS = (
    "benchmarks/tq3_4s_moe_bench.py",
    "benchmarks/tq3_4s_kernel_bench.py",
    "python/freetoken/kernel/csrc/gguf/ggml-common.h",
    "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu",
    "python/freetoken/kernel/csrc/gguf/moe_vec.cuh",
    "python/freetoken/kernel/csrc/gguf/vecdotq.cuh",
    "python/freetoken/kernel/gguf.py",
    "python/freetoken/layers/activation.py",
    "python/freetoken/models/gguf/dequant.py",
    "python/freetoken/moe/fused_q4_0.py",
)


def source_provenance() -> dict[str, str]:
    return {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_INPUTS
    }


def packed_bank(slots: int, rows: int, columns: int, seed: int):
    import torch

    return torch.stack(
        [make_packed_matrix(rows, columns, seed + slot)[0] for slot in range(slots)]
    )


def exact_reference(hidden, gate_up_packed, down_packed, topk_ids, topk_weights):
    import torch
    import torch.nn.functional as F

    from freetoken.models.gguf.dequant import dequant_tq3_4s

    slots = gate_up_packed.shape[0]
    h = hidden.shape[1]
    i = down_packed.shape[-1] * 2
    gate_up = dequant_tq3_4s(gate_up_packed.flatten(), torch.float32).reshape(
        slots, 2 * i, h
    )
    down = dequant_tq3_4s(down_packed.flatten(), torch.float32).reshape(slots, h, i)
    routed = []
    for route, expert_tensor in enumerate(topk_ids[0]):
        expert = int(expert_tensor)
        projected = hidden[0].float() @ gate_up[expert].t()
        gate, up = projected.chunk(2)
        inter = F.silu(gate) * up
        routed.append(inter @ down[expert].t() * topk_weights[0, route])
    return torch.stack([sum(routed)])


def timed_cuda_samples(operation, *, warmup: int, iterations: int) -> list[float]:
    import torch

    result = None
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    if result is None:
        raise RuntimeError("MoE benchmark operation did not execute")
    return samples


def run_case(dtype, *, seed: int, warmup: int, iterations: int) -> dict[str, object]:
    import torch

    from freetoken.models.gguf.dequant import GGML_TQ3_4S
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    h = ORNITH_MOE_GEOMETRY["hidden"]
    i = ORNITH_MOE_GEOMETRY["intermediate"]
    top_k = ORNITH_MOE_GEOMETRY["top_k"]
    slots = ORNITH_MOE_GEOMETRY["resident_slots"]
    gate_up_cpu = packed_bank(slots, 2 * i, h, seed)
    down_cpu = packed_bank(slots, h, i, seed + 100)
    generator = torch.Generator().manual_seed(seed + 200)
    hidden_cpu = (torch.randn(1, h, generator=generator) * 0.25).to(dtype)
    topk_ids_cpu = torch.arange(top_k, dtype=torch.int32).reshape(1, top_k)
    topk_weights_cpu = torch.arange(1, top_k + 1, dtype=torch.float32).reshape(1, top_k)
    topk_weights_cpu /= topk_weights_cpu.sum()

    gate_up = gate_up_cpu.cuda()
    down = down_cpu.cuda()
    hidden = hidden_cpu.cuda()
    topk_ids = topk_ids_cpu.cuda()
    topk_weights = topk_weights_cpu.cuda()

    def operation():
        return fused_experts_gguf(
            hidden,
            gate_up,
            down,
            topk_weights,
            topk_ids,
            "silu",
            GGML_TQ3_4S,
        )

    actual = operation()
    torch.cuda.synchronize()
    reference = exact_reference(
        hidden_cpu, gate_up_cpu, down_cpu, topk_ids_cpu, topk_weights_cpu
    )
    quality = quality_metrics(actual, reference)
    acceptance = quality_acceptance(
        quality, min_cosine=0.999, max_relative_l2=0.03
    )
    timing = summarize_samples(
        timed_cuda_samples(operation, warmup=warmup, iterations=iterations)
    )

    gate_up_bytes = top_k * (2 * i) * (h // 2)
    down_bytes = top_k * h * (i // 2)
    packed_bytes = gate_up_bytes + down_bytes
    p50_ms = float(timing["p50_ms"])
    arithmetic_flops = top_k * (2 * (2 * i) * h + 2 * h * i)
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "seed": seed,
        "geometry": dict(ORNITH_MOE_GEOMETRY),
        "packed_weight_bytes_touched_per_call": packed_bytes,
        "packed_weight_mib_touched_per_call": packed_bytes / (1024**2),
        "timing": timing,
        "derived": {
            "effective_packed_read_gib_s_at_p50": packed_bytes
            / (p50_ms / 1000.0)
            / (1024**3),
            "nominal_matmul_tflop_s_at_p50": arithmetic_flops / (p50_ms * 1e9),
        },
        "quality_vs_exact_fp32_cpu": quality,
        "acceptance": acceptance,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable artifact: {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        raise SystemExit(f"this target benchmark requires exact SM75, got {capability}")
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must both be positive")

    before = nvidia_smi_snapshot()
    cases = []
    for index, dtype in enumerate((torch.float16, torch.bfloat16)):
        print(f"[tq3-moe] {dtype}", flush=True)
        case = run_case(
            dtype,
            seed=args.seed + index * 1000,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        cases.append(case)
        print(
            f"[tq3-moe] p50={case['timing']['p50_ms']:.6f} ms "
            f"rel_l2={case['quality_vs_exact_fp32_cpu']['relative_l2']:.6f}",
            flush=True,
        )

    artifact = {
        "schema_version": 1,
        "kind": "tq3_4s_sm75_resident_top8_moe_microbenchmark",
        "scope": "one Ornith routed SwiGLU layer with eight resident expert slots",
        "not_measured": [
            "model tokens/second",
            "expert cache transfer or miss latency",
            "PCIe or NVMe",
            "prefill throughput",
            "end-to-end model quality",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "parameters": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "base_seed": args.seed,
            "min_cosine": 0.999,
            "max_relative_l2": 0.03,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "hardware": {
            "cuda_device": torch.cuda.get_device_name(),
            "compute_capability": list(capability),
            "before": before,
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
    print(f"[tq3-moe] wrote {args.output}", flush=True)
    return 0 if artifact["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
