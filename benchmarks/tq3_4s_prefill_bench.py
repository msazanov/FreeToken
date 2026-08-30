#!/usr/bin/env python3
"""TQ3_4S prefill sweep at real Ornith dense/MoE layer geometry on SM75.

This is a layer microbenchmark, not model tokens/second.  It records the exact
dispatch selected at the batch-six boundary, every CUDA-event sample and peak
PyTorch VRAM allocation.  All routed experts are resident; cache misses, PCIe,
NVMe, attention/GDN and the scheduler are intentionally outside this slice.
"""

from __future__ import annotations

import argparse
import gc
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
    summarize_samples,
)


PREFILL_TOKEN_SWEEP = (1, 6, 7, 16, 64, 128, 256, 512, 1024)
ORNITH_DENSE_GEOMETRY = {"input": 2048, "output": 1024}
ORNITH_MOE_GEOMETRY = {
    "hidden": 2048,
    "intermediate": 512,
    "top_k": 8,
    "resident_slots": 8,
}
SOURCE_INPUTS = (
    "benchmarks/tq3_4s_prefill_bench.py",
    "benchmarks/tq3_4s_kernel_bench.py",
    "python/freetoken/layers/gguf.py",
    "python/freetoken/moe/fused_q4_0.py",
    "python/freetoken/models/gguf/dequant.py",
    "python/freetoken/kernel/gguf.py",
    "python/freetoken/kernel/csrc/gguf/ggml-common.h",
    "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu",
    "python/freetoken/kernel/csrc/gguf/moe_vec.cuh",
    "python/freetoken/kernel/csrc/gguf/vecdotq.cuh",
)


def source_provenance() -> dict[str, str]:
    return {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_INPUTS
    }


def dense_dispatch(tokens: int) -> str:
    return "packed_mmvq" if tokens <= 6 else "exact_materialized_dequant_plus_gemm"


def moe_dispatch(_tokens: int) -> str:
    return "packed_selected_expert_mmvq"


def _packed_bank(slots: int, rows: int, columns: int, seed: int):
    import torch

    return torch.stack(
        [make_packed_matrix(rows, columns, seed + slot)[0] for slot in range(slots)]
    )


def _timed_cuda_samples(operation, *, warmup: int, iterations: int):
    import torch

    result = None
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()
    del result
    gc.collect()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()

    samples = []
    output = None
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        del output
    torch.cuda.synchronize()
    return samples, {
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_allocated_delta_bytes": int(torch.cuda.max_memory_allocated())
        - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_reserved_delta_bytes": int(torch.cuda.max_memory_reserved())
        - baseline_reserved,
    }


def _case_result(kind: str, tokens: int, dispatch: str, operation, *, warmup: int, iterations: int):
    import torch

    probe = operation()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(probe).all().item()):
        raise RuntimeError(f"{kind} tokens={tokens}: non-finite output")
    output_shape = list(probe.shape)
    output_abs_mean = float(probe.float().abs().mean().item())
    del probe
    samples, memory = _timed_cuda_samples(
        operation, warmup=warmup, iterations=iterations
    )
    timing = summarize_samples(samples)
    p50_ms = float(timing["p50_ms"])
    return {
        "kind": kind,
        "tokens": tokens,
        "dispatch": dispatch,
        "timing": timing,
        "derived": {
            "layer_input_tokens_per_second_at_p50": tokens / (p50_ms / 1000.0),
            "milliseconds_per_input_token_at_p50": p50_ms / tokens,
        },
        "memory": memory,
        "sanity": {
            "output_shape": output_shape,
            "output_abs_mean": output_abs_mean,
            "all_finite": True,
        },
    }


def run_dense_sweep(dtype, tokens_sweep, *, seed: int, warmup: int, iterations: int):
    import torch

    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_TQ3_4S

    columns = ORNITH_DENSE_GEOMETRY["input"]
    rows = ORNITH_DENSE_GEOMETRY["output"]
    packed_cpu, _ = make_packed_matrix(rows, columns, seed)
    packed = packed_cpu.cuda()
    packed_bytes = packed.numel()
    materialized_bytes = rows * columns * torch.empty((), dtype=dtype).element_size()
    cases = []
    for tokens in tokens_sweep:
        generator = torch.Generator().manual_seed(seed + tokens)
        activation = (torch.randn(tokens, columns, generator=generator) * 0.25).to(
            dtype
        ).cuda()

        def operation():
            return fused_mul_mat_gguf(activation, packed, GGML_TQ3_4S)

        case = _case_result(
            "dense_gate_up",
            tokens,
            dense_dispatch(tokens),
            operation,
            warmup=warmup,
            iterations=iterations,
        )
        case["geometry"] = {**ORNITH_DENSE_GEOMETRY, "tokens": tokens}
        case["packed_weight_bytes"] = packed_bytes
        case["exact_materialized_weight_bytes"] = materialized_bytes
        cases.append(case)
        print(
            f"[tq3-prefill] dense tokens={tokens} dispatch={case['dispatch']} "
            f"p50={case['timing']['p50_ms']:.6f} ms",
            flush=True,
        )
        del activation
    return cases


def run_moe_sweep(dtype, tokens_sweep, *, seed: int, warmup: int, iterations: int):
    import torch

    from freetoken.models.gguf.dequant import GGML_TQ3_4S
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    h = ORNITH_MOE_GEOMETRY["hidden"]
    i = ORNITH_MOE_GEOMETRY["intermediate"]
    top_k = ORNITH_MOE_GEOMETRY["top_k"]
    slots = ORNITH_MOE_GEOMETRY["resident_slots"]
    gate_up = _packed_bank(slots, 2 * i, h, seed).cuda()
    down = _packed_bank(slots, h, i, seed + 100).cuda()
    packed_bytes_per_token = gate_up.numel() + down.numel()
    cases = []
    for tokens in tokens_sweep:
        generator = torch.Generator().manual_seed(seed + 1000 + tokens)
        hidden = (torch.randn(tokens, h, generator=generator) * 0.25).to(dtype).cuda()
        base_ids = torch.arange(top_k, dtype=torch.int32)
        ids = torch.stack([base_ids.roll(token % top_k) for token in range(tokens)]).cuda()
        raw_weights = torch.rand(tokens, top_k, generator=generator)
        weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).cuda()

        def operation():
            return fused_experts_gguf(
                hidden,
                gate_up,
                down,
                weights,
                ids,
                "silu",
                GGML_TQ3_4S,
            )

        case = _case_result(
            "resident_top8_moe",
            tokens,
            moe_dispatch(tokens),
            operation,
            warmup=warmup,
            iterations=iterations,
        )
        case["geometry"] = {**ORNITH_MOE_GEOMETRY, "tokens": tokens}
        case["packed_weight_bytes_touched_per_input_token"] = packed_bytes_per_token
        cases.append(case)
        print(
            f"[tq3-prefill] moe tokens={tokens} dispatch={case['dispatch']} "
            f"p50={case['timing']['p50_ms']:.6f} ms",
            flush=True,
        )
        del hidden, ids, weights
    return cases


def _parse_tokens(value: str) -> tuple[int, ...]:
    tokens = tuple(int(part) for part in value.split(","))
    if not tokens or any(token <= 0 for token in tokens):
        raise argparse.ArgumentTypeError("tokens must be a non-empty comma list of positive ints")
    return tokens


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tokens", type=_parse_tokens, default=PREFILL_TOKEN_SWEEP
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable artifact: {args.output}")
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must both be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        raise SystemExit(f"this target benchmark requires exact SM75, got {capability}")
    dtype = getattr(torch, args.dtype)

    before = nvidia_smi_snapshot()
    cases = []
    status = "success"
    failure = None
    try:
        cases.extend(
            run_dense_sweep(
                dtype,
                args.tokens,
                seed=args.seed,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
        cases.extend(
            run_moe_sweep(
                dtype,
                args.tokens,
                seed=args.seed + 10_000,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
    except torch.OutOfMemoryError as exc:
        status = "oom"
        failure = repr(exc)
        torch.cuda.empty_cache()

    artifact = {
        "schema_version": 1,
        "kind": "tq3_4s_sm75_prefill_layer_sweep",
        "scope": "real Ornith dense projection and resident top-8 MoE layer",
        "not_measured": [
            "model tokens/second",
            "expert cache transfer or misses",
            "PCIe or NVMe",
            "attention or GDN",
            "scheduler or sampling",
            "end-to-end model quality",
        ],
        "correctness_gates": [
            "tests/kernels/test_gguf_tq3_4s.py::test_tq3_4s_large_batch_prefill_fallback_matches_exact_reference",
            "tests/moe/test_tq3_4s_moe.py::test_tq3_4s_top8_prefill_matches_exact_routed_reference",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "parameters": {
            "tokens": list(args.tokens),
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
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
        "status": status,
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"[tq3-prefill] status={status} cases={len(cases)} wrote {args.output}")
    return 0 if status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
