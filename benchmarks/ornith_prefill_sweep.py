"""Probe Ornith's largest stable chunked-prefill size on the RTX 2070 profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_ledger import append_event, classify_failure, make_failure_event, normalize_artifact
from benchmarks.qwen38_turing_profile import parse_context_points


def parse_chunk_values(value: str) -> tuple[int, ...]:
    try:
        chunks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("chunks must be comma-separated integers") from exc
    if not chunks or any(chunk <= 0 for chunk in chunks) or tuple(sorted(set(chunks))) != chunks:
        raise ValueError("chunks must be strictly ascending positive integers")
    return chunks


def _ornith_server_args(chunk: int) -> list[str]:
    pairs: list[str | None] = [
        "--served-model-name", "Ornith 1.5 35b", "--moe-backend", "offload",
        "--expert-load", "serial", "--memory-ratio", "0.85",
        "--max-running-requests", "1", "--max-seq-len-override", "122880",
        "--num-tokens", "122880", "--kv-reserve-tokens", "122880",
        "--moe-cache-auto", None, "--disable-moe-prefill-overlap", None,
        "--max-prefill-length", str(chunk), "--cuda-graph-max-bs", "1",
        "--disable-pynccl", None, "--cache-type", "radix",
        "--enable-cache-report", None, "--attention-backend", "triton",
        "--kv-cache-dtype", "int8",
    ]
    result: list[str] = []
    for value in pairs:
        result.extend(["--server-arg", value] if value is not None else [])
    return result


def run_sweep(args: argparse.Namespace) -> Path:
    chunks = parse_chunk_values(args.chunks)
    contexts = parse_context_points(args.contexts)
    if contexts != (args.context_tokens,):
        raise ValueError("the sweep currently requires exactly one --contexts value matching --context-tokens")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-ornith-prefill-sweep-%H%M%S")
    sweep_dir = (args.results_root / (args.run_name or stamp)).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=False)
    events: list[dict[str, Any]] = []
    for chunk in chunks:
        attempt_id = f"{sweep_dir.name}-p{chunk}"
        runner_name = f"{sweep_dir.name}-p{chunk}"
        command = [
            str(args.python), "benchmarks/qwen38_turing_profile.py",
            "--model-path", str(args.model_path), "--model-name", "Ornith 1.5 35b",
            "--contexts", args.contexts, "--decode-tokens", str(args.decode_tokens),
            "--seed", str(args.seed), "--run-name", runner_name,
            "--quantization", "Q4_K_M",
            "--runtime-profile", f"turing-q4km-int8kv-prefill{chunk}",
            "--trace-stride-events", str(args.trace_stride_events),
            "--python", str(args.python), *_ornith_server_args(chunk),
        ]
        if args.ignore_eos:
            command.append("--ignore-eos")
        stdout_log = sweep_dir / f"p{chunk}.runner.stdout.log"
        stderr_log = sweep_dir / f"p{chunk}.runner.stderr.log"
        completed = subprocess.run(
            command, cwd=args.repo_root,
            env={**os.environ, "PYTHONPATH": str(args.repo_root / "python")},
            text=True, capture_output=True, check=False,
        )
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        artifact = args.results_root / runner_name / f"context-{args.context_tokens}.json"
        parameters = {"max_prefill_length": chunk}
        if completed.returncode == 0 and artifact.exists():
            event = normalize_artifact(artifact)
            event.update({
                "attempt_id": attempt_id,
                "parameters": parameters,
                "runtime_profile": f"turing-q4km-int8kv-prefill{chunk}",
                "stdout_log": str(stdout_log), "stderr_log": str(stderr_log),
            })
        else:
            error = completed.stderr or completed.stdout or f"runner exited {completed.returncode}"
            event = make_failure_event(
                attempt_id=attempt_id, model="Ornith 1.5 35b",
                requested_context_tokens=args.context_tokens, parameters=parameters,
                failure_kind=classify_failure(error), error=error,
                stdout_log=str(stdout_log), stderr_log=str(stderr_log),
            )
        append_event(args.ledger, event)
        events.append(event)
        if event["status"] != "success" and args.stop_on_failure:
            break
    summary = {"schema": 1, "model": "Ornith 1.5 35b", "context_tokens": args.context_tokens, "events": events}
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sweep_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--contexts", default="16k")
    parser.add_argument("--context-tokens", type=int, default=16_384)
    parser.add_argument("--chunks", default="1024,1280,1536,1792,2048,2560,3072,4096")
    parser.add_argument("--decode-tokens", type=int, default=255)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-stride-events", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "benchmarks" / "results")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "benchmark-events.jsonl")
    parser.add_argument("--run-name")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(run_sweep(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
