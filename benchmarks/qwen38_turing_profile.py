"""Reproducible, prompt-private Qwen3.8 route/copy profiling runner.

Each context point starts the requested Qwen process on its own explicit local
port and stores only logs and aggregate JSON beneath ``benchmarks/results``.
The runner owns and terminates only that child PID; it never discovers or stops
other FreeToken services (including Ornith).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_CONTEXT_POINTS = (1024, 16_384, 65_536, 114_688)


def parse_context_points(value: str) -> tuple[int, ...]:
    points: list[int] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        multiplier = 1024 if text.endswith("k") else 1
        digits = text[:-1] if multiplier != 1 else text
        try:
            point = int(digits) * multiplier
        except ValueError as exc:
            raise ValueError(f"invalid context point {item!r}") from exc
        if point <= 0:
            raise ValueError("context points must be positive")
        points.append(point)
    if not points:
        raise ValueError("at least one context point is required")
    return tuple(points)


def parse_proc_counters(io_record: str, stat_record: str) -> dict[str, int | None]:
    """Read only the process counters needed for a result record."""
    io_values: dict[str, int] = {}
    for line in io_record.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            try:
                io_values[key.strip()] = int(value.strip())
            except ValueError:
                pass
    # Field two is parenthesized and may contain spaces. The remainder starts at
    # field three (state); minflt and majflt are offsets 7 and 9 respectively.
    closing = stat_record.rfind(")")
    fields = stat_record[closing + 1 :].split() if closing >= 0 else []
    minor_faults = _int_at(fields, 7)
    major_faults = _int_at(fields, 9)
    return {
        "io_read_bytes": io_values.get("read_bytes"),
        "major_faults": major_faults,
        "minor_faults": minor_faults,
    }


def process_counter_delta(
    before: dict[str, int | None] | None, after: dict[str, int | None] | None
) -> dict[str, int | None]:
    if before is None or after is None:
        return {"io_read_bytes": None, "major_faults": None, "minor_faults": None}
    return {
        name: None
        if before.get(name) is None or after.get(name) is None
        else max(0, int(after[name]) - int(before[name]))
        for name in ("io_read_bytes", "major_faults", "minor_faults")
    }


def read_proc_counters(pid: int) -> dict[str, int | None] | None:
    try:
        return parse_proc_counters(
            Path(f"/proc/{pid}/io").read_text(encoding="utf-8"),
            Path(f"/proc/{pid}/stat").read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        return None


def unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def result_directory(repo_root: Path, run_name: str | None = None) -> Path:
    root = repo_root.resolve()
    base = (root / "benchmarks" / "results").resolve()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-qwen38-%H%M%S")
    destination = (base / (run_name or stamp)).resolve()
    if destination.parent != base:
        raise ValueError("result directory must be directly under benchmarks/results")
    return destination


def child_environment(
    repo_root: Path, *, inherited: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the Qwen child environment without request-body logging enabled."""
    environment = dict(os.environ if inherited is None else inherited)
    # The parent may deliberately log API traffic, but this runner constructs a
    # long prompt in memory and must not let that body escape its result tree.
    environment.pop("FREETOKEN_API_LOG_DIR", None)
    source_root = str(repo_root.resolve() / "python")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def _int_at(values: list[str], index: int) -> int | None:
    try:
        return int(values[index])
    except (IndexError, ValueError):
        return None


def _get_json(origin: str, endpoint: str, timeout_s: float = 15.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{origin}{endpoint}", timeout=timeout_s) as response:
        return json.load(response)


def _wait_for_server(origin: str, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Qwen process exited during startup with {process.returncode}")
        try:
            health = _get_json(origin, "/health", timeout_s=2)
        except (OSError, ValueError):
            time.sleep(0.25)
            continue
        if health.get("status") == "ok":
            return
        if health.get("status") == "error":
            raise RuntimeError(f"Qwen health reported startup error at {origin}: {health!r}")
        time.sleep(0.25)
    raise TimeoutError(f"Qwen did not become ready at {origin}")


def _sample_gpu() -> dict[str, float | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        util, memory = (item.strip() for item in output.splitlines()[0].split(","))
        return {"utilization_percent": float(util), "memory_used_mib": float(memory)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"utilization_percent": None, "memory_used_mib": None}


class ProcessSampler:
    def __init__(self, pid: int, interval_s: float = 1.0) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[dict[str, float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append({"monotonic_s": time.monotonic(), **_sample_gpu()})
            self._stop.wait(self.interval_s)


def _stream_completion(
    origin: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, float | int | str | None]:
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    first_token_at: float | None = None
    last_token_at: float | None = None
    prompt_tokens = completion_tokens = 0
    response_digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            event = json.loads(body)
            usage = event.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            if any(
                isinstance((choice.get("delta") or {}).get(key), str)
                and (choice.get("delta") or {}).get(key)
                for choice in event.get("choices", [])
                for key in ("content", "reasoning_content")
            ):
                now = time.monotonic()
                first_token_at = first_token_at or now
                last_token_at = now
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                for key in ("content", "reasoning_content"):
                    text = delta.get(key)
                    if isinstance(text, str) and text:
                        response_digest.update(text.encode("utf-8"))
    finished = time.monotonic()
    ttft_s = None if first_token_at is None else first_token_at - started
    decode_window_s = None if last_token_at is None or first_token_at is None else last_token_at - first_token_at
    return {
        "elapsed_s": finished - started,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "response_sha256": response_digest.hexdigest(),
        "prompt_tps": None if not ttft_s else prompt_tokens / ttft_s,
        "decode_tps": None
        if not decode_window_s or completion_tokens < 2
        else (completion_tokens - 1) / decode_window_s,
    }


def _fixed_prompt(model_path: str, context_tokens: int) -> str:
    """Build a deterministic prompt in memory; callers never serialize it."""
    from freetoken.utils.hf import load_tokenizer

    tokenizer = load_tokenizer(model_path)
    body = "routing telemetry benchmark " * (context_tokens + 32)
    ids = tokenizer.encode(body, add_special_tokens=False)[:context_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False)


def _server_command(args: argparse.Namespace, port: int) -> list[str]:
    return [
        str(args.python),
        "-m",
        "freetoken.cli",
        "serve",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-running-requests",
        "1",
        "--moe-collect-stats",
        *args.server_arg,
    ]


def make_result_record(
    *,
    context_tokens: int,
    seed: int,
    port: int,
    pid: int,
    metrics: dict[str, float | int | str | None],
    process: dict[str, int | None],
    gpu_samples: list[dict[str, float | None]],
    server_stats: dict[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    """Build a prompt-private result record from aggregate stream metrics."""
    return {
        "schema": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "requested_context_tokens": context_tokens,
        "sampling": {"temperature": 0, "seed": seed},
        "server": {"host": "127.0.0.1", "port": port, "pid": pid},
        "metrics": metrics,
        "response_sha256": metrics["response_sha256"],
        "process": process,
        "gpu_samples": gpu_samples,
        "server_stats": server_stats,
        "artifact": str(artifact),
    }


def run_context_point(args: argparse.Namespace, context_tokens: int, results_dir: Path) -> Path:
    port = args.port or unused_local_port()
    origin = f"http://127.0.0.1:{port}"
    artifact = results_dir / f"context-{context_tokens}.json"
    stdout = (results_dir / f"context-{context_tokens}.stdout.log").open("wb")
    stderr = (results_dir / f"context-{context_tokens}.stderr.log").open("wb")
    environment = child_environment(args.repo_root)
    process = subprocess.Popen(
        _server_command(args, port),
        stdout=stdout,
        stderr=stderr,
        cwd=args.repo_root,
        env=environment,
    )
    try:
        _wait_for_server(origin, process, args.startup_timeout_s)
        prompt = _fixed_prompt(args.model_path, context_tokens)
        sampler = ProcessSampler(process.pid, args.sample_interval_s)
        before = read_proc_counters(process.pid)
        sampler.start()
        try:
            metrics = _stream_completion(
                origin,
                {
                    "model": args.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": args.decode_tokens,
                    "temperature": 0,
                    "seed": args.seed,
                    "reasoning_effort": "none",
                },
                args.request_timeout_s,
            )
        finally:
            sampler.stop()
        after = read_proc_counters(process.pid)
        # Terminal completion is the lifecycle boundary that publishes E1/E4.
        stats = _get_json(origin, "/v1/stats", timeout_s=15)
        row = make_result_record(
            context_tokens=context_tokens,
            seed=args.seed,
            port=port,
            pid=process.pid,
            metrics=metrics,
            process=process_counter_delta(before, after),
            gpu_samples=sampler.samples,
            server_stats=stats,
            artifact=artifact,
        )
        artifact.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        stdout.close()
        stderr.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="qwen38-turing-profile")
    parser.add_argument("--contexts", default="1k,16k,64k,112k")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--port", type=int)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-name")
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=7200.0)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contexts = parse_context_points(args.contexts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    results_dir = result_directory(args.repo_root, args.run_name)
    results_dir.mkdir(parents=True, exist_ok=False)
    for context_tokens in contexts:
        artifact = run_context_point(args, context_tokens, results_dir)
        print(artifact, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
