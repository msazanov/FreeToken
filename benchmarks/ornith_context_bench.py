"""Single-request long-context and repository-compaction benchmark for Ornith.

The runner exercises the already-running OpenAI-compatible FreeToken server. It
measures client-visible SSE TTFT, decode rate, exact usage returned by the
server, cache/runtime snapshots, GPU telemetry and host RAM/swap while asking
Ornith to compress real source from this repository. Results are JSON artifacts
under ``benchmarks/results/``; it never writes prompts or model data to /tmp.

Example:

    PYTHONPATH=python .venv/bin/python benchmarks/ornith_context_bench.py \
      --origin http://127.0.0.1:1919 \
      --tiers 1k,16k,64k,112k
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_TIERS = (1024, 16_384, 65_536, 114_688)
IGNORED_DIRS = {".beads", ".codegraph", ".git", ".venv", "__pycache__"}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".json", ".yaml", ".yml"}
PRIORITY_SOURCES = (
    "README.md",
    "AGENTS.md",
    "TESTLOG.md",
    "CHANGELOG.md",
    "python/freetoken/server/control_api.py",
    "python/freetoken/server/stats.py",
    "python/freetoken/attention/triton.py",
    "python/freetoken/kernel/triton/attention.py",
    "python/freetoken/kvcache/mha_pool.py",
)
REQUIRED_ANCHORS = ("ornith", "rtx 2070", "control_api.py", "build_stats", "tq4")
PRIORITY_ANCHORS = (
    ("README.md", "RTX 2070 fork mission"),
    ("AGENTS.md", "RTX 2070 experiment records"),
    ("TESTLOG.md", "existing TQ4"),
    ("CHANGELOG.md", "Accepted"),
    ("python/freetoken/server/control_api.py", "v1/stats"),
    ("python/freetoken/server/stats.py", "def build_stats"),
)


@dataclasses.dataclass(frozen=True)
class SSEEvent:
    generated_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def parse_context_tiers(value: str) -> tuple[int, ...]:
    """Parse a comma list such as ``1k,16K,64000`` into positive token counts."""
    result: list[int] = []
    for raw in value.split(","):
        text = raw.strip().lower()
        if not text:
            continue
        multiplier = 1024 if text.endswith("k") else 1
        digits = text[:-1] if multiplier != 1 else text
        try:
            tokens = int(digits) * multiplier
        except ValueError as exc:
            raise ValueError(f"invalid context tier {raw!r}") from exc
        if tokens <= 0:
            raise ValueError("context tiers must be positive")
        result.append(tokens)
    if not result:
        raise ValueError("at least one context tier is required")
    return tuple(result)


def _repository_files(root: Path, priority_paths: Iterable[str]) -> Iterator[Path]:
    yielded: set[Path] = set()
    for relative in priority_paths:
        path = root / relative
        if path.is_file():
            yielded.add(path)
            yield path
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in yielded:
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def build_repository_dossier(
    root: Path, *, max_chars: int, priority_paths: Iterable[str] = ()
) -> str:
    """Return a bounded, deterministic dossier of actual repository text files."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    used = 0
    for path in _repository_files(root, priority_paths):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        block = f"\n\n===== FILE: {relative} =====\n{source}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        chunks.append(block[:remaining])
        used += min(len(block), remaining)
    return "".join(chunks).lstrip()


def build_priority_evidence(root: Path, *, max_chars: int) -> str:
    """Extract small, real source windows that carry the compression task facts."""
    chunks: list[str] = []
    used = 0
    for relative, anchor in PRIORITY_ANCHORS:
        path = root / relative
        if not path.is_file() or used >= max_chars:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        start_at = source.lower().find(anchor.lower())
        if start_at < 0:
            continue
        # A short window is enough for a real citation and prevents README or a
        # large implementation file from crowding out the other evidence.
        excerpt_start = max(0, start_at - 120)
        excerpt = source[excerpt_start : start_at + 520]
        block = f"\n\n===== EVIDENCE FILE: {relative} =====\n{excerpt}"
        remaining = max_chars - used
        chunks.append(block[:remaining])
        used += min(len(block), remaining)
    return "".join(chunks).lstrip()


def dossier_budget(requested_tokens: int) -> int:
    """Reserve a fixed margin for the task and the checkpoint chat template."""
    return max(128, requested_tokens - 160)


def parse_sse_event(line: str) -> SSEEvent:
    """Parse a single OpenAI SSE ``data:`` line without depending on an SDK."""
    if not line.startswith("data:"):
        return SSEEvent()
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return SSEEvent()
    payload = json.loads(body)
    generated: list[str] = []
    for choice in payload.get("choices", []):
        delta = choice.get("delta") or {}
        for key in ("reasoning_content", "content"):
            value = delta.get(key)
            if isinstance(value, str):
                generated.append(value)
    usage = payload.get("usage") or {}
    return SSEEvent(
        generated_text="".join(generated),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def artifact_path(results_dir: Path, tokens: int, scenario: str) -> Path:
    return results_dir / f"{scenario}-{tokens}.json"


def _get_json(origin: str, path: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(f"{origin.rstrip('/')}{path}", timeout=timeout) as response:
        return json.load(response)


def _server_model(origin: str) -> tuple[str, str]:
    models = _get_json(origin, "/v1/models")
    card = models["data"][0]
    return str(card["id"]), str(card["root"])


def _fit_dossier_to_tokens(model_path: str, dossier: str, target_tokens: int) -> tuple[str, int]:
    """Use FreeToken's matching tokenizer; the server-reported usage remains authoritative."""
    from freetoken.utils.hf import load_tokenizer

    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(dossier, add_special_tokens=False)
    selected = ids[:target_tokens]
    return tokenizer.decode(selected, skip_special_tokens=False), len(selected)


def build_compression_prompt(
    *, root: Path, model_path: str, requested_tokens: int
) -> tuple[str, int]:
    # Budget conservatively for the task, chat-template delimiters and a small
    # response. The returned API usage records the exact final prompt length.
    evidence = build_priority_evidence(root, max_chars=min(2_800, requested_tokens * 4))
    dossier = evidence + "\n\n" + build_repository_dossier(
        root,
        max_chars=requested_tokens * 8,
        priority_paths=(),
    )
    body, body_tokens = _fit_dossier_to_tokens(
        model_path, dossier, dossier_budget(requested_tokens)
    )
    task = """

TASK — repository-context compression
Compress this FreeToken repository dossier into at most 12 concise bullets for a
new maintainer. Retain concrete evidence, not generic advice. State: (1) the
RTX 2070 target hardware and model, (2) the endpoint and function that expose
runtime stats, (3) the active TQ4 optimization, (4) one rejected optimization,
and (5) the files a maintainer must update after an experiment. Include exact
file paths where the dossier provides them.
""".strip()
    header = f"BENCHMARK_CONTEXT_TIER={requested_tokens}\nREPOSITORY DOSSIER:\n"
    return f"{header}{body}\n\n{task}", body_tokens


def _gpu_snapshot() -> dict:
    query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        fields = [item.strip() for item in output.splitlines()[0].split(",")]
        return {
            "gpu_util_percent": float(fields[0]),
            "gpu_memory_used_mib": float(fields[1]),
            "gpu_memory_total_mib": float(fields[2]),
            "gpu_temperature_c": float(fields[3]),
            "gpu_power_w": float(fields[4]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"gpu_snapshot_error": True}


def _host_snapshot() -> dict:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.split()[0]) * 1024
    return {
        "mem_total_bytes": values.get("MemTotal", 0),
        "mem_available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
    }


class RuntimeSampler:
    def __init__(self, origin: str, interval_s: float = 1.0):
        self.origin = origin
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = {"monotonic_s": time.monotonic(), **_gpu_snapshot(), **_host_snapshot()}
            try:
                sample["server_stats"] = _get_json(self.origin, "/v1/stats", timeout=5)
            except (OSError, ValueError):
                sample["server_stats_error"] = True
            self.samples.append(sample)
            self._stop.wait(self.interval_s)


def _stream_request(origin: str, payload: dict, timeout_s: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{origin.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    first_token_at: float | None = None
    last_token_at: float | None = None
    text: list[str] = []
    prompt_tokens = completion_tokens = 0
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            event = parse_sse_event(raw_line.decode("utf-8").strip())
            if event.generated_text:
                now = time.monotonic()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                text.append(event.generated_text)
            prompt_tokens = event.prompt_tokens or prompt_tokens
            completion_tokens = event.completion_tokens or completion_tokens
    finished = time.monotonic()
    return {
        "wall_s": finished - started,
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "decode_window_s": None
        if first_token_at is None or last_token_at is None
        else last_token_at - first_token_at,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "response_text": "".join(text),
    }


def _quality(result: dict) -> dict:
    response = result["response_text"].lower()
    found = [anchor for anchor in REQUIRED_ANCHORS if anchor in response]
    return {"required_anchors": list(REQUIRED_ANCHORS), "found_anchors": found, "score": len(found)}


def run_tier(
    *, origin: str, root: Path, model: str, model_path: str, requested_tokens: int,
    timeout_s: float, output_tokens: int, results_dir: Path
) -> dict:
    prompt, dossier_tokens = build_compression_prompt(
        root=root, model_path=model_path, requested_tokens=requested_tokens
    )
    sampler = RuntimeSampler(origin)
    before = {"stats": _get_json(origin, "/v1/stats"), "cache": _get_json(origin, "/v1/cache/status")}
    sampler.start()
    try:
        result = _stream_request(
            origin,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": output_tokens,
                "temperature": 0,
                "reasoning_effort": "none",
            },
            timeout_s,
        )
    finally:
        sampler.stop()
    after = {"stats": _get_json(origin, "/v1/stats"), "cache": _get_json(origin, "/v1/cache/status")}
    decode_window = result["decode_window_s"]
    decode_tps = None
    if decode_window and decode_window > 0 and result["completion_tokens"] > 1:
        decode_tps = (result["completion_tokens"] - 1) / decode_window
    row = {
        "schema": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scenario": "repository-compression",
        "requested_context_tokens": requested_tokens,
        "dossier_body_tokens": dossier_tokens,
        "model": model,
        "origin": origin,
        "metrics": {**result, "prefill_tps_estimate": None if not result["ttft_s"] else result["prompt_tokens"] / result["ttft_s"], "decode_tps": decode_tps},
        "quality": _quality(result),
        "runtime_before": before,
        "runtime_after": after,
        "runtime_samples": sampler.samples,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_path(results_dir, requested_tokens, "compression")
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    row["artifact"] = str(path)
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:1919")
    parser.add_argument("--tiers", default="1k,16k,64k,112k")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, model_path = _server_model(args.origin)
    for tier in parse_context_tiers(args.tiers):
        row = run_tier(
            origin=args.origin,
            root=args.repo_root.resolve(),
            model=model,
            model_path=model_path,
            requested_tokens=tier,
            timeout_s=args.timeout_s,
            output_tokens=args.max_output_tokens,
            results_dir=args.output_dir,
        )
        metrics = row["metrics"]
        print(
            f"{tier}: prompt={metrics['prompt_tokens']} TTFT={metrics['ttft_s']:.2f}s "
            f"decode={metrics['decode_tps']} tok/s quality={row['quality']['score']}/"
            f"{len(REQUIRED_ANCHORS)} artifact={row['artifact']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
