"""Single-request long-context and repository-compaction benchmark for Ornith.

The runner exercises the already-running OpenAI-compatible FreeToken server. It
measures client-visible SSE TTFT, decode rate, exact usage returned by the
server, cache/runtime snapshots, GPU telemetry and host RAM/swap while asking
Ornith to compress real source from this repository. Results are JSON artifacts
under ``benchmarks/results/``; it never writes prompts or model data to /tmp.

Example:

    PYTHONPATH=python .venv/bin/python benchmarks/ornith_context_bench.py \
      --origin http://127.0.0.1:1919 \
      --model-sha256 <precomputed-checkpoint-sha256> \
      --tiers 1k,16k,64k,112k
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
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
RUNTIME_SOURCE_INPUTS = (
    "benchmarks/ornith_context_bench.py",
    "python/freetoken/engine/engine.py",
    "python/freetoken/attention/triton.py",
    "python/freetoken/kvcache/mha_pool.py",
    "python/freetoken/models/qwen3_5_moe/gguf.py",
    "python/freetoken/kernel/gguf.py",
    "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu",
    "python/freetoken/kernel/csrc/gguf/moe_vec.cuh",
    "python/freetoken/kernel/csrc/gguf/vecdotq.cuh",
)


@dataclasses.dataclass(frozen=True)
class SSEEvent:
    generated_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None


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
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    return SSEEvent(
        generated_text="".join(generated),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cached_tokens=None if cached_tokens is None else int(cached_tokens),
    )


def artifact_path(results_dir: Path, tokens: int, scenario: str) -> Path:
    return results_dir / f"{scenario}-{tokens}.json"


def parse_parameters(values: Iterable[str]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` labels used to identify a benchmark slice."""
    parameters: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"parameter must use KEY=VALUE, got {raw!r}")
        parameters[key] = value
    return parameters


def _git_identity(root: Path) -> dict[str, object]:
    """Capture the exact source revision, including uncommitted experiment code."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True, timeout=5
        ).strip()
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", "."],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).stdout
        untracked_raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            timeout=10,
        )
        untracked_sha256 = {}
        for raw_relative in untracked_raw.split(b"\0"):
            if not raw_relative:
                continue
            relative = raw_relative.decode("utf-8", errors="surrogateescape")
            path = root / relative
            if path.is_file():
                untracked_sha256[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "dirty": None,
            "status": [],
            "working_tree_diff_sha256": None,
            "untracked_sha256": {},
        }
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_sha256": untracked_sha256,
    }


def _source_identity(root: Path) -> dict[str, str | None]:
    """Hash runtime inputs, including untracked benchmark code absent from git diff."""
    return {
        relative: (
            hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if (root / relative).is_file()
            else None
        )
        for relative in RUNTIME_SOURCE_INPUTS
    }


def _software_identity() -> dict[str, str | None]:
    import torch

    try:
        import triton

        triton_version = triton.__version__
    except (ImportError, AttributeError):
        triton_version = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
    }


def _hardware_identity() -> dict[str, str | None]:
    """Read GPU identity without creating a CUDA context in the benchmark client."""
    query = "name,uuid,driver_version,compute_cap"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        name, uuid, driver, compute_capability = (
            part.strip() for part in output.splitlines()[0].split(",", 3)
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {
            "gpu_name": None,
            "gpu_uuid": None,
            "driver": None,
            "compute_capability": None,
        }
    return {
        "gpu_name": name,
        "gpu_uuid": uuid,
        "driver": driver,
        "compute_capability": compute_capability,
    }


def _model_identity(
    model_path: str, *, content_sha256: str, revision: str | None
) -> dict[str, object]:
    """Pin model content supplied by the intake gate plus local file stat evidence.

    The runner requires a precomputed content digest instead of reading tens of
    GiB during the timed context sweep and perturbing the host page cache.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
        raise ValueError("model content SHA-256 must be 64 lowercase hex characters")
    root = Path(model_path).resolve()
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(path for path in root.rglob("*") if path.is_file())
    else:
        raise ValueError(f"model path does not exist: {root}")
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in candidates
    ]
    inferred_revision = revision
    if inferred_revision is None and "snapshots" in root.parts:
        snapshot_index = root.parts.index("snapshots")
        if snapshot_index + 1 < len(root.parts):
            inferred_revision = root.parts[snapshot_index + 1]
    return {
        "path": str(root),
        "content_sha256": content_sha256,
        "revision": inferred_revision,
        "files": files,
    }


def _runtime_parameters(runtime_before: dict) -> dict[str, object]:
    """Keep the serving configuration alongside manual ``--parameter`` labels."""
    stats = runtime_before.get("stats") or {}
    cache = runtime_before.get("cache") or {}
    geometry = cache.get("geometry") or {}
    kv = stats.get("kv") or {}
    return {
        "context_budget_tokens": (stats.get("model") or {}).get("ctx"),
        "kv_dtype": kv.get("dtype"),
        "kv_pages": kv.get("total_pages"),
        "moe_cache_slots": geometry.get("moe_cache_size"),
        "mamba_slots": geometry.get("num_mamba_slots"),
    }


def slice_index_entry(row: dict) -> dict:
    """Flatten one artifact into a plot-ready point for revision/parameter comparisons."""
    metrics = row["metrics"]
    slice_meta = row["slice"]
    return {
        "timestamp_utc": row["timestamp_utc"],
        "artifact": row["artifact"],
        "series": slice_meta["series"],
        "label": slice_meta["label"],
        "git_commit": slice_meta["git"]["commit"],
        "git_dirty": slice_meta["git"]["dirty"],
        "parameters": slice_meta["parameters"],
        "runtime_parameters": slice_meta["runtime_parameters"],
        "sampling_mode": slice_meta["sampling"]["mode"],
        "context_tokens": metrics["prompt_tokens"],
        "cached_tokens": metrics.get("cached_tokens", 0),
        "new_prompt_tokens": metrics.get("new_prompt_tokens", metrics["prompt_tokens"]),
        "is_cold_prefill": metrics.get("is_cold_prefill"),
        "requested_context_tokens": row["requested_context_tokens"],
        "ttft_s": metrics["ttft_s"],
        "prefill_tps": metrics["prefill_tps_estimate"],
        "decode_tps": metrics["decode_tps"],
    }


def prefill_metrics(result: dict) -> dict[str, object]:
    """Return plot-safe prompt metrics, excluding cache hits from cold prefill."""
    prompt_tokens = max(0, int(result.get("prompt_tokens") or 0))
    cached_value = result.get("cached_tokens")
    cached_tokens = None if cached_value is None else max(0, int(cached_value))
    new_prompt_tokens = (
        None if cached_tokens is None else max(0, prompt_tokens - cached_tokens)
    )
    ttft_s = result.get("ttft_s")
    naive_rate = None
    if ttft_s is not None and ttft_s > 0:
        naive_rate = prompt_tokens / ttft_s
    is_cold = None if cached_tokens is None else cached_tokens == 0
    return {
        "new_prompt_tokens": new_prompt_tokens,
        "is_cold_prefill": is_cold,
        "prefill_tps_estimate": naive_rate if is_cold is True else None,
        "naive_total_prompt_over_ttft_tps": naive_rate,
    }


def append_slice_index(results_dir: Path, row: dict) -> Path:
    """Append a compact, plot-ready point without mutating past measurements."""
    path = results_dir / "slices.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(slice_index_entry(row), ensure_ascii=False, sort_keys=True) + "\n")
    return path


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
    cached_tokens: int | None = None
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
            if event.cached_tokens is not None:
                cached_tokens = event.cached_tokens
    finished = time.monotonic()
    return {
        "wall_s": finished - started,
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "decode_window_s": None
        if first_token_at is None or last_token_at is None
        else last_token_at - first_token_at,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "response_text": "".join(text),
    }


def _quality(result: dict) -> dict:
    response = result["response_text"].lower()
    found = [anchor for anchor in REQUIRED_ANCHORS if anchor in response]
    return {"required_anchors": list(REQUIRED_ANCHORS), "found_anchors": found, "score": len(found)}


def run_tier(
    *, origin: str, root: Path, model: str, model_path: str, requested_tokens: int,
    timeout_s: float, output_tokens: int, results_dir: Path, slice_series: str,
    slice_label: str, parameters: dict[str, str], model_identity: dict[str, object],
    hardware_identity: dict[str, str | None]
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
    path = artifact_path(results_dir, requested_tokens, "compression")
    row = {
        "schema": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scenario": "repository-compression",
        "requested_context_tokens": requested_tokens,
        "dossier_body_tokens": dossier_tokens,
        "model": model,
        "origin": origin,
        "metrics": {**result, **prefill_metrics(result), "decode_tps": decode_tps},
        "quality": _quality(result),
        "runtime_before": before,
        "runtime_after": after,
        "runtime_samples": sampler.samples,
        "slice": {
            "series": slice_series,
            "label": slice_label,
            "git": _git_identity(root),
            "parameters": parameters,
            "runtime_parameters": _runtime_parameters(before),
            # FreeToken's current OpenAI route does not forward a seed to its sampler.
            # temperature=0 selects argmax, so the result is deterministic without one.
            "sampling": {"mode": "greedy-argmax", "temperature": 0.0, "seed": None},
        },
        "provenance": {
            "software": _software_identity(),
            "hardware": hardware_identity,
            "model": model_identity,
            "source_sha256": _source_identity(root),
        },
        "artifact": str(path),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_slice_index(results_dir, row)
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:1919")
    parser.add_argument("--tiers", default="1k,16k,64k,112k")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--slice-series", default="ornith-rtx2070")
    parser.add_argument("--slice-label", default="baseline")
    parser.add_argument("--parameter", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--model-sha256",
        help="required precomputed model/checkpoint content SHA-256; never hash huge weights during a timed sweep",
    )
    parser.add_argument("--model-revision", help="optional Hugging Face revision or local model label")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        parameters = parse_parameters(args.parameter)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model, model_path = _server_model(args.origin)
    if args.model_sha256 is None:
        raise SystemExit(
            "--model-sha256 is required so benchmark points cannot outlive their model identity"
        )
    try:
        model_identity = _model_identity(
            model_path,
            content_sha256=args.model_sha256,
            revision=args.model_revision,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    hardware_identity = _hardware_identity()
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
            slice_series=args.slice_series,
            slice_label=args.slice_label,
            parameters=parameters,
            model_identity=model_identity,
            hardware_identity=hardware_identity,
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
