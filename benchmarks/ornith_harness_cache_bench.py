"""Measure FreeToken radix-cache reuse with a DeepSeek Harness-shaped conversation.

This runner deliberately does not claim to replay a private active DSH session:
the deployed telemetry plug-in exposes runtime statistics, not the raw request
body.  It does preserve DSH's stable coding-agent/system and Ornith tool
protocol prefix, then measures a cold request, byte-identical warm replay, and
a realistic continuation that adds one assistant turn plus a short user delta.

The server must be started with ``--enable-cache-report``.  Result artifacts
contain no prompt body, only request provenance, timings and server-reported
``usage.prompt_tokens_details.cached_tokens``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

try:  # ``python benchmarks/foo.py`` and ``import benchmarks.foo`` both work.
    from benchmarks.ornith_context_bench import (
        RuntimeSampler,
        _get_json,
        _git_identity,
        _runtime_parameters,
        _server_model,
        _stream_request,
        build_compression_prompt,
        parse_context_tiers,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script entry point
    from ornith_context_bench import (
        RuntimeSampler,
        _get_json,
        _git_identity,
        _runtime_parameters,
        _server_model,
        _stream_request,
        build_compression_prompt,
        parse_context_tiers,
    )


HARNESS_SYSTEM_PREFIX = """You are a coding agent powered by Ornith 1.5 35b model.
Your working directory is {{cwd}}.

Tool calls are protocol, not prose. Emit only catalogued tools in Qwen3-Coder
XML form: <tool_call><function=NAME>...</function></tool_call>. Close both
</function> and </tool_call>. After the final tool call, stop and wait for tool
results. Never print partial tool tags as assistant prose."""


def build_harness_shaped_messages(*, repository_prompt: str, case_tag: str) -> list[dict[str, str]]:
    """Build the stable-prefix form used for cold and exact-warm cache probes."""
    return [
        {"role": "system", "content": HARNESS_SYSTEM_PREFIX},
        {"role": "user", "content": repository_prompt},
        {
            "role": "user",
            "content": (
                f"CACHE_CASE={case_tag}\n"
                "Read the repository dossier and return exactly one concise, concrete fact."
            ),
        },
    ]


def append_turn(
    base_messages: list[dict[str, str]], prior_answer: str, append_tag: str
) -> list[dict[str, str]]:
    """Turn one response into a history prefix and append a deliberately tiny delta."""
    return [
        *base_messages,
        {"role": "assistant", "content": prior_answer},
        {
            "role": "user",
            "content": f"CACHE_APPEND={append_tag}\nContinue with one concise fact.",
        },
    ]


def cache_metrics(result: dict) -> dict[str, float | int]:
    """Use FreeToken's authoritative per-response cache accounting."""
    prompt_tokens = int(result.get("prompt_tokens") or 0)
    cached_tokens = min(prompt_tokens, max(0, int(result.get("cached_tokens") or 0)))
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "new_prompt_tokens": prompt_tokens - cached_tokens,
        "cache_hit_ratio": 0.0 if prompt_tokens == 0 else cached_tokens / prompt_tokens,
    }


def result_directory(output_dir: Path, date: str, run_label: str | None) -> Path:
    """Keep diagnostic runs from colliding with canonical cache evidence."""
    suffix = "" if not run_label else f"-{run_label}"
    return output_dir / f"{date}-ornith-harness-cache{suffix}"


def _run_request(
    *, origin: str, model: str, messages: list[dict[str, str]], timeout_s: float,
    output_tokens: int, reasoning_effort: str, ignore_eos: bool,
) -> tuple[dict, list[dict]]:
    sampler = RuntimeSampler(origin)
    sampler.start()
    try:
        result = _stream_request(
            origin,
            {
                "model": model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": output_tokens,
                "temperature": 0,
                "reasoning_effort": reasoning_effort,
                "ignore_eos": ignore_eos,
            },
            timeout_s,
        )
    finally:
        sampler.stop()
    result["cache"] = cache_metrics(result)
    return result, sampler.samples


def run_tier(
    *, origin: str, root: Path, model: str, model_path: str, requested_tokens: int,
    timeout_s: float, output_tokens: int, reasoning_effort: str, case_tag: str,
    ignore_eos: bool = False,
) -> dict:
    repository_prompt, dossier_tokens = build_compression_prompt(
        root=root, model_path=model_path, requested_tokens=requested_tokens
    )
    base_messages = build_harness_shaped_messages(
        repository_prompt=repository_prompt, case_tag=f"{case_tag}-cold"
    )
    before = {"stats": _get_json(origin, "/v1/stats"), "cache": _get_json(origin, "/v1/cache/status")}
    cold, cold_samples = _run_request(
        origin=origin, model=model, messages=base_messages, timeout_s=timeout_s,
        output_tokens=output_tokens, reasoning_effort=reasoning_effort, ignore_eos=ignore_eos,
    )
    warm, warm_samples = _run_request(
        origin=origin, model=model, messages=base_messages, timeout_s=timeout_s,
        output_tokens=output_tokens, reasoning_effort=reasoning_effort, ignore_eos=ignore_eos,
    )
    append_messages = append_turn(base_messages, cold["response_text"], f"{case_tag}-append")
    append, append_samples = _run_request(
        origin=origin, model=model, messages=append_messages, timeout_s=timeout_s,
        output_tokens=output_tokens, reasoning_effort=reasoning_effort, ignore_eos=ignore_eos,
    )
    after = {"stats": _get_json(origin, "/v1/stats"), "cache": _get_json(origin, "/v1/cache/status")}
    return {
        "schema": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scenario": "harness-shaped-radix-cache",
        "requested_context_tokens": requested_tokens,
        "dossier_body_tokens": dossier_tokens,
        "prompt_provenance": {
            "exact_active_session_replay": False,
            "reason": "DSH telemetry does not expose raw request bodies",
            "stable_prefix": "DSH coding-agent system prompt plus Ornith tool protocol",
            "history_shape": "cold, byte-identical warm, assistant-history append",
        },
        "model": model,
        "origin": origin,
        "turns": {
            "cold": cold,
            "warm": warm,
            "append": append,
        },
        "runtime_before": before,
        "runtime_after": after,
        "runtime_samples": {"cold": cold_samples, "warm": warm_samples, "append": append_samples},
        "slice": {
            "git": _git_identity(root),
            "runtime_parameters": _runtime_parameters(before),
            "sampling": {"mode": "greedy-argmax", "temperature": 0.0, "seed": None,
                         "reasoning_effort": reasoning_effort, "ignore_eos": ignore_eos},
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:1919")
    parser.add_argument("--tiers", default="1k,16k")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--reasoning-effort", choices=("off", "on"), default="off")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="profile a fixed decode window; output is not a quality result")
    parser.add_argument("--case-prefix", default="harness-cache")
    parser.add_argument("--run-label", default=None,
                        help="separate artifact directory, required for nonstandard diagnostics")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, model_path = _server_model(args.origin)
    results_dir = result_directory(
        args.output_dir, datetime.now(UTC).strftime("%Y-%m-%d"), args.run_label
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    for requested_tokens in parse_context_tiers(args.tiers):
        row = run_tier(
            origin=args.origin, root=args.repo_root.resolve(), model=model, model_path=model_path,
            requested_tokens=requested_tokens, timeout_s=args.timeout_s,
            output_tokens=args.max_output_tokens, reasoning_effort=args.reasoning_effort,
            case_tag=f"{args.case_prefix}-{requested_tokens}", ignore_eos=args.ignore_eos,
        )
        path = results_dir / f"cache-{requested_tokens}.json"
        row["artifact"] = str(path)
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cold, warm, append = (row["turns"][name] for name in ("cold", "warm", "append"))
        print(
            f"{requested_tokens}: cold={cold['cache']['cache_hit_ratio']:.1%} "
            f"warm={warm['cache']['cache_hit_ratio']:.1%} "
            f"append={append['cache']['cache_hit_ratio']:.1%} artifact={path}",
            flush=True,
        )
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
