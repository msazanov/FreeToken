#!/usr/bin/env python3
"""Live public acceptance gate for HuggingVoice's 10/11-tool browser sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

if __package__:
    from .gemma_speaker_memory_acceptance import (
        TOOLS as SPEAKER_MEMORY_TOOLS,
        VOICE_SYSTEM_PROMPT,
        _speaker_context,
    )
else:
    from gemma_speaker_memory_acceptance import (  # type: ignore[import-not-found]
        TOOLS as SPEAKER_MEMORY_TOOLS,
        VOICE_SYSTEM_PROMPT,
        _speaker_context,
    )


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


WEB_SEARCH = _function(
    "web_search",
    "Search the web for current or factual information you don't already know "
    "(news, prices, facts, documentation). Returns the top results with titles, "
    "snippets and URLs.",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    },
)
CAMERA_SNAPSHOT = _function(
    "camera_snapshot",
    "Capture the current frame from the user's webcam so you can see what they are showing "
    "you. Use it whenever the user refers to something visual or asks you to look.",
    {"type": "object", "properties": {}, "required": []},
)

# Browser order matters: demo/main.js adds optional browser tools before server memory tools.
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "remember_name_10",
        "expected_tool": "speaker_memory_remember_name",
        "user": "Меня зовут Марат. Запомни моё имя.",
        "tools": [WEB_SEARCH, *SPEAKER_MEMORY_TOOLS],
    },
    {
        "name": "web_search_10",
        "expected_tool": "web_search",
        "user": "Поищи в интернете актуальную погоду в Москве.",
        "tools": [WEB_SEARCH, *SPEAKER_MEMORY_TOOLS],
    },
    {
        "name": "camera_snapshot_11",
        "expected_tool": "camera_snapshot",
        "user": "Посмотри, что я показываю в камеру.",
        "tools": [WEB_SEARCH, CAMERA_SNAPSHOT, *SPEAKER_MEMORY_TOOLS],
    },
)


def _validate_response(
    body: object, expected_tool: str, speaker_ref: str
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(body, dict):
        return False, {}
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    calls = message.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    call = calls[0] if len(calls) == 1 and isinstance(calls[0], dict) else {}
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    raw_arguments = function.get("arguments")
    decoded: object = None
    if isinstance(raw_arguments, str) and raw_arguments:
        try:
            decoded = json.loads(raw_arguments)
        except json.JSONDecodeError:
            pass
    arguments_valid = isinstance(decoded, dict)
    arguments = decoded if arguments_valid else {}
    passed = (
        len(calls) == 1
        and call.get("type") == "function"
        and function.get("name") == expected_tool
        and choice.get("finish_reason") == "tool_calls"
        and message.get("content") in (None, "")
        and arguments_valid
    )
    if expected_tool == "speaker_memory_remember_name":
        passed = passed and arguments == {"name": "Марат", "speaker_ref": speaker_ref}
    elif expected_tool == "web_search":
        query = arguments.get("query")
        passed = (
            passed
            and isinstance(query, str)
            and "погод" in query.casefold()
            and "моск" in query.casefold()
        )
    elif expected_tool == "camera_snapshot":
        passed = passed and arguments == {}
    return passed, arguments


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        "python/freetoken/arbiter/proxy.py",
        "benchmarks/gemma_huggingvoice_tool_superset_acceptance.py",
    )
    proxy = root / source_paths[0]
    acceptance_script = root / source_paths[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *source_paths],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_head": head,
        "tracked_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "arbiter_proxy_sha256": hashlib.sha256(proxy.read_bytes()).hexdigest(),
        "acceptance_script_sha256": hashlib.sha256(acceptance_script.read_bytes()).hexdigest(),
        "source_paths": list(source_paths),
    }


def _probe(
    client: httpx.Client,
    url: str,
    model: str,
    scenario: dict[str, Any],
    run: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    speaker_ref = f"sr_superset_{scenario['name']}_{run:03d}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VOICE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _speaker_context(speaker_ref) + "\n" + scenario["user"],
            },
        ],
        "tools": scenario["tools"],
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, "thinking_mode": "disabled"},
    }
    started = time.perf_counter()
    try:
        response = client.post(url, json=payload)
        elapsed = time.perf_counter() - started
        try:
            decoded = response.json()
        except ValueError:
            decoded = {"error": {"message": "non-JSON response", "body": response.text[:2000]}}
    except httpx.HTTPError as exc:
        return {
            "run": run,
            "passed": False,
            "http_status": None,
            "latency_s": time.perf_counter() - started,
            "speaker_ref": speaker_ref,
            "response": {"error": {"message": str(exc), "type": type(exc).__name__}},
        }
    passed, arguments = _validate_response(decoded, scenario["expected_tool"], speaker_ref)
    return {
        "run": run,
        "passed": response.status_code == 200 and passed,
        "http_status": response.status_code,
        "latency_s": elapsed,
        "speaker_ref": speaker_ref,
        "arguments": arguments,
        "response": decoded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", default="gemma-4-e2b")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    url = f"{args.endpoint.rstrip('/')}/chat/completions"
    scenario_reports = []
    with httpx.Client(timeout=args.timeout) as client:
        for scenario in SCENARIOS:
            runs = [
                _probe(
                    client,
                    url,
                    args.model,
                    scenario,
                    run,
                    args.temperature,
                    args.max_tokens,
                )
                for run in range(1, args.runs + 1)
            ]
            latencies = [result["latency_s"] for result in runs]
            scenario_reports.append(
                {
                    "scenario": scenario["name"],
                    "expected_tool": scenario["expected_tool"],
                    "tool_count": len(scenario["tools"]),
                    "successes": sum(bool(result["passed"]) for result in runs),
                    "latency_p50_s": statistics.median(latencies),
                    "runs": runs,
                }
            )

    report = {
        "endpoint": args.endpoint,
        "model": args.model,
        "source": _source_provenance(),
        "runs_per_scenario": args.runs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "scenarios": scenario_reports,
    }
    report["passed"] = all(
        scenario["successes"] == args.runs for scenario in scenario_reports
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
