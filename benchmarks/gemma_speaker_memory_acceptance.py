#!/usr/bin/env python3
"""Live Gemma/HuggingVoice tool-call and Russian TTFT acceptance gate.

This intentionally talks to the public arbiter.  It does not import HuggingVoice and does not
execute a tool; success means the real model selected ``speaker_memory_remember_name`` and the
OpenAI-compatible response preserved the structured call with the expected arguments.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx


VOICE_SYSTEM_PROMPT = """Ты голосовой помощник. По умолчанию отвечай по-русски. Отвечай по-английски, когда пользователь явно говорит по-английски. Не переключайся на английский только из-за кода, названий продуктов или английских технических терминов. Отвечай естественно, кратко и пригодно для озвучивания. Для естественной русской речи используй вопросительные и восклицательные знаки, запятые, тире и многоточия по смыслу: Silero v5.5 RU передаёт интонацию через пунктуацию и автоматически обрабатывает ударения и омографы. Смех и заминки пиши произносимо — например, «ха-ха», «хм», «эм». Не выводи сценические ремарки вроде «[смеётся]», команды шёпота, названия эмоций или SSML-теги: этот голосовой канал их не интерпретирует.

Работа с памятью голосов HuggingVoice:
- Считай voice_id вероятностным сходством, а не аутентификацией человека.
- Для state=ambiguous или conflict задай один короткий естественный уточняющий вопрос; при mixed не обучай память.
- После явного представления вызови speaker_memory_remember_name. После ответа на уточнение вызови speaker_memory_confirm или speaker_memory_reject.
- speaker_memory_inspect используй только когда нужно понять текущую связь.
- Личные факты сохраняй через speaker_memory_remember_fact и читай через speaker_memory_recall только для подтверждённого known-спикера.
- Для unknown, ambiguous, conflict или mixed не раскрывай и не угадывай личные факты.
- speaker_memory_forget вызывай только после явной просьбы удалить конкретный факт или все факты человека.
- Если пользователь явно говорит, что текущий голос — телевизор или нежелательный фон, вызови speaker_memory_block_voice. Никогда не блокируй голос по собственной догадке.
- После явного исправления ошибочной блокировки вызови speaker_memory_unblock_voice, пока speaker_ref ещё действителен.
- Имя употребляй естественно и не повторяй в каждом ответе. Никогда не заявляй, что распознавание голоса абсолютно точно.
"""

_REFERENCE = {
    "type": "string",
    "minLength": 4,
    "description": "Short-lived speaker reference from trusted HuggingVoice context.",
}
_PERSON = {
    "type": "string",
    "minLength": 3,
    "description": "Candidate person ID returned by speaker_memory_inspect.",
}


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


TOOLS = [
    _function(
        "speaker_memory_inspect",
        "Inspect the current speaker identity state before clarifying a name.",
        {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE},
            "required": ["speaker_ref"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_remember_name",
        "Remember a name explicitly given by the current speaker.",
        {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE,
                "name": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["speaker_ref", "name"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_confirm",
        "Confirm that the current speaker is the proposed person after an affirmative answer.",
        {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE, "person_id": _PERSON},
            "required": ["speaker_ref", "person_id"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_reject",
        "Reject a proposed person after the current speaker denies the match.",
        {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE, "person_id": _PERSON},
            "required": ["speaker_ref", "person_id"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_block_voice",
        "Ignore the current voice after the user explicitly identifies unwanted background.",
        {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE,
                "reason": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["speaker_ref", "reason"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_unblock_voice",
        "Remove the current voice from the background blacklist after an explicit correction.",
        {
            "type": "object",
            "properties": {"speaker_ref": _REFERENCE},
            "required": ["speaker_ref"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_remember_fact",
        "Remember a personal fact explicitly stated by the confirmed current speaker.",
        {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE,
                "fact": {"type": "string", "minLength": 1, "maxLength": 500},
                "topic": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["speaker_ref", "fact"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_recall",
        "Recall relevant private facts only for a confirmed current speaker.",
        {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE,
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["speaker_ref", "query"],
            "additionalProperties": False,
        },
    ),
    _function(
        "speaker_memory_forget",
        "Forget one fact or all facts after an explicit request from the confirmed speaker.",
        {
            "type": "object",
            "properties": {
                "speaker_ref": _REFERENCE,
                "scope": {"type": "string", "enum": ["fact", "facts"]},
                "fact_id": {"type": "string", "minLength": 3},
            },
            "required": ["speaker_ref", "scope"],
            "additionalProperties": False,
        },
    ),
]


def _speaker_context(speaker_ref: str) -> str:
    payload = {
        "speaker_ref": speaker_ref,
        "voice_id": "v_acceptance",
        "state": "unknown",
        "candidate": None,
        "margin": None,
        "recommendation": "ask_name",
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"<huggingvoice_speaker_context>{encoded}</huggingvoice_speaker_context>"


def _tool_probe(
    client: httpx.Client,
    url: str,
    model: str,
    run: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    speaker_ref = f"sr_acceptance_{run:03d}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VOICE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _speaker_context(speaker_ref)
                + "\nМеня зовут Марат. Запомни моё имя.",
            },
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, "thinking_mode": "disabled"},
    }
    started = time.perf_counter()
    try:
        response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        return {
            "run": run,
            "passed": False,
            "http_status": None,
            "latency_s": time.perf_counter() - started,
            "expected_speaker_ref": speaker_ref,
            "response": {"error": {"message": str(exc), "type": type(exc).__name__}},
        }
    elapsed = time.perf_counter() - started
    try:
        decoded = response.json()
    except ValueError:
        decoded = {
            "error": {
                "message": "upstream returned a non-JSON response",
                "body": response.text[:2000],
            }
        }
    body = decoded if isinstance(decoded, dict) else {"error": {"body": decoded}}
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    calls = message.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    arguments: dict[str, Any] = {}
    call = calls[0] if len(calls) == 1 and isinstance(calls[0], dict) else {}
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    if call:
        try:
            decoded_arguments = json.loads(function.get("arguments") or "{}")
            if isinstance(decoded_arguments, dict):
                arguments = decoded_arguments
        except (json.JSONDecodeError, TypeError):
            pass
    passed = (
        response.status_code == 200
        and len(calls) == 1
        and function.get("name") == "speaker_memory_remember_name"
        and arguments.get("speaker_ref") == speaker_ref
        and arguments.get("name") == "Марат"
        and choice.get("finish_reason") == "tool_calls"
        and message.get("content") in (None, "")
    )
    return {
        "run": run,
        "passed": passed,
        "http_status": response.status_code,
        "latency_s": elapsed,
        "expected_speaker_ref": speaker_ref,
        "response": body,
    }


def _latency_metrics(latencies: list[float]) -> dict[str, float | None]:
    warm_latencies = latencies[1:]
    return {
        "tool_latency_mean_s": statistics.mean(latencies),
        "tool_latency_p50_s": statistics.median(latencies),
        "tool_latency_warm_p50_s": (
            statistics.median(warm_latencies) if warm_latencies else None
        ),
        "tool_latency_max_s": max(latencies),
    }


def _ttft_probe(client: httpx.Client, url: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Отвечай по-русски одним коротким предложением."},
            {"role": "user", "content": "Назови столицу России."},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False, "thinking_mode": "disabled"},
    }
    started = time.perf_counter()
    first_content_at: float | None = None
    chunks: list[str] = []
    status = 0
    try:
        with client.stream("POST", url, json=payload) as response:
            status = response.status_code
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                choices = event.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else {}
                delta = choice.get("delta") if isinstance(choice, dict) else {}
                delta = delta if isinstance(delta, dict) else {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    chunks.append(content)
    except httpx.HTTPError as exc:
        return {
            "passed": False,
            "http_status": None,
            "ttft_s": None,
            "total_s": time.perf_counter() - started,
            "content": "",
            "error": {"message": str(exc), "type": type(exc).__name__},
        }
    finished = time.perf_counter()
    return {
        "passed": status == 200 and first_content_at is not None and bool(chunks),
        "http_status": status,
        "ttft_s": None if first_content_at is None else first_content_at - started,
        "total_s": finished - started,
        "content": "".join(chunks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", default="gemma-4-e2b")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    url = f"{args.endpoint.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=args.timeout) as client:
        tool_runs = [
            _tool_probe(client, url, args.model, run, args.temperature, args.max_tokens)
            for run in range(1, args.runs + 1)
        ]
        ttft = _ttft_probe(client, url, args.model)

    latencies = [run["latency_s"] for run in tool_runs]
    report = {
        "endpoint": args.endpoint,
        "model": args.model,
        "runs": args.runs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tool_successes": sum(bool(run["passed"]) for run in tool_runs),
        "tool_success_rate": sum(bool(run["passed"]) for run in tool_runs) / args.runs,
        **_latency_metrics(latencies),
        "tool_runs": tool_runs,
        "ru_ttft": ttft,
    }
    report["passed"] = report["tool_successes"] == args.runs and bool(ttft["passed"])
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
