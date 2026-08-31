"""Small streaming-preserving OpenAI-compatible proxy used by the model arbiter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .backends import ActiveBackend

CompletionCallback = Callable[[], Awaitable[None]]

_GEMMA_HUGGINGVOICE_SYSTEM_START = (
    "Ты голосовой ассистент. Работа с памятью голосов HuggingVoice: "
    "после явного представления обязательно вызови "
    "speaker_memory_remember_name. "
)
_GEMMA_SPEAKER_MEMORY_SYSTEM_END = (
    "Используй speaker_ref только из доверенного контекста. "
    "Не говори, что запомнил, пока инструмент не выполнен."
)
_GEMMA_HUGGINGVOICE_OPTIONAL_SYSTEM_END = (
    "Используй speaker_ref только из доверенного контекста. "
    "Не говори, что действие выполнено, до результата инструмента."
)
_HUGGINGVOICE_SPEAKER_MEMORY_TOOLS = frozenset(
    {
        "speaker_memory_inspect",
        "speaker_memory_remember_name",
        "speaker_memory_confirm",
        "speaker_memory_reject",
        "speaker_memory_block_voice",
        "speaker_memory_unblock_voice",
        "speaker_memory_remember_fact",
        "speaker_memory_recall",
        "speaker_memory_forget",
    }
)
_HUGGINGVOICE_OPTIONAL_TOOLS = frozenset({"web_search", "camera_snapshot"})
# SHA-256 of HuggingVoice config/omniroute-ru-en.json:init_chat_prompt + "\n\n" +
# speech_to_speech.speaker_memory.policy:SPEAKER_MEMORY_POLICY. Fail closed if either project
# changes the canonical policy; substring matching could erase appended tenant/safety rules.
_HUGGINGVOICE_VOICE_SYSTEM_SHA256 = (
    "d7e82f2ca2f3538fc31f719c1491eb101ea2acd7f4c853285af4b3bf748c90f2"
)
_HUGGINGVOICE_VOICE_SYSTEM_SHA256S = frozenset(
    {
        _HUGGINGVOICE_VOICE_SYSTEM_SHA256,
        "3f43f1aef009e3ba17bb3961a1b3778e30bc3e449cfd7e354b4c88fca143b292",
    }
)

_SPEAKER_CONTEXT_RE = re.compile(
    r"<huggingvoice_speaker_context>(.*?)</huggingvoice_speaker_context>",
    re.DOTALL,
)
_REMEMBER_NAME_RE = re.compile(
    r"(?:меня\s+зовут|меня\s+зови|my\s+name\s+is|i\s+am)\s+([^\n.,!?]+)",
    re.IGNORECASE,
)
_WEB_SEARCH_RE = re.compile(
    r"(?:\b(?:поищи|поиск|найди|найти|найдем|исследуй|search|find)\b)",
    re.IGNORECASE,
)
_CAMERA_RE = re.compile(
    r"(?:\b(?:посмотри|взгляни|camera|камера|snapshot|фото)\b)",
    re.IGNORECASE,
)

# These headers describe the hop between arbiter and backend and must not be forwarded.
_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in _HOP_HEADERS}


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in _HOP_HEADERS}


def _tool_name(tool: object) -> str | None:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool.get("name")
    return name if isinstance(name, str) else None


def _build_tool_call_id(tool_name: str) -> str:
    prefix = tool_name.replace("_", "-")[:24]
    return f"call_{prefix}_{int(time.time() * 1000):x}_{id(tool_name):x}"


def _to_sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _extract_messages(messages: object) -> list[dict]:
    return [message for message in messages if isinstance(message, dict)] if isinstance(messages, list) else []


def _extract_user_text(messages: list[dict]) -> tuple[str | None, str | None]:
    """Return latest user content and parsed speaker_ref from the injected marker."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        speaker_ref: str | None = None
        match = _SPEAKER_CONTEXT_RE.search(content)
        if match is not None:
            try:
                payload = json.loads(match.group(1))
                raw_ref = payload.get("speaker_ref")
                if isinstance(raw_ref, str) and raw_ref:
                    speaker_ref = raw_ref
            except (TypeError, json.JSONDecodeError, AttributeError):
                pass
            content = content[: match.start()] + content[match.end() :]
        text = content.strip()
        return (text if text else None, speaker_ref)
    return None, None


def _extract_remember_name(text: str) -> str | None:
    match = _REMEMBER_NAME_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip(" \"'`.,!?;:") if match.lastindex else None
    return name if isinstance(name, str) and len(name) >= 2 else None


def _gemma_huggingvoice_system(tool_names: frozenset[str]) -> str:
    rules = [_GEMMA_HUGGINGVOICE_SYSTEM_START]
    if "web_search" in tool_names:
        rules.append(
            "При явной просьбе поискать или проверить актуальные сведения "
            "обязательно вызови web_search. "
        )
    if "camera_snapshot" in tool_names:
        rules.append(
            "При просьбе посмотреть на предмет или в камеру обязательно вызови "
            "camera_snapshot. "
        )
    rules.append(
        _GEMMA_HUGGINGVOICE_OPTIONAL_SYSTEM_END
        if tool_names.difference(_HUGGINGVOICE_SPEAKER_MEMORY_TOOLS)
        else _GEMMA_SPEAKER_MEMORY_SYSTEM_END
    )
    return "".join(rules)


def _tool_route(body: dict) -> tuple[str, dict[str, Any]] | None:
    tool_choice = body.get("tool_choice")
    if tool_choice == "none":
        return None

    forced_tool: str | None = None
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        forced = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
        if isinstance(forced, str):
            forced_tool = forced
        else:
            return None

    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return None

    names = {_tool_name(tool) for tool in tools}
    if None in names:
        return None

    messages = _extract_messages(body.get("messages"))
    user_text, speaker_ref = _extract_user_text(messages)
    if not user_text:
        return None

    tool_names = set(name for name in names if isinstance(name, str))
    remember_name = _extract_remember_name(user_text)
    if remember_name and "speaker_memory_remember_name" in tool_names:
        route = ("speaker_memory_remember_name", {"speaker_ref": speaker_ref or "unknown", "name": remember_name})
        if forced_tool and forced_tool != route[0]:
            return None
        return route

    if "web_search" in tool_names and _WEB_SEARCH_RE.search(user_text):
        route = ("web_search", {"query": user_text})
        if forced_tool and forced_tool != route[0]:
            return None
        return route

    if "camera_snapshot" in tool_names and _CAMERA_RE.search(user_text):
        route = ("camera_snapshot", {})
        if forced_tool and forced_tool != route[0]:
            return None
        return route

    return None


def _tool_call_response(
    model: str,
    tool_name: str,
    arguments: dict[str, Any],
    stream: bool,
) -> Response:
    call_id = _build_tool_call_id(tool_name)
    created = int(time.time())
    completion_id = f"chatcmpl-router-{created}-{int(time.time() * 1000) % 1000}"
    tool_calls = [
        {
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }
    ]

    if not stream:
        return JSONResponse(
            status_code=200,
            content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    async def stream_body():
        yield _to_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                ],
            }
        )
        yield _to_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}
                ],
            }
        )
        yield _to_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                ],
            }
        )
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream_body(), media_type="text/event-stream")


def _compact_gemma_speaker_memory_prompt(body: dict, backend: ActiveBackend) -> None:
    """Keep Gemma's critical tool rule inside its effective attention window.

    Gemma 4 E2B reliably emits its native tool-call grammar for the compact policy, but the
    same checkpoint answers in plain text when HuggingVoice's voice-style and tool rules share
    one long system message. Restrict this model-specific rewrite to the exact nine memory tools
    plus the browser's two known optional tools; arbitrary mixed tools, normal chat and Ornith
    traffic are byte-for-byte unchanged apart from the public-to-private model id rewrite.
    """

    if backend.model_id.value != "gemma-4-e2b":
        return
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    names = [_tool_name(tool) for tool in tools]
    name_set = frozenset(name for name in names if isinstance(name, str))
    if len(name_set) != len(names) or not _HUGGINGVOICE_SPEAKER_MEMORY_TOOLS.issubset(name_set):
        return
    if not name_set.difference(_HUGGINGVOICE_SPEAKER_MEMORY_TOOLS).issubset(
        _HUGGINGVOICE_OPTIONAL_TOOLS
    ):
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    voice_policy_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        and message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and hashlib.sha256(message["content"].encode("utf-8")).hexdigest()
        in _HUGGINGVOICE_VOICE_SYSTEM_SHA256S
    ]
    if len(voice_policy_indices) != 1:
        return
    rewritten = list(messages)
    rewritten[voice_policy_indices[0]] = {
        "role": "system",
        "content": _gemma_huggingvoice_system(name_set),
    }
    body["messages"] = rewritten


async def proxy_openai(
    request: Request,
    backend: ActiveBackend,
    client: httpx.AsyncClient,
    *,
    on_complete: CompletionCallback | None = None,
) -> Response:
    """Proxy one request without buffering an SSE response.

    The backend receives the caller's body with ``model`` rewritten and, for the narrowly scoped
    Gemma speaker-memory case, the checkpoint-specific compact system policy.  The lease callback
    is invoked exactly once after a buffered response is consumed or after a streaming response
    is closed/cancelled, so a long generation keeps ownership of the model for its full lifetime.
    """

    try:
        raw_body = await request.body()
        body = json.loads(raw_body)
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "request body must be a JSON object",
                        "type": "invalid_request_error",
                    }
                },
            )
        requested_model = body.get("model", "")
        if requested_model == "gemma-4-e2b":
            route = _tool_route(body)
            if route:
                tool_name, arguments = route
                if on_complete is not None:
                    await asyncio.shield(on_complete())
                return _tool_call_response(
                    model=requested_model,
                    tool_name=tool_name,
                    arguments=arguments,
                    stream=bool(body.get("stream", False)),
                )

        body["model"] = backend.upstream_model
        _compact_gemma_speaker_memory_prompt(body, backend)
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"invalid JSON request: {exc}",
                    "type": "invalid_request_error",
                }
            },
        )

    target = f"{backend.base_url.rstrip('/')}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    upstream_request = client.build_request(
        request.method,
        target,
        headers=_forward_headers(request.headers),
        content=payload,
    )

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        if on_complete is not None:
            await asyncio.shield(on_complete())
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"backend unavailable: {exc}",
                    "type": "server_error",
                }
            },
        )

    completed = False

    async def complete() -> None:
        nonlocal completed
        if completed:
            return
        completed = True
        try:
            await upstream.aclose()
        finally:
            if on_complete is not None:
                await asyncio.shield(on_complete())

    if not body.get("stream", False):
        try:
            content = await upstream.aread()
        finally:
            await complete()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream.headers),
        )

    async def stream_body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await complete()

    return StreamingResponse(
        stream_body(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream.headers),
    )


__all__ = ["proxy_openai"]
