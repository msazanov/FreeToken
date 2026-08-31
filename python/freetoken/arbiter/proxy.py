"""Small streaming-preserving OpenAI-compatible proxy used by the model arbiter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .backends import ActiveBackend

CompletionCallback = Callable[[], Awaitable[None]]

_GEMMA_SPEAKER_MEMORY_SYSTEM = (
    "Ты голосовой ассистент. Работа с памятью голосов HuggingVoice: "
    "после явного представления обязательно вызови "
    "speaker_memory_remember_name. Используй speaker_ref только из доверенного "
    "контекста. Не говори, что запомнил, пока инструмент не выполнен."
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


def _compact_gemma_speaker_memory_prompt(body: dict, backend: ActiveBackend) -> None:
    """Keep Gemma's critical tool rule inside its effective attention window.

    Gemma 4 E2B reliably emits its native tool-call grammar for the compact policy, but the
    same checkpoint answers in plain text when HuggingVoice's voice-style and all memory rules
    share one long system message.  Restrict this model-specific rewrite to the private Gemma
    backend and an exclusively ``speaker_memory_*`` tool set; normal chat and Ornith traffic are
    byte-for-byte unchanged apart from the public-to-private model id rewrite.
    """

    if backend.model_id.value != "gemma-4-e2b":
        return
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    names = [_tool_name(tool) for tool in tools]
    if (
        len(names) != len(_HUGGINGVOICE_SPEAKER_MEMORY_TOOLS)
        or frozenset(names) != _HUGGINGVOICE_SPEAKER_MEMORY_TOOLS
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
        and "Работа с памятью голосов HuggingVoice:" in message["content"]
        and "Silero v5.5 RU" in message["content"]
        and "speaker_memory_block_voice" in message["content"]
    ]
    if len(voice_policy_indices) != 1:
        return
    rewritten = list(messages)
    rewritten[voice_policy_indices[0]] = {
        "role": "system",
        "content": _GEMMA_SPEAKER_MEMORY_SYSTEM,
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
