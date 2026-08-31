"""Small streaming-preserving OpenAI-compatible proxy used by the model arbiter."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .backends import ActiveBackend

CompletionCallback = Callable[[], Awaitable[None]]

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


async def proxy_openai(
    request: Request,
    backend: ActiveBackend,
    client: httpx.AsyncClient,
    *,
    on_complete: CompletionCallback | None = None,
) -> Response:
    """Proxy one request without buffering an SSE response.

    The backend receives the caller's body with only ``model`` rewritten.  The lease callback is
    invoked exactly once after a buffered response is consumed or after a streaming response is
    closed/cancelled, so a long generation keeps ownership of the model for its full lifetime.
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
            await on_complete()
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
                await on_complete()

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
