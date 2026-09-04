"""Public one-port model arbiter for Ornith, Gemma and LFM2.5."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .backends import BackendController
from .model import ModelId
from .proxy import proxy_openai
from .scheduler import LeaseScheduler


@dataclass(frozen=True, slots=True)
class ArbiterConfig:
    queue_timeout_s: float = 1800.0
    max_queue_depth: int | None = 16
    request_timeout_s: float = 1800.0


def _error(status_code: int, message: str, *, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "code": code,
            }
        },
    )


def _model_from_body(body: Any) -> ModelId | None:
    if not isinstance(body, dict):
        return None
    value = body.get("model")
    try:
        return ModelId(value)
    except (TypeError, ValueError):
        return None


def build_arbiter_app(
    config: ArbiterConfig,
    controller: BackendController,
    client: httpx.AsyncClient,
    *,
    scheduler: LeaseScheduler | None = None,
) -> FastAPI:
    scheduler = scheduler or LeaseScheduler(
        tie_break=ModelId.ORNITH,
        max_queue_depth=config.max_queue_depth,
    )
    app = FastAPI(title="FreeToken model arbiter", version="1")
    app.state.config = config
    app.state.controller = controller
    app.state.scheduler = scheduler
    app.state.http_client = client
    app.state.started_at = time.time()
    app.state.counters = {"requests": 0, "completed": 0, "errors": 0}
    app.state.reconcile_lock = asyncio.Lock()
    app.state.reconciled = False

    async def ensure_reconciled() -> None:
        if app.state.reconciled:
            return
        async with app.state.reconcile_lock:
            if app.state.reconciled:
                return
            reconcile = getattr(controller, "reconcile", None)
            if reconcile is not None:
                await reconcile()
            app.state.reconciled = True

    @app.get("/health")
    async def health() -> dict[str, Any]:
        active = controller.current
        return {
            "status": "ok",
            "service": "freetoken-arbiter",
            "active_model": None if active is None else active.model_id.value,
            "scheduler": _jsonable_snapshot(scheduler.snapshot()),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        active = controller.current
        return {
            "service": "freetoken-arbiter",
            "uptime_s": max(0.0, time.time() - app.state.started_at),
            "counters": dict(app.state.counters),
            "active_backend": None
            if active is None
            else {
                "model": active.model_id.value,
                "runtime": active.runtime,
                "base_url": active.base_url,
            },
            "scheduler": _jsonable_snapshot(scheduler.snapshot()),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        created = int(app.state.started_at)
        return {
            "object": "list",
            "data": [
                {"id": model_id.value, "object": "model", "created": created, "owned_by": "local"}
                for model_id in (ModelId.ORNITH, ModelId.GEMMA, ModelId.LFM)
            ],
        }

    async def completion(request: Request) -> Any:
        try:
            body = await request.json()
        except (ValueError, TypeError) as exc:
            return _error(400, f"invalid JSON request: {exc}", code="invalid_json")
        model_id = _model_from_body(body)
        if model_id is None:
            return _error(
                404,
                "model must be one of ornith-35b, gemma-4-e2b or LFM2.5-2.6B",
                code="model_not_found",
            )

        app.state.counters["requests"] += 1
        try:
            await ensure_reconciled()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - do not route traffic on ambiguous state
            app.state.counters["errors"] += 1
            return _error(503, f"backend state is not reconciled: {exc}", code="state_ambiguous")

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            lease = await scheduler.acquire(
                model_id,
                request_id,
                timeout=config.queue_timeout_s,
            )
        except TimeoutError:
            app.state.counters["errors"] += 1
            return _error(504, "timed out waiting for the model lease", code="queue_timeout")
        except asyncio.QueueFull:
            app.state.counters["errors"] += 1
            return _error(429, "model queue is full", code="queue_full")

        try:
            backend = await controller.prepare(model_id)
        except asyncio.CancelledError:
            await asyncio.shield(lease.release())
            raise
        except Exception as exc:  # noqa: BLE001 - translate lifecycle errors at API boundary
            await asyncio.shield(lease.release())
            app.state.counters["errors"] += 1
            return _error(503, f"model backend is not ready: {exc}", code="backend_not_ready")

        released = False

        async def release_all() -> None:
            nonlocal released
            if released:
                return
            try:
                await controller.release(model_id)
            finally:
                await lease.release()
                released = True
            app.state.counters["completed"] += 1

        try:
            return await proxy_openai(
                request,
                backend,
                client,
                on_complete=release_all,
            )
        except asyncio.CancelledError:
            await asyncio.shield(release_all())
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a lease on proxy errors
            await asyncio.shield(release_all())
            app.state.counters["errors"] += 1
            return _error(502, f"backend proxy failed: {exc}", code="proxy_failed")

    app.post("/v1/chat/completions")(completion)
    app.post("/v1/responses")(completion)
    return app


def _jsonable_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    value = dict(snapshot)
    state = value.get("state")
    if hasattr(state, "value"):
        value["state"] = state.value
    owner = value.get("owner")
    if hasattr(owner, "value"):
        value["owner"] = owner.value
    queue_depths = value.get("queue_depths")
    if isinstance(queue_depths, dict):
        value["queue_depths"] = {
            getattr(key, "value", key): depth for key, depth in queue_depths.items()
        }
    return value


__all__ = ["ArbiterConfig", "build_arbiter_app"]
