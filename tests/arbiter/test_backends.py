from __future__ import annotations

import httpx
import pytest

from freetoken.arbiter.backends import (
    ActiveBackend,
    BackendConfig,
    BackendController,
    BackendError,
)
from freetoken.arbiter.model import ModelId


class RecordingLifecycle:
    def __init__(self, *, gpu_start_error: Exception | None = None) -> None:
        self.events: list[tuple[str, object]] = []
        self.gpu_start_error = gpu_start_error

    async def daemon_start(self, model_path: str, port: int, args: list[str]) -> None:
        self.events.append(("gpu_start", (model_path, port, args)))
        if self.gpu_start_error is not None:
            raise self.gpu_start_error

    async def daemon_stop(self) -> None:
        self.events.append(("gpu_stop", None))

    async def cpu_start(self, unit: str) -> None:
        self.events.append(("cpu_start", unit))

    async def cpu_stop(self, unit: str) -> None:
        self.events.append(("cpu_stop", unit))


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://arbiter.test")


def _ready_handler(requests: list[tuple[str, str, dict | None]], *, cpu_ok: bool = True):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        payload = None
        if request.content:
            payload = httpx.Response(200, content=request.content).json()
        requests.append((request.method, path, payload))
        if path.endswith("/cache/rebuild"):
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "geometry": {"num_tokens": 65536, "moe_cache_size": payload["moe_cache_size"]},
                },
            )
        if path.endswith("/health"):
            if "19193" in str(request.url) and not cpu_ok:
                return httpx.Response(503, json={"status": "error"})
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "backend-model"}]})
        return httpx.Response(404, json={"error": "not found"})

    return handler


@pytest.mark.asyncio
async def test_gemma_gpu_failure_selects_cpu_backend_before_readiness_commit():
    requests: list[tuple[str, str, dict | None]] = []
    lifecycle = RecordingLifecycle(gpu_start_error=RuntimeError("SM75 GPU admission failed"))
    config = BackendConfig(
        gemma_expected_model="backend-model",
        gemma_model_path="/models/gemma.gguf",
        gemma_gpu_args=("--ctx-size", "4096"),
    )

    async with _client(_ready_handler(requests)) as client:
        controller = BackendController(config, client, lifecycle=lifecycle)
        backend = await controller.prepare(ModelId.GEMMA)

    assert backend == ActiveBackend(ModelId.GEMMA, config.gemma_cpu_url, "backend-model", "gemma-cpu")
    assert [name for name, _ in lifecycle.events] == ["gpu_start", "gpu_stop", "cpu_start"]
    assert any(path.endswith("/health") and "19193" in path for _, path, _ in requests)


@pytest.mark.asyncio
async def test_ornith_park_is_moe_only_and_preserves_kv_geometry():
    requests: list[tuple[str, str, dict | None]] = []
    config = BackendConfig(
        ornith_expected_model="backend-model",
        ornith_active_slots=2311,
        ornith_parked_slots=256,
        ornith_kv_tokens=65536,
    )

    async with _client(_ready_handler(requests)) as client:
        controller = BackendController(config, client, lifecycle=RecordingLifecycle())
        await controller.prepare(ModelId.GEMMA)

    rebuilds = [(method, path, payload) for method, path, payload in requests if path.endswith("/cache/rebuild")]
    assert rebuilds
    payload = rebuilds[0][2]
    assert payload == {"moe_cache_size": 256, "mode": "if_idle", "timeout": config.rebuild_timeout_s}
    assert "num_pages" not in payload
    assert "num_mamba_slots" not in payload


@pytest.mark.asyncio
async def test_failed_cpu_readiness_is_reported_without_faking_backend():
    requests: list[tuple[str, str, dict | None]] = []
    lifecycle = RecordingLifecycle(gpu_start_error=RuntimeError("gpu unavailable"))
    config = BackendConfig(gemma_expected_model="backend-model")

    async with _client(_ready_handler(requests, cpu_ok=False)) as client:
        controller = BackendController(config, client, lifecycle=lifecycle)
        with pytest.raises(BackendError, match="gemma-cpu"):
            await controller.prepare(ModelId.GEMMA)
