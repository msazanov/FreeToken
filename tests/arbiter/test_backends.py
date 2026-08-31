from __future__ import annotations

import asyncio

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


def _ready_handler(
    requests: list[tuple[str, str, dict | None]],
    *,
    cpu_ok: bool = True,
    ornith_ok: bool = True,
    gemma_gpu_ok: bool = True,
    moe_size: int = 256,
    daemon_running: bool = False,
    daemon_model: str | None = None,
    daemon_port: int = 19193,
    rebuild_updates_geometry: bool = True,
):
    current_moe_size = moe_size

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current_moe_size
        path = request.url.path
        url = str(request.url)
        payload = None
        if request.content:
            payload = httpx.Response(200, content=request.content).json()
        requests.append((request.method, url, payload))
        if path.endswith("/cache/rebuild"):
            if rebuild_updates_geometry:
                current_moe_size = payload["moe_cache_size"]
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "geometry": {"num_tokens": 65536, "moe_cache_size": payload["moe_cache_size"]},
                },
            )
        if path.endswith("/cache/status"):
            return httpx.Response(
                200,
                json={
                    "state": "serving",
                    "geometry": {
                        "num_pages": 65536,
                        "page_size": 1,
                        "moe_cache_size": current_moe_size,
                        "num_mamba_slots": 8,
                        "num_swa_pages": 0,
                    },
                },
            )
        if path.endswith("/engine/status"):
            return httpx.Response(
                200,
                json={
                    "running": daemon_running,
                    "starting": False,
                    "stopping": False,
                    "model": daemon_model,
                    "port": daemon_port if daemon_running else None,
                },
            )
        if path.endswith("/health"):
            if "19191" in url and not ornith_ok:
                return httpx.Response(503, json={"status": "error"})
            if "19193" in url and not gemma_gpu_ok:
                return httpx.Response(503, json={"status": "error"})
            if "19195" in url and not cpu_ok:
                return httpx.Response(503, json={"status": "error"})
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "backend-model"}]})
        return httpx.Response(404, json={"error": "not found"})

    return handler


def test_gemma_gpu_failure_selects_cpu_backend_before_readiness_commit():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        lifecycle = RecordingLifecycle(gpu_start_error=RuntimeError("SM75 GPU admission failed"))
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            gemma_model_path="/models/gemma.gguf",
            gemma_gpu_args=("--ctx-size", "4096"),
        )

        async with _client(_ready_handler(requests)) as client:
            controller = BackendController(config, client, lifecycle=lifecycle)
            backend = await controller.prepare(ModelId.GEMMA)

        assert backend == ActiveBackend(ModelId.GEMMA, config.gemma_cpu_url, "backend-model", "gemma-cpu")
        assert [name for name, _ in lifecycle.events] == ["gpu_start", "gpu_stop", "cpu_start"]
        assert any(path.endswith("/health") and "19195" in path for _, path, _ in requests)

    asyncio.run(scenario())


def test_ornith_park_is_moe_only_and_preserves_kv_geometry():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
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

    asyncio.run(scenario())


def test_failed_cpu_readiness_is_reported_without_faking_backend():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        lifecycle = RecordingLifecycle(gpu_start_error=RuntimeError("gpu unavailable"))
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            readiness_timeout_s=0.01,
            poll_interval_s=0.001,
        )

        async with _client(_ready_handler(requests, cpu_ok=False)) as client:
            controller = BackendController(config, client, lifecycle=lifecycle)
            with pytest.raises(BackendError, match="gemma-cpu"):
                await controller.prepare(ModelId.GEMMA)

    asyncio.run(scenario())


def test_switching_from_cpu_gemma_stops_cpu_fallback_before_ornith():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        lifecycle = RecordingLifecycle()
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
        )

        async with _client(_ready_handler(requests)) as client:
            controller = BackendController(config, client, lifecycle=lifecycle)
            await controller.prepare(ModelId.GEMMA)
            # The test lifecycle's CPU path is the active Gemma backend.  Model it explicitly to
            # exercise the resource-release branch that production reaches after GPU fallback.
            controller._current = ActiveBackend(
                ModelId.GEMMA, config.gemma_cpu_url, "backend-model", "gemma-cpu"
            )
            await controller.prepare(ModelId.ORNITH)

        assert ("cpu_stop", config.gemma_cpu_unit) in lifecycle.events
        assert controller.current is not None
        assert controller.current.model_id is ModelId.ORNITH

    asyncio.run(scenario())


def test_reconcile_adopts_exact_running_gemma_gpu_with_parked_ornith():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            gemma_model_path="/models/gemma.gguf",
        )

        async with _client(
            _ready_handler(
                requests,
                cpu_ok=False,
                daemon_running=True,
                daemon_model="/models/gemma.gguf",
                daemon_port=config.gemma_gpu_port,
                moe_size=config.ornith_parked_slots,
            )
        ) as client:
            controller = BackendController(config, client, lifecycle=RecordingLifecycle())
            await controller.reconcile()

        assert controller.current == ActiveBackend(
            ModelId.GEMMA, config.gemma_gpu_url, "backend-model", "gemma-gpu"
        )

    asyncio.run(scenario())


def test_reconcile_rejects_unknown_running_daemon():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        config = BackendConfig(ornith_expected_model="backend-model")

        async with _client(
            _ready_handler(
                requests,
                cpu_ok=False,
                daemon_running=True,
                daemon_model="/models/another.gguf",
                daemon_port=config.gemma_gpu_port,
            )
        ) as client:
            controller = BackendController(config, client, lifecycle=RecordingLifecycle())
            with pytest.raises(BackendError, match="unknown Gemma daemon"):
                await controller.reconcile()

    asyncio.run(scenario())


def test_reconcile_rejects_gpu_and_active_ornith_overlap():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            gemma_model_path="/models/gemma.gguf",
            ornith_active_slots=2311,
        )

        async with _client(
            _ready_handler(
                requests,
                cpu_ok=False,
                daemon_running=True,
                daemon_model="/models/gemma.gguf",
                daemon_port=config.gemma_gpu_port,
                moe_size=config.ornith_active_slots,
            )
        ) as client:
            controller = BackendController(config, client, lifecycle=RecordingLifecycle())
            with pytest.raises(BackendError, match="overlap"):
                await controller.reconcile()

    asyncio.run(scenario())


def test_reconcile_keeps_ornith_active_when_only_cpu_gemma_is_ready():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
        )

        async with _client(
            _ready_handler(
                requests,
                cpu_ok=True,
                gemma_gpu_ok=False,
                moe_size=config.ornith_active_slots,
            )
        ) as client:
            controller = BackendController(config, client, lifecycle=RecordingLifecycle())
            await controller.reconcile()

        assert controller.current == ActiveBackend(
            ModelId.GEMMA, config.gemma_cpu_url, "backend-model", "gemma-cpu"
        )

    asyncio.run(scenario())


def test_cache_geometry_mismatch_is_rejected():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            ornith_parked_slots=256,
        )

        async with _client(
            _ready_handler(requests, moe_size=128, rebuild_updates_geometry=False)
        ) as client:
            controller = BackendController(config, client, lifecycle=RecordingLifecycle())
            with pytest.raises(BackendError, match="MoE cache rebuild verification"):
                await controller.prepare(ModelId.GEMMA)

    asyncio.run(scenario())
