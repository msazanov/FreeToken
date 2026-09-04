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

    async def ornith_start(self, unit: str) -> None:
        self.events.append(("ornith_start", unit))

    async def ornith_stop(self, unit: str) -> None:
        self.events.append(("ornith_stop", unit))

    async def cpu_start(self, unit: str) -> None:
        self.events.append(("cpu_start", unit))

    async def cpu_stop(self, unit: str) -> None:
        self.events.append(("cpu_stop", unit))

    async def lfm_start(self, unit: str) -> None:
        self.events.append(("lfm_start", unit))

    async def lfm_stop(self, unit: str) -> None:
        self.events.append(("lfm_stop", unit))


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://arbiter.test")


def _ready_handler(
    requests: list[tuple[str, str, dict | None]],
    *,
    cpu_ok: bool = True,
    ornith_ok: bool = True,
    gemma_gpu_ok: bool = True,
    lfm_ok: bool = False,
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
            if "19197" in url and not lfm_ok:
                return httpx.Response(503, json={"status": "error"})
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/models"):
            model = "backend-model"
            if "19197" in url and lfm_ok:
                model = "LFM2.5-2.6B"
            return httpx.Response(200, json={"data": [{"id": model}]})
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
        assert [name for name, _ in lifecycle.events] == [
            "ornith_stop",
            "lfm_stop",
            "gpu_start",
            "gpu_stop",
            "cpu_start",
        ]
        assert any(path.endswith("/health") and "19195" in path for _, path, _ in requests)

    asyncio.run(scenario())


def test_lfm_start_stops_ornith_and_commits_only_after_readiness():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        lifecycle = RecordingLifecycle()
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            lfm_expected_model="LFM2.5-2.6B",
            readiness_timeout_s=1,
            poll_interval_s=0,
        )
        async with _client(
            _ready_handler(requests, lfm_ok=True, ornith_ok=True, gemma_gpu_ok=False)
        ) as client:
            controller = BackendController(config, client, lifecycle=lifecycle)
            backend = await controller.prepare(ModelId.LFM)

        assert backend == ActiveBackend(
            ModelId.LFM, config.lfm_url, config.lfm_expected_model, "lfm-gpu"
        )
        assert ("ornith_stop", config.ornith_unit) in lifecycle.events
        assert ("lfm_start", config.lfm_unit) in lifecycle.events

    asyncio.run(scenario())


def test_gemma_gpu_transition_stops_ornith_before_starting_gpu_model():
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
            backend = await controller.prepare(ModelId.GEMMA)

        assert backend.runtime == "gemma-gpu"
        assert [name for name, _ in controller.lifecycle.events][:3] == [
            "ornith_stop",
            "lfm_stop",
            "gpu_start",
        ]
        _, (_model_path, _port, args) = controller.lifecycle.events[2]
        assert args[args.index("--max-seq-len-override") + 1] == "8192"
        assert args[args.index("--num-tokens") + 1] == "8192"
        assert args[args.index("--kv-reserve-tokens") + 1] == "8192"

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


def test_reconcile_adopts_exact_running_gemma_gpu_when_ornith_is_stopped():
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
                ornith_ok=False,
                daemon_running=True,
                daemon_model="/models/gemma.gguf",
                daemon_port=config.gemma_gpu_port,
            )
        ) as client:
            controller = BackendController(config, client, lifecycle=RecordingLifecycle())
            await controller.reconcile()

        assert controller.current == ActiveBackend(
            ModelId.GEMMA, config.gemma_gpu_url, "backend-model", "gemma-gpu"
        )

    asyncio.run(scenario())


def test_reconcile_waits_for_active_ornith_to_finish_loading():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        ornith_health_checks = 0
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            ornith_active_slots=256,
            ornith_parked_slots=128,
            readiness_timeout_s=0.05,
            poll_interval_s=0.001,
        )
        base_handler = _ready_handler(
            requests,
            cpu_ok=False,
            gemma_gpu_ok=False,
            moe_size=config.ornith_active_slots,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal ornith_health_checks
            if "19191" in str(request.url) and request.url.path.endswith("/health"):
                ornith_health_checks += 1
                if ornith_health_checks == 1:
                    return httpx.Response(503, json={"status": "loading"})
            return base_handler(request)

        lifecycle = RecordingLifecycle()

        async def unit_is_active(unit: str) -> bool:
            return unit == config.ornith_unit

        lifecycle.unit_is_active = unit_is_active

        async with _client(handler) as client:
            controller = BackendController(config, client, lifecycle=lifecycle)
            await controller.reconcile()

        assert ornith_health_checks >= 2
        assert controller.current == ActiveBackend(
            ModelId.ORNITH, config.ornith_url, "backend-model", "ornith-gpu"
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


def test_ornith_cache_geometry_mismatch_is_rejected():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
            ornith_active_slots=2311,
        )

        async with _client(
            _ready_handler(requests, moe_size=128, rebuild_updates_geometry=False)
        ) as client:
            controller = BackendController(config, client, lifecycle=RecordingLifecycle())
            with pytest.raises(BackendError, match="MoE cache rebuild verification"):
                await controller.prepare(ModelId.ORNITH)

    asyncio.run(scenario())


def test_ornith_is_started_again_when_returning_from_gemma_gpu():
    async def scenario():
        requests: list[tuple[str, str, dict | None]] = []
        ornith_ready = True
        lifecycle = RecordingLifecycle()
        config = BackendConfig(
            ornith_expected_model="backend-model",
            gemma_expected_model="backend-model",
        )
        base_handler = _ready_handler(requests)

        def handler(request: httpx.Request) -> httpx.Response:
            if (
                "19191" in str(request.url)
                and request.url.path.endswith("/health")
                and not ornith_ready
            ):
                return httpx.Response(503, json={"status": "error"})
            return base_handler(request)

        async def ornith_stop(unit: str) -> None:
            nonlocal ornith_ready
            ornith_ready = False
            lifecycle.events.append(("ornith_stop", unit))

        async def ornith_start(unit: str) -> None:
            nonlocal ornith_ready
            ornith_ready = True
            lifecycle.events.append(("ornith_start", unit))

        lifecycle.ornith_stop = ornith_stop
        lifecycle.ornith_start = ornith_start

        async with _client(handler) as client:
            controller = BackendController(config, client, lifecycle=lifecycle)
            await controller.prepare(ModelId.GEMMA)
            await controller.prepare(ModelId.ORNITH)

        assert ("ornith_start", config.ornith_unit) in lifecycle.events
        assert ("gpu_stop", None) in lifecycle.events
        assert controller.current is not None
        assert controller.current.model_id is ModelId.ORNITH

    asyncio.run(scenario())
