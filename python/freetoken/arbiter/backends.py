"""Backend lifecycle and readiness checks for the torch-free model arbiter.

The arbiter deliberately treats FreeToken and llama.cpp as remote processes.  It never imports
torch and never tries to migrate a partially generated request between runtimes.  A controller
change is made only while the scheduler owns the model lease.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .model import ModelId


class BackendError(RuntimeError):
    """A backend could not be made safe to receive a request."""


@dataclass(frozen=True, slots=True)
class ActiveBackend:
    model_id: ModelId
    base_url: str
    upstream_model: str
    runtime: str


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Paths, ports and cache geometry for the two-model RTX 2070 topology."""

    ornith_url: str = "http://127.0.0.1:19191"
    ornith_expected_model: str = "Ornith 1.5 35b"
    gemma_gpu_url: str = "http://127.0.0.1:19192"
    gemma_cpu_url: str = "http://127.0.0.1:19193"
    gemma_expected_model: str = "gemma-4-e2b"
    daemon_url: str = "http://127.0.0.1:1900"
    daemon_token: str | None = None
    gemma_model_path: str = (
        "/home/random/dev/huggingvoice-llm-bench/models/gemma4-e2b/"
        "gemma-4-E2B_q4_0-it.gguf"
    )
    gemma_gpu_port: int = 19192
    gemma_gpu_args: tuple[str, ...] = (
        "--served-model-name",
        "gemma-4-e2b",
        "--host",
        "127.0.0.1",
        "--dtype",
        "bfloat16",
        "--moe-backend",
        "offload",
        "--max-running-requests",
        "1",
        "--max-seq-len-override",
        "4096",
        "--num-tokens",
        "4096",
        "--kv-reserve-tokens",
        "4096",
        "--cuda-graph-max-bs",
        "1",
        "--attention-backend",
        "triton",
        "--reasoning-parser",
        "none",
    )
    gemma_cpu_unit: str = "llama-gemma-cpu.service"
    ornith_active_slots: int = 2311
    ornith_parked_slots: int = 256
    ornith_kv_tokens: int = 65536
    rebuild_timeout_s: float = 300.0
    readiness_timeout_s: float = 300.0
    poll_interval_s: float = 0.25
    request_timeout_s: float = 1800.0


class BackendLifecycle(Protocol):
    async def daemon_start(self, model_path: str, port: int, args: list[str]) -> None: ...

    async def daemon_stop(self) -> None: ...

    async def cpu_start(self, unit: str) -> None: ...

    async def cpu_stop(self, unit: str) -> None: ...


class SystemBackendLifecycle:
    """Concrete lifecycle adapter used by the production arbiter."""

    def __init__(self, client: httpx.AsyncClient, config: BackendConfig) -> None:
        self._client = client
        self._config = config

    def _headers(self) -> dict[str, str]:
        if self._config.daemon_token is None:
            return {}
        return {"X-FT-Token": self._config.daemon_token}

    async def daemon_start(self, model_path: str, port: int, args: list[str]) -> None:
        try:
            response = await self._client.post(
                f"{self._config.daemon_url.rstrip('/')}/engine/start",
                headers=self._headers(),
                json={"model": model_path, "port": port, "args": args},
            )
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise BackendError(f"Gemma GPU start failed: {exc}") from exc

    async def daemon_stop(self) -> None:
        try:
            response = await self._client.post(
                f"{self._config.daemon_url.rstrip('/')}/engine/stop",
                headers=self._headers(),
                json={},
            )
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise BackendError(f"Gemma GPU stop failed: {exc}") from exc

    async def _systemctl(self, verb: str, unit: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            verb,
            unit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", "replace").strip()
            raise BackendError(f"systemctl --user {verb} {unit} failed: {detail}")

    async def cpu_start(self, unit: str) -> None:
        await self._systemctl("start", unit)

    async def cpu_stop(self, unit: str) -> None:
        await self._systemctl("stop", unit)


class BackendController:
    """Switch and validate private runtimes while holding the arbiter's execution lease."""

    def __init__(
        self,
        config: BackendConfig,
        client: httpx.AsyncClient,
        *,
        lifecycle: BackendLifecycle | None = None,
        sleep=asyncio.sleep,
        monotonic=time.monotonic,
    ) -> None:
        self.config = config
        self.client = client
        self.lifecycle = lifecycle or SystemBackendLifecycle(client, config)
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = asyncio.Lock()
        self._current: ActiveBackend | None = None

    @property
    def current(self) -> ActiveBackend | None:
        return self._current

    async def prepare(self, model_id: ModelId) -> ActiveBackend:
        """Return a ready backend, switching resources only when the requested model changes."""
        model_id = ModelId(model_id)
        async with self._lock:
            if self._current is not None and self._current.model_id is model_id:
                return self._current

            if model_id is ModelId.ORNITH:
                await self._stop_gemma_gpu_if_needed()
                await self._rebuild_ornith(self.config.ornith_active_slots)
                backend = ActiveBackend(
                    ModelId.ORNITH,
                    self.config.ornith_url,
                    self.config.ornith_expected_model,
                    "ornith-gpu",
                )
                await self._wait_ready(backend)
                self._current = backend
                return backend

            await self._park_ornith_if_ready()
            try:
                await self.lifecycle.daemon_start(
                    self.config.gemma_model_path,
                    self.config.gemma_gpu_port,
                    list(self.config.gemma_gpu_args),
                )
                gpu = ActiveBackend(
                    ModelId.GEMMA,
                    self.config.gemma_gpu_url,
                    self.config.gemma_expected_model,
                    "gemma-gpu",
                )
                await self._wait_ready(gpu)
                self._current = gpu
                return gpu
            except Exception as gpu_exc:  # noqa: BLE001 - CPU is the explicit pre-commit fallback
                await self._safe_daemon_stop()
                try:
                    await self.lifecycle.cpu_start(self.config.gemma_cpu_unit)
                    cpu = ActiveBackend(
                        ModelId.GEMMA,
                        self.config.gemma_cpu_url,
                        self.config.gemma_expected_model,
                        "gemma-cpu",
                    )
                    await self._wait_ready(cpu)
                except Exception as cpu_exc:  # noqa: BLE001 - preserve both causes
                    raise BackendError(
                        f"Gemma GPU and CPU readiness failed; gpu={gpu_exc}; cpu={cpu_exc}"
                    ) from cpu_exc
                self._current = cpu
                return cpu

    async def release(self, model_id: ModelId) -> None:
        """Release request-level resources without evicting warm model state."""
        # The scheduler decides when another model may acquire the lease.  Keeping this method
        # explicit gives the proxy a stable lifecycle hook and makes future idle policies safe.
        if self._current is not None and self._current.model_id is ModelId(model_id):
            return

    async def _stop_gemma_gpu_if_needed(self) -> None:
        if self._current is not None and self._current.runtime == "gemma-gpu":
            await self.lifecycle.daemon_stop()
            self._current = None

    async def _safe_daemon_stop(self) -> None:
        try:
            await self.lifecycle.daemon_stop()
        except Exception:
            # A failed start may have no child.  CPU fallback remains the safe next attempt; the
            # original GPU failure is retained in the eventual BackendError if CPU also fails.
            pass

    async def _park_ornith_if_ready(self) -> None:
        if not await self._is_ready(self.config.ornith_url, self.config.ornith_expected_model):
            return
        await self._rebuild_ornith(self.config.ornith_parked_slots)

    async def _rebuild_ornith(self, slots: int) -> None:
        payload = {
            "moe_cache_size": slots,
            "mode": "if_idle",
            "timeout": self.config.rebuild_timeout_s,
        }
        try:
            response = await self.client.post(
                f"{self.config.ornith_url.rstrip('/')}/v1/cache/rebuild", json=payload
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise BackendError(f"Ornith MoE cache rebuild to {slots} failed: {exc}") from exc
        if result.get("status") != "ok":
            raise BackendError(f"Ornith MoE cache rebuild to {slots} returned {result!r}")
        geometry = result.get("geometry") or {}
        kv_tokens = geometry.get("num_tokens", geometry.get("kv_tokens"))
        if kv_tokens is not None and int(kv_tokens) != self.config.ornith_kv_tokens:
            raise BackendError(
                f"Ornith KV invariant changed during MoE-only rebuild: "
                f"{kv_tokens} != {self.config.ornith_kv_tokens}"
            )

    async def _is_ready(self, base_url: str, expected_model: str) -> bool:
        try:
            health = await self.client.get(f"{base_url.rstrip('/')}/health")
            if health.status_code != 200 or health.json().get("status") != "ok":
                return False
            models = await self.client.get(f"{base_url.rstrip('/')}/v1/models")
            models.raise_for_status()
            data = models.json().get("data") or []
            return any(item.get("id") == expected_model for item in data)
        except (httpx.HTTPError, ValueError, OSError):
            return False

    async def _wait_ready(self, backend: ActiveBackend) -> None:
        deadline = self._monotonic() + self.config.readiness_timeout_s
        while self._monotonic() < deadline:
            if await self._is_ready(backend.base_url, backend.upstream_model):
                return
            await self._sleep(self.config.poll_interval_s)
        raise BackendError(
            f"{backend.runtime} readiness timeout at {backend.base_url}; "
            f"expected model {backend.upstream_model!r}"
        )


__all__ = [
    "ActiveBackend",
    "BackendConfig",
    "BackendController",
    "BackendError",
    "BackendLifecycle",
]
