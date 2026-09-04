"""Backend lifecycle and readiness checks for the torch-free model arbiter.

The arbiter deliberately treats FreeToken and llama.cpp as remote processes.  It never imports
torch and never tries to migrate a partially generated request between runtimes.  A controller
change is made only while the scheduler owns the model lease.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .model import ModelId

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


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
    """Paths, ports and cache geometry for the three-model RTX 2070 topology."""

    ornith_url: str = "http://127.0.0.1:19191"
    ornith_expected_model: str = "Ornith 1.5 35b"
    ornith_unit: str = "freetoken-ornith.service"
    # FreeToken binds an internal ZMQ endpoint at serve_port + 1, so 19192 is already owned by
    # the Ornith server on 19191.  Keep the second FreeToken server two ports away.
    gemma_gpu_url: str = "http://127.0.0.1:19193"
    gemma_cpu_url: str = "http://127.0.0.1:19195"
    gemma_expected_model: str = "gemma-4-e2b"
    daemon_url: str = "http://127.0.0.1:1900"
    daemon_token: str | None = None
    gemma_model_path: str = (
        "/home/random/dev/huggingvoice-llm-bench/models/gemma4-e2b/"
        "gemma-4-E2B_q4_0-it.gguf"
    )
    gemma_gpu_port: int = 19193
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
        "8192",
        "--num-tokens",
        "8192",
        "--kv-reserve-tokens",
        "8192",
        "--cuda-graph-max-bs",
        "1",
        "--attention-backend",
        "triton",
        "--reasoning-parser",
        "gemma4",
    )
    gemma_cpu_unit: str = "llama-gemma-cpu.service"
    lfm_url: str = "http://127.0.0.1:19197"
    lfm_expected_model: str = "LFM2.5-2.6B"
    lfm_unit: str = "llama-lfm25.service"
    ornith_active_slots: int = 2311
    ornith_parked_slots: int = 256
    ornith_kv_tokens: int = 65536
    ornith_mamba_slots: int | None = 8
    ornith_swa_tokens: int | None = 0
    rebuild_timeout_s: float = 300.0
    readiness_timeout_s: float = 300.0
    poll_interval_s: float = 0.25
    lifecycle_timeout_s: float = 30.0
    probe_timeout_s: float = 2.0
    request_timeout_s: float = 1800.0


class BackendLifecycle(Protocol):
    async def ornith_start(self, unit: str) -> None: ...

    async def ornith_stop(self, unit: str) -> None: ...

    async def unit_is_active(self, unit: str) -> bool: ...

    async def daemon_start(self, model_path: str, port: int, args: list[str]) -> None: ...

    async def daemon_stop(self) -> None: ...

    async def cpu_start(self, unit: str) -> None: ...

    async def cpu_stop(self, unit: str) -> None: ...

    async def lfm_start(self, unit: str) -> None: ...

    async def lfm_stop(self, unit: str) -> None: ...


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

    async def ornith_start(self, unit: str) -> None:
        await self._systemctl("start", unit)

    async def ornith_stop(self, unit: str) -> None:
        await self._systemctl("stop", unit)

    async def unit_is_active(self, unit: str) -> bool:
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "is-active",
            "--quiet",
            unit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=self._config.lifecycle_timeout_s)
        except asyncio.CancelledError:
            proc.kill()
            await proc.communicate()
            raise
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise BackendError(
                f"systemctl --user is-active {unit} timed out after "
                f"{self._config.lifecycle_timeout_s}s"
            ) from exc
        return proc.returncode == 0

    async def _systemctl(self, verb: str, unit: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            verb,
            unit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._config.lifecycle_timeout_s
            )
        except asyncio.CancelledError:
            proc.kill()
            await proc.communicate()
            raise
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise BackendError(
                f"systemctl --user {verb} {unit} timed out after "
                f"{self._config.lifecycle_timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", "replace").strip()
            raise BackendError(f"systemctl --user {verb} {unit} failed: {detail}")

    async def cpu_start(self, unit: str) -> None:
        await self._systemctl("start", unit)

    async def cpu_stop(self, unit: str) -> None:
        await self._systemctl("stop", unit)

    async def lfm_start(self, unit: str) -> None:
        await self._systemctl("start", unit)

    async def lfm_stop(self, unit: str) -> None:
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

    async def reconcile(self) -> None:
        """Adopt only a provably safe state left by a controller restart.

        The arbiter keeps its current backend in memory, while the daemon and the two model
        servers outlive it.  On restart, never infer ownership from a stale local variable: probe
        every private endpoint and the daemon's persisted engine identity, then fail closed when
        two GPU owners or an unknown process are possible.
        """
        async with self._lock:
            daemon = await self._daemon_status()
            daemon_running = bool(daemon.get("running"))
            daemon_starting = bool(daemon.get("starting"))
            daemon_stopping = bool(daemon.get("stopping"))
            if daemon_starting or daemon_stopping:
                raise BackendError("Gemma daemon is starting or stopping; state is ambiguous")

            gemma_gpu_ready = await self._is_ready(
                self.config.gemma_gpu_url, self.config.gemma_expected_model
            )
            if daemon_running:
                try:
                    daemon_port = int(daemon.get("port") or -1)
                except (TypeError, ValueError):
                    daemon_port = -1
                if (
                    daemon.get("model") != self.config.gemma_model_path
                    or daemon_port != self.config.gemma_gpu_port
                ):
                    raise BackendError(
                        "unknown Gemma daemon is running; refusing to adopt its GPU state"
                    )
                if not gemma_gpu_ready:
                    raise BackendError(
                        "Gemma daemon reports running but its GPU endpoint is not ready"
                    )
            elif gemma_gpu_ready:
                raise BackendError(
                    "Gemma GPU endpoint is ready without a matching daemon; refusing unsafe overlap"
                )

            gemma_cpu_ready = await self._is_ready(
                self.config.gemma_cpu_url, self.config.gemma_expected_model
            )
            if gemma_gpu_ready and gemma_cpu_ready:
                raise BackendError("Gemma GPU and CPU backends are both ready")

            ornith_ready = await self._is_ready(
                self.config.ornith_url, self.config.ornith_expected_model
            )
            ornith_unit_active = await self._unit_is_active_or_none(self.config.ornith_unit)
            if ornith_unit_active is True and not ornith_ready:
                loading_ornith = ActiveBackend(
                    ModelId.ORNITH,
                    self.config.ornith_url,
                    self.config.ornith_expected_model,
                    "ornith-gpu",
                )
                try:
                    await self._wait_ready(loading_ornith)
                except BackendError as exc:
                    raise BackendError(
                        "Ornith service is active but its endpoint did not become ready; "
                        "refusing to infer a safe GPU owner"
                    ) from exc
                ornith_ready = True
            if ornith_unit_active is False and ornith_ready:
                raise BackendError(
                    "Ornith endpoint is ready while its service is inactive; "
                    "refusing to adopt a stale GPU process"
                )
            ornith_geometry = await self._ornith_cache_geometry() if ornith_ready else None
            ornith_moe = None if ornith_geometry is None else self._geometry_moe(ornith_geometry)

            lfm_ready = await self._is_ready(self.config.lfm_url, self.config.lfm_expected_model)
            lfm_unit_active = await self._unit_is_active_or_none(self.config.lfm_unit)
            if lfm_unit_active is True and not lfm_ready:
                loading_lfm = ActiveBackend(
                    ModelId.LFM, self.config.lfm_url, self.config.lfm_expected_model, "lfm-gpu"
                )
                await self._wait_ready(loading_lfm)
                lfm_ready = True
            if lfm_unit_active is False and lfm_ready:
                raise BackendError(
                    "LFM endpoint is ready while its service is inactive; refusing stale process"
                )
            if sum((gemma_gpu_ready, lfm_ready, ornith_ready)) > 1:
                raise BackendError("multiple GPU backends are ready; refusing unsafe overlap")

            if gemma_gpu_ready:
                if ornith_ready:
                    raise BackendError(
                        "Gemma GPU and Ornith overlap; refusing to adopt two GPU owners"
                    )
                self._current = ActiveBackend(
                    ModelId.GEMMA,
                    self.config.gemma_gpu_url,
                    self.config.gemma_expected_model,
                    "gemma-gpu",
                )
                return

            if gemma_cpu_ready:
                self._current = ActiveBackend(
                    ModelId.GEMMA,
                    self.config.gemma_cpu_url,
                    self.config.gemma_expected_model,
                    "gemma-cpu",
                )
                return

            if ornith_ready:
                if ornith_moe == self.config.ornith_active_slots:
                    self._current = ActiveBackend(
                        ModelId.ORNITH,
                        self.config.ornith_url,
                        self.config.ornith_expected_model,
                        "ornith-gpu",
                    )
                elif ornith_moe == self.config.ornith_parked_slots:
                    # Parked Ornith is safe but not the active request owner.  The next request
                    # will expand it transactionally before forwarding traffic.
                    self._current = None
                else:
                    raise BackendError(
                        f"Ornith cache geometry is neither active nor parked: {ornith_moe!r}"
                    )
                return

            if lfm_ready:
                self._current = ActiveBackend(
                    ModelId.LFM, self.config.lfm_url, self.config.lfm_expected_model, "lfm-gpu"
                )
                return

            self._current = None

    async def prepare(self, model_id: ModelId) -> ActiveBackend:
        """Return a ready backend, switching resources only when the requested model changes."""
        model_id = ModelId(model_id)
        started = self._monotonic()
        logger.info("prepare.start model=%s", model_id.value)
        async with self._lock:
            if self._current is not None and self._current.model_id is model_id:
                if await self._is_ready(self._current.base_url, self._current.upstream_model):
                    logger.info(
                        "prepare.warm model=%s elapsed_ms=%.1f",
                        model_id.value,
                        (self._monotonic() - started) * 1000,
                    )
                    return self._current
                if self._current.runtime == "gemma-gpu":
                    await self._safe_daemon_stop()
                elif self._current.runtime == "gemma-cpu":
                    await self._safe_cpu_stop()
                elif self._current.runtime == "lfm-gpu":
                    await self._safe_lfm_stop()
                self._current = None

            if model_id is ModelId.ORNITH:
                await self._stop_gemma_if_needed()
                await self._stop_lfm_if_needed()
                try:
                    await self._ensure_ornith_ready()
                    await self._rebuild_ornith(self.config.ornith_active_slots)
                except Exception:
                    self._current = None
                    raise
                backend = ActiveBackend(
                    ModelId.ORNITH,
                    self.config.ornith_url,
                    self.config.ornith_expected_model,
                    "ornith-gpu",
                )
                await self._wait_ready(backend)
                self._current = backend
                return backend

            if model_id is ModelId.LFM:
                logger.info("prepare.stop_previous model=%s", model_id.value)
                await self._stop_gemma_if_needed()
                try:
                    await self._stop_ornith_for_gemma()
                    self._current = None
                    logger.info("prepare.start_unit model=%s unit=%s", model_id.value, self.config.lfm_unit)
                    await self.lifecycle.lfm_start(self.config.lfm_unit)
                    backend = ActiveBackend(
                        ModelId.LFM,
                        self.config.lfm_url,
                        self.config.lfm_expected_model,
                        "lfm-gpu",
                    )
                    await self._wait_ready(backend)
                    self._current = backend
                    return backend
                except Exception:
                    await self._safe_lfm_stop()
                    self._current = None
                    raise

            try:
                ornith_was_active = await self._stop_ornith_for_gemma()
            except Exception:
                self._current = None
                raise
            # LFM2.5 and Gemma are both GPU owners.  Stop LFM even when it was
            # adopted after an arbiter restart, otherwise switching models can
            # leave two ready GPU backends and make the next request unsafe.
            await self._safe_lfm_stop()
            self._current = None
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
                restored_ornith: ActiveBackend | None = None
                restore_exc: Exception | None = None
                if ornith_was_active:
                    try:
                        restored_ornith = await self._restore_ornith_active()
                    except Exception as exc:  # noqa: BLE001 - retain the GPU and restore causes
                        restore_exc = exc
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
                    await self._safe_cpu_stop()
                    self._current = restored_ornith
                    if restore_exc is not None:
                        raise BackendError(
                            f"Gemma GPU and CPU readiness failed and Ornith restore failed; "
                            f"gpu={gpu_exc}; cpu={cpu_exc}; restore={restore_exc}"
                        ) from restore_exc
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

    async def _stop_gemma_if_needed(self) -> None:
        if self._current is None:
            return
        if self._current.runtime == "gemma-gpu":
            await self.lifecycle.daemon_stop()
        elif self._current.runtime == "gemma-cpu":
            await self.lifecycle.cpu_stop(self.config.gemma_cpu_unit)
        self._current = None

    async def _stop_lfm_if_needed(self) -> None:
        if self._current is not None and self._current.runtime == "lfm-gpu":
            await self._safe_lfm_stop()
            self._current = None

    async def _unit_is_active_or_none(self, unit: str) -> bool | None:
        probe = getattr(self.lifecycle, "unit_is_active", None)
        if probe is None:
            return None
        return bool(await probe(unit))

    async def _stop_ornith_for_gemma(self) -> bool:
        """Stop Ornith completely before giving its GPU and RAM to Gemma.

        A cache resize is intentionally not used here.  Gemma is dense and its load path needs
        the memory occupied by the Ornith process itself, not merely its expert cache.  The
        service state probe also prevents us from starting Gemma over a stale endpoint.
        """
        unit_active = await self._unit_is_active_or_none(self.config.ornith_unit)
        endpoint_ready = await self._is_ready(
            self.config.ornith_url, self.config.ornith_expected_model
        )
        if unit_active is False:
            if endpoint_ready:
                raise BackendError(
                    "Ornith endpoint is ready while its service is inactive; refusing Gemma start"
                )
            return False
        if unit_active is None and not endpoint_ready:
            return False
        await self.lifecycle.ornith_stop(self.config.ornith_unit)
        return True

    async def _ensure_ornith_ready(self) -> None:
        if await self._is_ready(self.config.ornith_url, self.config.ornith_expected_model):
            return
        await self.lifecycle.ornith_start(self.config.ornith_unit)
        backend = ActiveBackend(
            ModelId.ORNITH,
            self.config.ornith_url,
            self.config.ornith_expected_model,
            "ornith-gpu",
        )
        await self._wait_ready(backend)

    async def _restore_ornith_active(self) -> ActiveBackend:
        await self._ensure_ornith_ready()
        await self._rebuild_ornith(self.config.ornith_active_slots)
        backend = ActiveBackend(
            ModelId.ORNITH,
            self.config.ornith_url,
            self.config.ornith_expected_model,
            "ornith-gpu",
        )
        await self._wait_ready(backend)
        return backend

    async def _safe_cpu_stop(self) -> None:
        try:
            await self.lifecycle.cpu_stop(self.config.gemma_cpu_unit)
        except Exception:
            pass

    async def _safe_daemon_stop(self) -> None:
        try:
            await self.lifecycle.daemon_stop()
        except Exception:
            # A failed start may have no child.  CPU fallback remains the safe next attempt; the
            # original GPU failure is retained in the eventual BackendError if CPU also fails.
            pass

    async def _safe_lfm_stop(self) -> None:
        try:
            await self.lifecycle.lfm_stop(self.config.lfm_unit)
        except Exception:
            pass

    async def _daemon_status(self) -> dict[str, object]:
        try:
            response = await self.client.get(
                f"{self.config.daemon_url.rstrip('/')}/engine/status",
                timeout=self.config.probe_timeout_s,
            )
            response.raise_for_status()
            document = response.json()
            if not isinstance(document, dict):
                raise ValueError(f"invalid daemon status: {document!r}")
            return document
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise BackendError(f"could not reconcile Gemma daemon: {exc}") from exc

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
        geometry = await self._ornith_cache_geometry()
        kv_tokens = geometry.get("num_tokens", geometry.get("kv_tokens"))
        if kv_tokens is None:
            num_pages = geometry.get("num_pages")
            page_size = geometry.get("page_size", 1)
            if num_pages is not None:
                kv_tokens = int(num_pages) * int(page_size)
        if kv_tokens is not None and int(kv_tokens) != self.config.ornith_kv_tokens:
            raise BackendError(
                f"Ornith KV invariant changed during MoE-only rebuild: "
                f"{kv_tokens} != {self.config.ornith_kv_tokens}"
            )
        actual_moe = geometry.get("moe_cache_size")
        if actual_moe is not None and int(actual_moe) != slots:
            raise BackendError(
                f"Ornith MoE cache rebuild verification failed: {actual_moe} != {slots}"
            )
        actual_mamba = geometry.get("num_mamba_slots")
        if (
            actual_mamba is not None
            and self.config.ornith_mamba_slots is not None
            and int(actual_mamba) != self.config.ornith_mamba_slots
        ):
            raise BackendError(
                f"Ornith Mamba geometry changed during MoE-only rebuild: "
                f"{actual_mamba} != {self.config.ornith_mamba_slots}"
            )
        actual_swa = geometry.get("num_swa_pages")
        if (
            actual_swa is not None
            and self.config.ornith_swa_tokens is not None
            and int(actual_swa) != self.config.ornith_swa_tokens
        ):
            raise BackendError(
                f"Ornith SWA geometry changed during MoE-only rebuild: "
                f"{actual_swa} != {self.config.ornith_swa_tokens}"
            )

    async def _ornith_cache_geometry(self) -> dict[str, object]:
        try:
            response = await self.client.get(
                f"{self.config.ornith_url.rstrip('/')}/v1/cache/status",
                timeout=self.config.probe_timeout_s,
            )
            response.raise_for_status()
            document = response.json()
            geometry = document.get("geometry") or document
            if not isinstance(geometry, dict):
                raise ValueError(f"invalid cache geometry: {document!r}")
            return geometry
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise BackendError(f"could not verify Ornith cache geometry: {exc}") from exc

    def _geometry_moe(self, geometry: dict[str, object]) -> int:
        actual_moe = geometry.get("moe_cache_size")
        if actual_moe is None:
            raise BackendError("Ornith cache geometry has no moe_cache_size")
        try:
            return int(actual_moe)
        except (TypeError, ValueError) as exc:
            raise BackendError(f"invalid Ornith moe_cache_size: {actual_moe!r}") from exc

    async def _is_ready(self, base_url: str, expected_model: str) -> bool:
        try:
            health = await self.client.get(
                f"{base_url.rstrip('/')}/health", timeout=self.config.probe_timeout_s
            )
            if health.status_code != 200 or health.json().get("status") != "ok":
                return False
            models = await self.client.get(
                f"{base_url.rstrip('/')}/v1/models", timeout=self.config.probe_timeout_s
            )
            models.raise_for_status()
            data = models.json().get("data") or []
            return any(item.get("id") == expected_model for item in data)
        except (httpx.HTTPError, ValueError, OSError):
            return False

    async def _wait_ready(self, backend: ActiveBackend) -> None:
        deadline = self._monotonic() + self.config.readiness_timeout_s
        wait_started = self._monotonic()
        probes = 0
        while self._monotonic() < deadline:
            probes += 1
            if await self._is_ready(backend.base_url, backend.upstream_model):
                logger.info(
                    "backend.ready model=%s runtime=%s probes=%d elapsed_ms=%.1f",
                    backend.model_id.value,
                    backend.runtime,
                    probes,
                    (self._monotonic() - wait_started) * 1000,
                )
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
