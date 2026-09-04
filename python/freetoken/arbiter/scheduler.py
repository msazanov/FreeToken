from __future__ import annotations

import asyncio
from collections import deque
from itertools import count
from typing import Deque

from .model import LeaseState, ModelId, QueuedRequest


class Lease:
    def __init__(self, scheduler: LeaseScheduler, request: QueuedRequest) -> None:
        self._scheduler = scheduler
        self._request = request
        self._released = False
        self._release_lock = asyncio.Lock()

    @property
    def model_id(self) -> ModelId:
        return self._request.model_id

    @property
    def request_id(self) -> str:
        return self._request.request_id

    @property
    def sequence(self) -> int:
        return self._request.sequence

    async def release(self) -> None:
        async with self._release_lock:
            if self._released:
                return
            await self._scheduler._release(self)
            self._released = True

    async def __aenter__(self) -> Lease:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.release()


class LeaseScheduler:
    def __init__(
        self,
        *,
        tie_break: ModelId = ModelId.ORNITH,
        max_queue_depth: int | None = None,
    ) -> None:
        self._tie_break = ModelId(tie_break)
        if max_queue_depth is not None and max_queue_depth < 0:
            raise ValueError("max_queue_depth must be non-negative")
        self._max_queue_depth = max_queue_depth
        self._condition = asyncio.Condition()
        self._queues: dict[ModelId, Deque[QueuedRequest]] = {
            ModelId.ORNITH: deque(),
            ModelId.GEMMA: deque(),
            ModelId.LFM: deque(),
        }
        self._sequence = count()
        self._active_request: QueuedRequest | None = None
        self._state = LeaseState.IDLE
        self._idle_drain_scheduled = False

    async def acquire(
        self,
        model_id: ModelId,
        request_id: str,
        *,
        timeout: float | None = None,
    ) -> Lease:
        model_id = ModelId(model_id)
        request = QueuedRequest(model_id, request_id, next(self._sequence))
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout

        async with self._condition:
            queue = self._queues[model_id]
            if (
                self._max_queue_depth is not None
                and len(queue) >= self._max_queue_depth
                and (self._active_request is not None or any(self._queues.values()))
            ):
                raise asyncio.QueueFull
            queue.append(request)
            if self._active_request is None:
                self._schedule_idle_drain()

            try:
                while self._active_request is not request:
                    if deadline is None:
                        await self._condition.wait()
                    else:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        await asyncio.wait_for(self._condition.wait(), remaining)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                if self._active_request is request:
                    self._active_request = None
                    self._state = LeaseState.IDLE
                    self._grant_next_locked(preferred=model_id)
                elif request in queue:
                    queue.remove(request)
                    self._condition.notify_all()
                raise

        return Lease(self, request)

    def snapshot(self) -> dict[str, object]:
        owner = None if self._active_request is None else self._active_request.model_id
        return {
            "owner": owner,
            "state": self._state,
            "active_count": 0 if self._active_request is None else 1,
            "queue_depths": {model_id: len(queue) for model_id, queue in self._queues.items()},
        }

    def _schedule_idle_drain(self) -> None:
        if self._idle_drain_scheduled:
            return
        self._idle_drain_scheduled = True
        asyncio.get_running_loop().call_soon(self._start_idle_drain)

    def _start_idle_drain(self) -> None:
        self._idle_drain_scheduled = False
        asyncio.create_task(self._drain_idle())

    async def _drain_idle(self) -> None:
        await asyncio.sleep(0)
        async with self._condition:
            if self._active_request is None:
                self._grant_next_locked()

    async def _release(self, lease: Lease) -> None:
        # Let already-created waiters enqueue before the switch fence is taken.
        await asyncio.sleep(0)
        async with self._condition:
            if self._active_request is not lease._request:
                return
            previous_model = lease.model_id
            self._active_request = None
            self._state = LeaseState.IDLE
            self._grant_next_locked(preferred=previous_model)

    def _grant_next_locked(self, preferred: ModelId | None = None) -> None:
        if self._active_request is not None:
            return
        available = [model_id for model_id, queue in self._queues.items() if queue]
        if not available:
            self._condition.notify_all()
            return
        if preferred in available:
            model_id = preferred
        elif len(available) == 2:
            model_id = self._tie_break
        else:
            model_id = available[0]
        self._active_request = self._queues[model_id].popleft()
        self._state = LeaseState.ACTIVE
        self._condition.notify_all()
