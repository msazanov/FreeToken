from __future__ import annotations

import asyncio

import pytest

from freetoken.arbiter import LeaseScheduler, LeaseState, ModelId, QueuedRequest


def _run(coro):
    return asyncio.run(coro)


def test_model_contract_has_stable_ids_and_request_ordering():
    request = QueuedRequest(ModelId.GEMMA, "g1", 7)

    assert ModelId.ORNITH.value == "ornith-35b"
    assert ModelId.GEMMA.value == "gemma-4-e2b"
    assert request.model_id is ModelId.GEMMA
    assert request.request_id == "g1"
    assert request.sequence == 7


def test_active_model_drains_before_other_model():
    async def scenario():
        scheduler = LeaseScheduler(tie_break=ModelId.ORNITH)
        first = await scheduler.acquire(ModelId.GEMMA, "g1")
        waiting_o = asyncio.create_task(scheduler.acquire(ModelId.ORNITH, "o1"))
        waiting_g = asyncio.create_task(scheduler.acquire(ModelId.GEMMA, "g2"))

        await first.release()
        next_lease = await waiting_g

        assert next_lease.model_id is ModelId.GEMMA
        assert not waiting_o.done()
        await next_lease.release()
        await (await waiting_o).release()

    _run(scenario())


def test_each_model_queue_is_fifo():
    async def scenario():
        scheduler = LeaseScheduler()
        first = await scheduler.acquire(ModelId.ORNITH, "o1")
        waiting = [
            asyncio.create_task(scheduler.acquire(ModelId.ORNITH, request_id))
            for request_id in ("o2", "o3")
        ]
        await asyncio.sleep(0)

        await first.release()
        second = await waiting[0]
        await second.release()
        third = await waiting[1]

        assert [second.request_id, third.request_id] == ["o2", "o3"]
        await third.release()

    _run(scenario())


def test_idle_tie_break_prefers_configured_model_when_both_queues_are_ready():
    async def scenario():
        scheduler = LeaseScheduler(tie_break=ModelId.ORNITH)
        waiting_g = asyncio.create_task(scheduler.acquire(ModelId.GEMMA, "g1"))
        waiting_o = asyncio.create_task(scheduler.acquire(ModelId.ORNITH, "o1"))
        await asyncio.sleep(0)

        first = await waiting_o
        assert first.model_id is ModelId.ORNITH
        assert not waiting_g.done()

        await first.release()
        await (await waiting_g).release()

    _run(scenario())


def test_cancelled_waiter_is_removed_and_does_not_block_fifo():
    async def scenario():
        scheduler = LeaseScheduler()
        first = await scheduler.acquire(ModelId.ORNITH, "o1")
        cancelled = asyncio.create_task(scheduler.acquire(ModelId.ORNITH, "o2"))
        survivor = asyncio.create_task(scheduler.acquire(ModelId.ORNITH, "o3"))
        await asyncio.sleep(0)

        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        await first.release()
        next_lease = await survivor
        assert next_lease.request_id == "o3"
        await next_lease.release()
        assert scheduler.snapshot()["queue_depths"] == {
            ModelId.ORNITH: 0,
            ModelId.GEMMA: 0,
            ModelId.LFM: 0,
        }

    _run(scenario())


def test_queue_limit_rejects_only_excess_waiters():
    async def scenario():
        scheduler = LeaseScheduler(max_queue_depth=1)
        first = await scheduler.acquire(ModelId.GEMMA, "g1")
        queued = asyncio.create_task(scheduler.acquire(ModelId.GEMMA, "g2"))
        await asyncio.sleep(0)

        with pytest.raises(asyncio.QueueFull):
            await scheduler.acquire(ModelId.GEMMA, "g3")

        await first.release()
        await (await queued).release()

    _run(scenario())


def test_zero_queue_depth_still_allows_one_active_lease():
    async def scenario():
        scheduler = LeaseScheduler(max_queue_depth=0)
        first = await scheduler.acquire(ModelId.GEMMA, "g1")

        with pytest.raises(asyncio.QueueFull):
            await scheduler.acquire(ModelId.GEMMA, "g2")

        assert scheduler.snapshot()["active_count"] == 1
        await first.release()

    _run(scenario())


def test_cancelled_granted_waiter_does_not_leave_active_lease_stuck():
    async def scenario():
        scheduler = LeaseScheduler()
        first = await scheduler.acquire(ModelId.GEMMA, "g1")
        waiting = asyncio.create_task(scheduler.acquire(ModelId.ORNITH, "o1"))
        await asyncio.sleep(0)

        await first.release()
        assert scheduler.snapshot()["owner"] is ModelId.ORNITH

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert scheduler.snapshot() == {
            "owner": None,
            "state": LeaseState.IDLE,
            "active_count": 0,
            "queue_depths": {
                ModelId.ORNITH: 0,
                ModelId.GEMMA: 0,
                ModelId.LFM: 0,
            },
        }

    _run(scenario())


def test_cancelled_release_can_be_retried_after_handoff_cancellation():
    async def scenario():
        scheduler = LeaseScheduler()
        first = await scheduler.acquire(ModelId.GEMMA, "g1")
        release_task = asyncio.create_task(first.release())
        await asyncio.sleep(0)
        assert not release_task.done()

        release_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release_task

        assert scheduler.snapshot()["owner"] is ModelId.GEMMA
        await first.release()
        assert scheduler.snapshot()["state"] is LeaseState.IDLE

    _run(scenario())


def test_waiter_timeout_is_removed_from_snapshot():
    async def scenario():
        scheduler = LeaseScheduler()
        first = await scheduler.acquire(ModelId.GEMMA, "g1")

        with pytest.raises(asyncio.TimeoutError):
            await scheduler.acquire(ModelId.ORNITH, "o1", timeout=0.001)

        snapshot = scheduler.snapshot()
        assert snapshot["owner"] is ModelId.GEMMA
        assert snapshot["state"] is LeaseState.ACTIVE
        assert snapshot["active_count"] == 1
        assert snapshot["queue_depths"] == {
            ModelId.ORNITH: 0,
            ModelId.GEMMA: 0,
            ModelId.LFM: 0,
        }
        await first.release()

    _run(scenario())


def test_async_context_releases_lease_and_snapshot_has_no_overlap():
    async def scenario():
        scheduler = LeaseScheduler()
        async with await scheduler.acquire(ModelId.ORNITH, "o1") as lease:
            assert lease.model_id is ModelId.ORNITH
            snapshot = scheduler.snapshot()
            assert snapshot == {
                "owner": ModelId.ORNITH,
                "state": LeaseState.ACTIVE,
                "active_count": 1,
                "queue_depths": {
                    ModelId.ORNITH: 0,
                    ModelId.GEMMA: 0,
                    ModelId.LFM: 0,
                },
            }

        assert scheduler.snapshot() == {
            "owner": None,
            "state": LeaseState.IDLE,
            "active_count": 0,
            "queue_depths": {
            ModelId.ORNITH: 0,
            ModelId.GEMMA: 0,
            ModelId.LFM: 0,
            },
        }

    _run(scenario())
