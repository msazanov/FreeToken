from contextlib import nullcontext
from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.message import DetokenizeMsg, UserMsg, UserReply
from freetoken.scheduler.scheduler import Scheduler
from freetoken.server.stats import StatsTracker, build_stats


def _state(stats: StatsTracker) -> SimpleNamespace:
    return SimpleNamespace(
        stats=stats,
        config=SimpleNamespace(
            served_model_name="test-model",
            max_seq_len=4096,
            page_size=1,
            model_config=SimpleNamespace(
                has_linear_attention=False,
                has_swa_attention=False,
                is_moe=True,
                dsv4_args=None,
            ),
        ),
        ready_at=None,
        gpus=[],
    )


def test_stats_tracker_publishes_only_the_terminal_moe_payload():
    stats = StatsTracker()
    state = _state(stats)

    assert build_stats(state, p95_ms=0, ttft_mean_ms=0)["moe"] is None

    stats.on_new_user(7)
    stats.observe(UserReply(uid=7, incremental_output="", finished=False))
    assert build_stats(state, p95_ms=0, ttft_mean_ms=0)["moe"] is None

    moe_stats = {"schema_version": 1, "miss": {"miss_rate": 0.5}}
    stats.observe(
        UserReply(uid=7, incremental_output="done", finished=True, moe_stats=moe_stats)
    )

    document = build_stats(state, p95_ms=12, ttft_mean_ms=3)
    assert document["moe"] == moe_stats
    assert document["instance_id"] is None
    assert document["requests"]["completed"] == 1
    assert document["requests"]["p95_ms"] == 12
    assert document["requests"]["ttft_mean_ms"] == 3


def test_moe_reset_waits_for_first_prefill_batch_not_user_queue_arrival():
    events = []
    cache = SimpleNamespace(reset_stats=lambda: events.append("reset"))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(moe_collect_stats=True)
    scheduler.engine = SimpleNamespace(max_seq_len=64, moe_offload_cache=cache)
    scheduler.prefill_manager = SimpleNamespace(
        add_one_req=lambda _msg: events.append("queued"),
    )
    scheduler.send_result = lambda _messages: None

    Scheduler._process_one_msg(
        scheduler,
        UserMsg(
            uid=2,
            input_ids=torch.tensor([1, 2], dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=1),
        ),
    )
    assert events == ["queued"]

    from freetoken.scheduler.scheduler import _reset_moe_stats_for_prefill

    queued_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    first_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[(2, 2, 0)])
    _reset_moe_stats_for_prefill(scheduler.config, scheduler.engine, queued_batch)
    assert events == ["queued"]
    _reset_moe_stats_for_prefill(scheduler.config, scheduler.engine, first_batch)
    assert events == ["queued", "reset"]


def test_moe_snapshot_follows_copy_done_synchronize():
    events = []
    cache = SimpleNamespace(
        telemetry_snapshot=lambda: events.append("snapshot") or {"schema_version": 1}
    )
    copy_done = SimpleNamespace(synchronize=lambda: events.append("synchronize"))
    req = Req(
        input_ids=torch.tensor([1], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=3,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=None,
    )
    req.complete_one()
    scheduler = SimpleNamespace(
        config=SimpleNamespace(moe_collect_stats=True, page_size=1),
        engine=SimpleNamespace(moe_offload_cache=cache),
        cache_manager=SimpleNamespace(lazy_free_region=lambda: nullcontext()),
        decode_manager=SimpleNamespace(remove_req=lambda _req: None, running_reqs=set()),
        prefill_manager=SimpleNamespace(pending_list=[]),
        finished_reqs=set(),
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        _free_req_resources=lambda _req: None,
        _kv_usage_pages=lambda: (0, 0),
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
        status_reporter=SimpleNamespace(report_batch=lambda *_args, **_kwargs: None),
        send_result=lambda replies: events.append(replies),
    )
    batch = SimpleNamespace(reqs=[req], is_prefill=False)
    last_data = (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42], dtype=torch.int32), copy_done),
    )

    Scheduler._process_last_data(scheduler, last_data)

    assert events[0:2] == ["synchronize", "snapshot"]
    replies = events[-1]
    assert len(replies) == 1
    assert isinstance(replies[0], DetokenizeMsg)
    assert replies[0].moe_stats == {"schema_version": 1}
