from types import SimpleNamespace

from freetoken.message import UserReply
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
