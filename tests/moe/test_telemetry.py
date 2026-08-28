from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace

from freetoken.engine.engine import _configure_moe_telemetry, _adjust_config
from freetoken.engine.config import EngineConfig
from freetoken.distributed import DistributedInfo
from freetoken.moe.offload_cache import OffloadMoeCache


def test_reset_stats_clears_routing_frequency_and_snapshot_labels_stationary_top_c():
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
    )
    cache.decode_freq.fill_(3)
    cache.reset_stats()
    assert torch.count_nonzero(cache.decode_freq).item() == 0

    cache.collect_decode_freq = True
    cache.decode_freq[0, 1] = 4
    snapshot = cache.telemetry_snapshot()

    assert snapshot["miss"] == cache.decode_miss_stats()
    assert snapshot["per_layer"] == cache.decode_miss_stats_per_layer()["per_layer"]
    assert snapshot["routing"]["method"] == "stationary_per_layer_top_c"
    assert "dynamic_oracle" not in snapshot["routing"]
    assert "oracle_hit_at_slots" not in snapshot["routing"]
    assert "stationary_top_c_hit_at_slots" in snapshot["routing"]
    assert snapshot["routing_frequency_enabled"] is True
    assert snapshot["cache"]["num_layers"] == 2
    assert snapshot["cache"]["num_experts"] == 4
    assert snapshot["cache"]["cache_size"] == 4


def test_auto_cuda_graphs_do_not_enable_routing_frequency_collection():
    cache = SimpleNamespace()
    config = SimpleNamespace(moe_collect_stats=True, cuda_graph_bs=None)

    _configure_moe_telemetry(cache, config)

    assert cache.collect_stats is True
    assert cache.collect_decode_freq is False


def test_telemetry_rejects_multiple_running_requests():
    config = EngineConfig(
        model_path="x",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=2,
        moe_collect_stats=True,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            is_moe=True,
            expert_quant="none",
            has_swa_attention=False,
            has_linear_attention=False,
            dsv4_args=None,
            single_stream_only=False,
        ),
    )

    with pytest.raises(ValueError, match="requires --max-running-requests 1"):
        _adjust_config(config)
