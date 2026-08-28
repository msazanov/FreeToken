from __future__ import annotations

import torch

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
