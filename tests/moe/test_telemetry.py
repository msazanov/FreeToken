from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from freetoken.engine.engine import _configure_moe_telemetry, _adjust_config
from freetoken.engine.config import EngineConfig
from freetoken.distributed import DistributedInfo
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache


def test_parser_accepts_protected_layer_policy():
    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"}

    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, _ = parse_args(
            ["--model", "/models/anon", "--moe-cache-policy", "protected_layer"]
        )

    assert args.moe_cache_policy == "protected_layer"


def test_terminal_telemetry_contains_only_scalar_protected_layer_geometry():
    cache = OffloadMoeCache(
        num_layers=48,
        num_experts=512,
        cache_size=256,
        device=torch.device("cpu"),
        quant_format="qwen4_gguf",
        minimum_cache_size=10,
        cache_policy="protected_layer",
    )

    geometry = cache.telemetry_snapshot()["cache"]

    assert geometry["policy"] == "protected_layer"
    assert geometry["protected_slots_per_layer"] == 5
    assert geometry["protected_slot_count"] == 240
    assert geometry["transient_slots"] == 16
    assert all(isinstance(value, (int, str)) for value in geometry.values())


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


def test_standard_prefill_records_routes_before_materialization_and_counts_evictions(monkeypatch):
    """The ordinary prefill path does not call ensure_experts()."""
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
    )
    cache.collect_stats = True
    cache.set_trace_phase("prefill")
    cache.id_of_slot.copy_(torch.tensor([4, 1, -1, 3], dtype=torch.int32))
    cache.slot_for_id[1, 0] = 0
    cache.slot_for_id[0, 1] = 1
    cache.slot_for_id[0, 3] = 3

    def materialize(cache, layer_id):
        cache.id_of_slot.copy_(torch.arange(4, 8, dtype=torch.int32))
        cache.slot_for_id.fill_(-1)
        cache.slot_for_id[layer_id].copy_(torch.arange(4, dtype=torch.int32))

    monkeypatch.setattr("freetoken.moe.offload_kernels.materialize_layer", materialize)

    cache.record_prefill_routes(1, torch.tensor([[0, 1], [1, 2]], dtype=torch.int32))
    cache.materialize_layer(1)
    layer = cache.telemetry_snapshot()["trace"]["prefill"]["layers"][1]

    assert {key: layer[key] for key in (
        "route_references", "route_unique", "l1_hits", "l1_misses", "evictions"
    )} == {
        "route_references": 4,
        "route_unique": 3,
        "l1_hits": 1,
        "l1_misses": 2,
        "evictions": 2,
    }


def test_offload_moe_prefill_routed_records_standard_prefill_trace(monkeypatch):
    """The standard host-bank production branch records routes before materializing."""
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
    )
    cache.collect_stats = True
    cache.set_trace_phase("prefill")
    cache.slot_for_id[0, 1] = 1
    monkeypatch.setattr(cache, "materialize_layer", lambda layer_id: None)
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    monkeypatch.setattr(cache, "bank_views", lambda n: ())

    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.layer_id = 0
    layer.num_experts = 4
    layer.offload_cache = cache
    layer._expert_gemm = lambda *args, **kwargs: torch.zeros((2, 3))

    output = layer._prefill_routed(
        torch.zeros((2, 3)),
        torch.ones((2, 2)),
        torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )

    assert output.shape == (2, 3)
    trace = cache.telemetry_snapshot()["trace"]["prefill"]["layers"][0]
    assert {key: trace[key] for key in (
        "route_references", "route_unique", "l1_hits", "l1_misses"
    )} == {
        "route_references": 4,
        "route_unique": 3,
        "l1_hits": 1,
        "l1_misses": 2,
    }


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
