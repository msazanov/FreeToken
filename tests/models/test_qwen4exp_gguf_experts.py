from __future__ import annotations

from types import SimpleNamespace

import torch


class _Tensor:
    def __init__(self, name: str, packed: torch.Tensor, ggml_type: int):
        self.name = name
        self._packed = packed
        self.ggml_type = ggml_type

    def packed(self) -> torch.Tensor:
        return self._packed


def test_qwen4_gguf_expert_sources_keep_gate_up_and_down_as_direct_views(monkeypatch):
    """The Qwen4 provider must never concatenate the two input projections.

    Qwen3.8's experts exceed this host's RAM if gate/up are materialized into
    a combined bank.  A provider therefore returns the three original GGUF
    tensor views, whose storage addresses prove no copy or cat occurred.
    """
    from freetoken.models.gguf.dequant import GGML_IQ2_S, GGML_IQ4_NL
    from freetoken.models.gguf import reader

    experts, hidden, intermediate = 2, 32, 32
    gate = torch.arange(experts * intermediate * 7, dtype=torch.uint8).reshape(
        experts * intermediate, 7
    )
    up = (gate + 1).contiguous()
    down = torch.arange(experts * hidden * 9, dtype=torch.uint8).reshape(experts * hidden, 9)
    tensors = [
        _Tensor("blk.0.ffn_gate_exps.weight", gate, GGML_IQ2_S),
        _Tensor("blk.0.ffn_up_exps.weight", up, GGML_IQ2_S),
        _Tensor("blk.0.ffn_down_exps.weight", down, GGML_IQ4_NL),
    ]
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: iter(tensors))

    from freetoken.models.qwen4_exp.gguf_experts import (
        gguf_expert_types,
        load_gguf_expert_sources,
    )

    config = SimpleNamespace(
        num_layers=1,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
    )
    sources = load_gguf_expert_sources("fixture.gguf", config)
    types = gguf_expert_types("fixture.gguf", config.num_layers)

    assert tuple(sources) == ("gate", "up", "down")
    assert sources["gate"][0].shape == (experts, intermediate, 7)
    assert sources["up"][0].shape == (experts, intermediate, 7)
    assert sources["down"][0].shape == (experts, hidden, 9)
    assert sources["gate"][0].data_ptr() == gate.data_ptr()
    assert sources["up"][0].data_ptr() == up.data_ptr()
    assert sources["down"][0].data_ptr() == down.data_ptr()
    assert types == {"gate": [GGML_IQ2_S], "up": [GGML_IQ2_S], "down": [GGML_IQ4_NL]}


def test_qwen4_gguf_provider_marks_three_file_backed_banks(monkeypatch):
    """The generic provider must preserve the Qwen4 three-bank contract."""
    from freetoken.models import qwen4_exp
    from freetoken.models.gguf.dequant import GGML_IQ2_S, GGML_IQ4_NL
    from freetoken.moe.expert_banks import _gguf_banks

    source = {
        "gate": [torch.zeros(2, 4, 3, dtype=torch.uint8)],
        "up": [torch.zeros(2, 4, 3, dtype=torch.uint8)],
        "down": [torch.zeros(2, 8, 5, dtype=torch.uint8)],
    }
    monkeypatch.setattr(qwen4_exp, "load_gguf_expert_sources", lambda *_a, **_k: source)
    monkeypatch.setattr(
        qwen4_exp,
        "gguf_expert_types",
        lambda *_a, **_k: {"gate": [GGML_IQ2_S], "up": [GGML_IQ2_S], "down": [GGML_IQ4_NL]},
    )
    config = SimpleNamespace(
        architectures=["Qwen4ExpForCausalLM"], num_layers=1, num_experts=2
    )

    banks = _gguf_banks("fixture.gguf", config, torch.device("cpu"), torch.bfloat16, False)

    assert banks.quant_format == "qwen4_gguf"
    assert tuple(banks.sources) == ("gate", "up", "down")
    assert banks.layer_residency == ["file_backed"]
    assert banks.gguf_expert_types == ((GGML_IQ2_S,), (GGML_IQ2_S,), (GGML_IQ4_NL,))


def test_qwen4_file_backed_cache_does_not_require_cpu_decode():
    """Pageable NVMe ranges are copied after routing, not treated as CPU experts."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=2,
        cache_size=2,
        device=torch.device("cpu"),
        quant_format="qwen4_gguf",
    )
    cache.set_bank_sources(
        {
            "gate": [torch.zeros(2, 4, 3, dtype=torch.uint8)],
            "up": [torch.zeros(2, 4, 3, dtype=torch.uint8)],
            "down": [torch.zeros(2, 8, 5, dtype=torch.uint8)],
        },
        layer_residency=["file_backed"],
    )

    assert cache.is_file_backed_layer(0)
    assert not cache.is_cpu_layer(0)


def test_qwen4_file_backed_copy_batches_selected_rows_without_per_row_dma(monkeypatch):
    """A routed GGUF miss gathers each bank once, then scatters into LRU slots."""
    from freetoken.moe import offload_cache
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
        quant_format="qwen4_gguf",
    )
    sources = {
        "gate": [torch.arange(4 * 2 * 3, dtype=torch.uint8).reshape(4, 2, 3)],
        "up": [torch.arange(40, 64, dtype=torch.uint8).reshape(4, 2, 3)],
        "down": [torch.arange(4 * 3 * 2, dtype=torch.uint8).reshape(4, 3, 2)],
    }
    cache.set_bank_sources(sources, layer_residency=["file_backed"])
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    cache.num_indices.fill_(3)
    cache.evict_slots[:3] = torch.tensor([3, 1, 0], dtype=torch.int32)
    cache.src_indices[:3] = torch.tensor([0, 2, 1], dtype=torch.int32)

    def old_per_row_copy(*_args, **_kwargs):
        raise AssertionError("file-backed selected copies must be batched per bank")

    monkeypatch.setattr(offload_cache, "_copy_compact_row_prefix", old_per_row_copy)
    cache.copy_missing()

    for name, source in sources.items():
        actual = cache.bank_caches[name]
        assert torch.equal(actual[3], source[0][0])
        assert torch.equal(actual[1], source[0][2])
        assert torch.equal(actual[0], source[0][1])


def test_qwen4_file_backed_lru_admission_avoids_shape_unrolled_flashlib_kernel(monkeypatch):
    """Qwen routed prefill must use the dynamic in-tree admission implementation."""
    from freetoken.moe import offload_kernels
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
        quant_format="qwen4_gguf",
    )
    source = torch.arange(4 * 2 * 3, dtype=torch.uint8).reshape(4, 2, 3)
    cache.set_bank_sources(
        {"gate": [source], "up": [source], "down": [source]},
        layer_residency=["file_backed"],
    )
    ids = torch.tensor([[3, 1, 3]], dtype=torch.int32)

    def unexpected_flashlib(*_args, **_kwargs):
        raise AssertionError("file-backed Qwen must not compile flashlib's static LRU kernel")

    monkeypatch.setattr(offload_kernels, "lru_ensure", unexpected_flashlib)
    cache.ensure_experts(0, ids)

    assert sorted(cache.src_indices[: int(cache.num_indices.item())].tolist()) == [1, 3]
    assert (ids >= 0).all()
    assert ids[0, 0].item() == ids[0, 2].item()


def test_qwen4_gguf_dispatch_passes_three_quant_types(monkeypatch):
    """The separate banks must reach a separate-projection GGUF GEMV path."""
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.models.gguf.dequant import GGML_IQ2_S, GGML_IQ4_NL
    from freetoken.moe import fused_q4_0
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.distributed.info import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    calls = []

    def fake_fused(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full((1, 8), 7.0)

    monkeypatch.setattr(fused_q4_0, "fused_experts_gguf_separate", fake_fused, raising=False)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=2,
        cache_size=2,
        device=torch.device("cpu"),
        quant_format="qwen4_gguf",
        gguf_expert_types=((GGML_IQ2_S,), (GGML_IQ2_S,), (GGML_IQ4_NL,)),
    )
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=2,
        top_k=1,
        hidden_size=8,
        intermediate_size=4,
        activation="silu",
    )
    hidden = torch.zeros(1, 8)
    result = layer._expert_gemm(
        cache,
        hidden,
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.int32),
        views=(
            torch.zeros(2, 4, 3, dtype=torch.uint8),
            torch.zeros(2, 4, 3, dtype=torch.uint8),
            torch.zeros(2, 8, 5, dtype=torch.uint8),
        ),
        n=None,
        alphas=None,
        is_prefill=False,
    )

    assert torch.equal(result, torch.full((1, 8), 7.0))
    assert calls[0][1]["gate_quant_type"] == GGML_IQ2_S
    assert calls[0][1]["up_quant_type"] == GGML_IQ2_S
    assert calls[0][1]["down_quant_type"] == GGML_IQ4_NL


def test_qwen4_file_backed_prefill_routes_experts_instead_of_materializing_layer(monkeypatch):
    """NVMe-backed prefill must not copy all 512 experts before routing."""
    from freetoken.distributed.info import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    events = []

    class _Cache:
        prefill_overlap = False

        def is_file_backed_layer(self, layer_id):
            assert layer_id == 0
            return True

        def ensure_experts(self, layer_id, ids):
            events.append("ensure")
            ids.add_(10)  # model the normal LRU rewrite to GPU slots

        def copy_missing(self):
            events.append("copy")

        def bank_views(self):
            return ("gate-slots", "up-slots", "down-slots")

        def alphas_for_slots(self, layer_id):
            return None

        def materialize_layer(self, layer_id):
            raise AssertionError("file-backed prefill must not materialize a whole expert layer")

    layer = OffloadMoELayer(
        layer_id=0, num_experts=2, top_k=1, hidden_size=8, intermediate_size=4, activation="silu"
    )
    layer.offload_cache = _Cache()
    seen = {}

    def fake_gemm(_self, _cache, _hidden, _weights, ids, **kwargs):
        seen["ids"] = ids.clone()
        seen.update(kwargs)
        return torch.full((1, 8), 3.0)

    monkeypatch.setattr(OffloadMoELayer, "_expert_gemm", fake_gemm)
    ids = torch.zeros(1, 1, dtype=torch.int32)
    result = layer._prefill_routed(torch.zeros(1, 8), torch.ones(1, 1), ids)

    assert torch.equal(result, torch.full((1, 8), 3.0))
    assert events == ["ensure", "copy"]
    assert seen["ids"].item() == 10
    assert seen["views"] == ("gate-slots", "up-slots", "down-slots")
    assert seen["n"] is None


def test_file_backed_qwen_cache_allows_router_sized_lru_on_cpu():
    """Unlike generic full-layer prefill, Qwen4 GGUF need not reserve 512 slots."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=48,
        num_experts=512,
        cache_size=10,
        device=torch.device("cpu"),
        quant_format="qwen4_gguf",
        minimum_cache_size=10,
    )

    assert cache.cache_size == 10
