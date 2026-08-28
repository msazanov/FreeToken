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
