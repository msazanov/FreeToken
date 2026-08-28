"""File-backed routed experts for Qwen3.8 Flash Next GGUF checkpoints.

The normal GGUF MoE provider builds a combined ``gate_up`` tensor.  That is
correct for models whose two input projections can safely be copied into a
single host bank, but Qwen3.8 Q4_K_M would require roughly 50 GiB of extra
anonymous RAM for that copy.  Its three GGUF tensors are already expert-major
contiguous ranges, so retain them as direct reader views instead.

The returned tensors are backed by ``GGUFReader``'s file mapping.  They use no
HostBank allocation and are intentionally pageable: the dedicated Qwen4 cache
path performs synchronous selected-expert H2D copies after routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


_EXPERT_SUFFIXES = {
    "ffn_gate_exps.weight": "gate",
    "ffn_up_exps.weight": "up",
    "ffn_down_exps.weight": "down",
}


def _expert_tensors(model_path: str, num_layers: int):
    """Yield ``(layer, bank_name, gguf_tensor)`` for Qwen4 routed experts."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    for tensor in iter_gguf_tensors(model_path):
        if not tensor.name.startswith("blk."):
            continue
        _, raw_layer, suffix = tensor.name.split(".", 2)
        layer = int(raw_layer)
        bank = _EXPERT_SUFFIXES.get(suffix)
        if bank is not None and layer < num_layers:
            yield layer, bank, tensor


def gguf_expert_types(model_path: str, num_layers: int) -> dict[str, list[int]]:
    """Return exact per-layer GGML types for separate Qwen4 expert banks."""
    types: dict[str, list[int | None]] = {
        name: [None] * num_layers for name in ("gate", "up", "down")
    }
    for layer, bank, tensor in _expert_tensors(model_path, num_layers):
        types[bank][layer] = tensor.ggml_type

    missing = {
        name: [layer for layer, value in enumerate(values) if value is None]
        for name, values in types.items()
    }
    if any(missing.values()):
        raise ValueError(f"missing Qwen4 GGUF expert tensors: {missing}")
    return {name: [int(value) for value in values] for name, values in types.items()}


def load_gguf_expert_sources(
    model_path: str, config: "ModelConfig", *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Expose Qwen4's original expert ranges as zero-copy file-backed views.

    ``layer_sink`` is not meaningful for an immutable GGUF mapping: the source
    must remain available for later cache misses, so conversion through the
    standard materialize-and-release contract is explicitly rejected.
    """
    if layer_sink is not None:
        raise NotImplementedError(
            "Qwen4 GGUF experts remain file-backed for on-demand routing; "
            "they cannot be streamed into a releasing conversion sink"
        )

    layers = config.num_layers
    experts = config.num_experts
    intermediate = config.moe_intermediate_size
    hidden = config.hidden_size
    sources: dict[str, list[torch.Tensor | None]] = {
        name: [None] * layers for name in ("gate", "up", "down")
    }

    for layer, bank, tensor in _expert_tensors(model_path, layers):
        packed = tensor.packed()
        expected_rows = experts * (hidden if bank == "down" else intermediate)
        if packed.ndim != 2 or packed.shape[0] != expected_rows:
            raise ValueError(
                f"{tensor.name}: expected {expected_rows} packed rows for Qwen4 {bank}, "
                f"got {tuple(packed.shape)}"
            )
        rows = hidden if bank == "down" else intermediate
        # ``packed`` is expert-major because GGUF's fastest-first dimensions are
        # [input, output, expert].  reshape is a view: this is the invariant that
        # keeps the NVMe file rather than a second 50-GiB RAM representation.
        source = packed.reshape(experts, rows, packed.shape[1])
        if not source.is_contiguous():
            raise ValueError(f"{tensor.name}: GGUF packed expert rows must be contiguous")
        sources[bank][layer] = source

    missing = {
        name: [layer for layer, source in enumerate(values) if source is None]
        for name, values in sources.items()
    }
    if any(missing.values()):
        raise ValueError(f"missing Qwen4 GGUF expert source views: {missing}")
    return {name: [source for source in values if source is not None] for name, values in sources.items()}


__all__ = ["gguf_expert_types", "load_gguf_expert_sources"]
