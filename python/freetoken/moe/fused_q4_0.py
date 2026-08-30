"""Grouped expert GEMM over native GGUF banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).

This module is general over any quantization type supported by the ``ggml_moe_a8_vec``
kernel (every format advertised by ``MOE_VEC_TYPES``). Q4_0
is currently the only type the rest of the pipeline plumbs through; support for other
types is added by parametrizing the quant type at the MoE bank loader, dequant.py, and
moe/expert_banks.py level.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_0, MOE_VEC_TYPES

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8 (or other quant format)
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8 (or other quant format)
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int,
    down_quant_type: int | None = None,
) -> torch.Tensor:
    """Fused GGUF MoE expert compute over any MMVQ-supported quantization type.

    This kernel operates directly on packed quantized weights (no materialization to bf16);
    dequantization happens inside the ``ggml_moe_a8_vec`` CUDA kernel. ``quant_type`` must be
    in ``MOE_VEC_TYPES``, which mirrors the supported types in ``ggml_moe_a8_vec``
    (gguf_kernel.cu:559).

    ``quant_type`` is the gate_up bank's type; ``down_quant_type`` defaults to it. They may
    differ because gate_up and down are separate banks with separate slot pools, and
    llama.cpp routinely quantizes the down projection differently from gate/up. Mixed
    layers share a max-stride pool; the caller supplies this layer's concrete types.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    if down_quant_type is None:
        down_quant_type = quant_type
    for label, qt in (("gate_up", quant_type), ("down", down_quant_type)):
        if qt not in MOE_VEC_TYPES:
            from freetoken.models.gguf.dequant import GGML_NAME
            raise NotImplementedError(
                f"fused GGUF MoE kernel does not support quant type "
                f"{GGML_NAME.get(qt, qt)} for the {label} bank "
                f"(only {sorted(MOE_VEC_TYPES)} supported)"
            )

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(quant_type)

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, int(down_quant_type), h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """GGUF Q4_0 MoE (backward-compat wrapper).

    This is a thin wrapper around ``fused_experts_gguf`` that hardcodes the Q4_0 type.
    All existing callers use this for now; the general function is available for future
    multi-quant pipelines.
    """
    return fused_experts_gguf(hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, int(GGML_Q4_0))


def fused_experts_gguf_separate(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,
    up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    *,
    gate_quant_type: int,
    up_quant_type: int,
    down_quant_type: int,
) -> torch.Tensor:
    """GGUF MoE over independent gate/up file-backed banks.

    Qwen3.8's original GGUF tensors cannot be joined into the conventional
    ``[E, 2I, ...]`` bank without a host-RAM copy larger than this machine.  Two
    MMVQ input GEMVs preserve the packed direct ranges and calculate the same
    SwiGLU intermediate as the combined path.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    if activation != "silu":
        raise NotImplementedError(
            f"separate GGUF gate/up only implements Qwen's silu activation, got {activation!r}"
        )
    for label, quant_type in (
        ("gate", gate_quant_type),
        ("up", up_quant_type),
        ("down", down_quant_type),
    ):
        if quant_type not in MOE_VEC_TYPES:
            from freetoken.models.gguf.dequant import GGML_NAME

            raise NotImplementedError(
                f"fused GGUF MoE kernel does not support quant type "
                f"{GGML_NAME.get(quant_type, quant_type)} for the {label} bank"
            )

    tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    intermediate = gate_q.shape[1]
    if up_q.shape[1] != intermediate:
        raise ValueError(
            f"separate GGUF gate/up intermediate mismatch: {intermediate} != {up_q.shape[1]}"
        )
    hidden = down_q.shape[1]
    gate = ggml_moe_a8_vec(
        hidden_states, gate_q, topk_ids, top_k, int(gate_quant_type), intermediate, tokens
    )
    up = ggml_moe_a8_vec(
        hidden_states, up_q, topk_ids, top_k, int(up_quant_type), intermediate, tokens
    )
    inter = F.silu(gate) * up
    out = ggml_moe_a8_vec(
        inter, down_q, topk_ids, 1, int(down_quant_type), hidden, tokens * top_k
    )
    out = out.reshape(tokens, top_k, hidden) * topk_weights.reshape(tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf", "fused_experts_gguf_q4_0", "fused_experts_gguf_separate"]
