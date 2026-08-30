"""SM75 correctness gates for packed TQ3_4S selected-expert execution."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason="needs the target SM75 CUDA device",
)


def _packed_tq3(slots: int, rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0,
        256,
        (slots, rows, columns // 32 * 16),
        dtype=torch.uint8,
        generator=generator,
    )
    blocks = packed.reshape(-1, 16)
    scale_bank = torch.tensor([0x00, 0x5F, 0x80, 0xFF], dtype=torch.uint8)
    block_index = torch.arange(blocks.shape[0]).view(-1, 1)
    lane_index = torch.arange(4).view(1, -1)
    blocks[:, :4] = scale_bank[(block_index + lane_index) % 4]
    return packed


def _dense_bank(packed: torch.Tensor, columns: int) -> torch.Tensor:
    from freetoken.models.gguf.dequant import dequant_tq3_4s

    slots, rows, _ = packed.shape
    return dequant_tq3_4s(packed.flatten(), torch.float32).reshape(slots, rows, columns)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tq3_4s_selected_experts_match_materialized_reference_and_slot_stride(dtype):
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_TQ3_4S

    slots, rows, columns, tokens, top_k = 3, 64, 256, 2, 2
    compact_cpu = _packed_tq3(slots, rows, columns, 20260901)
    compact = compact_cpu.cuda()
    generator = torch.Generator().manual_seed(20260902)
    x_cpu = (torch.randn(tokens, columns, generator=generator) * 0.5).to(dtype)
    x = x_cpu.cuda()
    ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32, device="cuda")

    actual = ggml_moe_a8_vec(
        x, compact, ids, top_k, GGML_TQ3_4S, rows, tokens
    ).float().cpu()
    dense = _dense_bank(compact_cpu, columns)
    expected = torch.stack(
        [x_cpu[token].float() @ dense[expert].t() for token in range(tokens) for expert in ids[token].cpu()]
    )

    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine.item() > 0.9999
    assert relative_l2.item() < 0.01

    # The cache may allocate every slot at a larger maximum stride than this
    # layer's native type. Put hostile bytes in each expert-level tail: output
    # must stay bit-identical if expert_stride_bytes is respected exactly.
    row_bytes = compact.shape[-1]
    padded = torch.zeros(
        slots, rows, row_bytes + 16, dtype=torch.uint8, device="cuda"
    )
    compact_flat = compact.reshape(slots, -1)
    padded_flat = padded.reshape(slots, -1)
    padded_flat[:, : compact_flat.shape[1]].copy_(compact_flat)
    for slot in range(slots):
        padded_flat[slot, compact_flat.shape[1] :].fill_(0xA0 + slot)

    padded_actual = ggml_moe_a8_vec(
        x, padded, ids, top_k, GGML_TQ3_4S, rows, tokens
    )
    torch.cuda.synchronize()
    assert torch.equal(padded_actual, actual.to(dtype).cuda())


def test_tq3_4s_full_swiglu_routing_accumulation_matches_exact_reference():
    from freetoken.models.gguf.dequant import GGML_TQ3_4S
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    slots, hidden, intermediate, tokens, top_k = 3, 64, 32, 2, 2
    gate_up_cpu = _packed_tq3(slots, 2 * intermediate, hidden, 20260903)
    down_cpu = _packed_tq3(slots, hidden, intermediate, 20260904)
    generator = torch.Generator().manual_seed(20260905)
    hidden_cpu = (torch.randn(tokens, hidden, generator=generator) * 0.25).half()
    topk_ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.65, 0.35], [0.4, 0.6]], dtype=torch.float32)

    actual = fused_experts_gguf(
        hidden_cpu.cuda(),
        gate_up_cpu.cuda(),
        down_cpu.cuda(),
        topk_weights.cuda(),
        topk_ids.cuda(),
        "silu",
        GGML_TQ3_4S,
    ).float().cpu()

    gate_up = _dense_bank(gate_up_cpu, hidden)
    down = _dense_bank(down_cpu, intermediate)
    expected_tokens = []
    for token in range(tokens):
        routed = []
        for route in range(top_k):
            expert = int(topk_ids[token, route])
            projected = hidden_cpu[token].float() @ gate_up[expert].t()
            gate, up = projected.chunk(2)
            inter = F.silu(gate) * up
            routed.append(inter @ down[expert].t() * topk_weights[token, route])
        expected_tokens.append(sum(routed))
    expected = torch.stack(expected_tokens)

    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine.item() > 0.999
    assert relative_l2.item() < 0.03


def test_tq3_4s_invalid_slot_ids_are_zeroed_without_oob_reads():
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_TQ3_4S

    slots, rows, columns = 2, 16, 32
    packed = _packed_tq3(slots, rows, columns, 20260906).cuda()
    x = torch.ones(1, columns, dtype=torch.float16, device="cuda")
    invalid_ids = torch.tensor([[-1, slots]], dtype=torch.int32, device="cuda")

    actual = ggml_moe_a8_vec(
        x, packed, invalid_ids, 2, GGML_TQ3_4S, rows, 1
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, torch.zeros_like(actual))
