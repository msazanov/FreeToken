"""The fused multi-bank ``copy_missing`` path must move exactly the same bytes as the
legacy per-bank ``fast_index_copy_jit`` loop, for every miss count (including the
zero-copy case), across banks of differing per-row sizes.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# mxfp4_triton 6-bank schema with mixed 16B-aligned per-row sizes (bytes), >=256 so the
# legacy per-bank kernel's vectorized template is valid. Sizes not covered by a model in
# kernel/aot_models.py must be listed in its TEST_FEATURE_SIZES so the per-bank kernels
# stay prebuilt under FREETOKEN_DISABLE_JIT=1.
FEATS = [8192, 512, 256, 4096, 512, 256]


def _pack_q4_0(slots, rows, cols):
    blocks = cols // 32
    nibbles = torch.randint(0, 256, (slots, rows, blocks, 16), dtype=torch.uint8)
    scales = (0.02 + 0.03 * torch.rand(slots, rows, blocks)).to(torch.float16)
    scale_bytes = scales.view(torch.uint8).reshape(slots, rows, blocks, 2)
    return torch.cat([scale_bytes, nibbles], dim=-1).reshape(slots, rows, blocks * 18)


def _build_cache(num_layers, num_experts, cache_size):
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=dev, cache_policy="lru", prefill_overlap=False, quant_format="mxfp4_triton",
    )
    schema = _BANK_SCHEMAS["mxfp4_triton"]
    # Views into one flat tensor: only per-layer addressing matters here, not
    # independent allocations.
    sources = {
        name: list(torch.randint(0, 256, (num_layers * num_experts, feat), dtype=torch.uint8, device=dev)
                   .split(num_experts))
        for name, feat in zip(schema, FEATS)
    }
    cache.set_bank_sources(sources)  # also builds the fused-copy descriptor
    return cache


@CUDA
def test_mixed_gguf_copy_plan_separates_source_width_from_slot_stride():
    """A compact Q4_K source must not inherit its Q6_K destination stride."""
    num_layers, num_experts = 2, 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cuda"),
        quant_format="gguf",
    )
    sources = {
        "gate_up": [
            torch.zeros(num_experts, 512, 144, dtype=torch.uint8, device="cuda")
            for _ in range(num_layers)
        ],
        "down": [
            torch.zeros(num_experts, 256, 210, dtype=torch.uint8, device="cuda"),
            torch.zeros(num_experts, 256, 144, dtype=torch.uint8, device="cuda"),
        ],
    }

    cache.set_bank_sources(sources)

    assert [t.tolist() for t in cache._copy_feat_bytes] == [
        [512 * 144, 256 * 210],
        [512 * 144, 256 * 144],
    ]
    assert cache._copy_dst_stride_bytes.tolist() == [512 * 144, 256 * 210]

    sources["gate_up"][1][2].fill_(55)
    sources["down"][1][2].fill_(77)
    for bank in cache.bank_caches.values():
        bank.zero_()
    cache._pending_src_layer = 1
    cache.evict_slots[0] = 3
    cache.src_indices[0] = 2
    cache.num_indices.fill_(1)

    cache.copy_missing()
    torch.cuda.synchronize()

    assert torch.all(cache.bank_caches["gate_up"][3] == 55)
    copied_down = cache.bank_caches["down"][3].flatten()
    native_bytes = 256 * 144
    assert torch.all(copied_down[:native_bytes] == 77)
    assert torch.all(copied_down[native_bytes:] == 0)


@CUDA
def test_gguf_moe_kernel_uses_tensor_slot_stride():
    """MMVQ finds experts correctly when every slot has an expert-level padded tail."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q4_0

    torch.manual_seed(4)
    slots, rows, cols = 4, 64, 256
    compact = _pack_q4_0(slots, rows, cols).cuda()
    padded = torch.zeros(slots, rows, compact.shape[-1] + 16, dtype=torch.uint8, device="cuda")
    compact_flat = compact.reshape(slots, -1)
    padded_flat = padded.reshape(slots, -1)
    padded_flat[:, : compact_flat.shape[1]].copy_(compact_flat)
    x = torch.randn(2, cols, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32, device="cuda")

    expected = ggml_moe_a8_vec(x, compact, ids, 2, GGML_Q4_0, rows, 2)
    actual = ggml_moe_a8_vec(x, padded, ids, 2, GGML_Q4_0, rows, 2)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("num_indices", [0, 1, 4, 8])
def test_fused_copy_matches_per_bank(num_indices):
    num_layers, num_experts, cache_size = 8, 8, 32
    layer_id = 3  # exercise a non-zero per-layer source selection, not just layer 0
    cache = _build_cache(num_layers, num_experts, cache_size)
    assert cache._copy_fused_ok, "fused copy should activate for 16B-aligned banks"
    # copy_missing resolves the per-layer source through this (normally set by
    # ensure_experts/materialize_layer); poked directly here since this test drives
    # evict_slots/src_indices/num_indices by hand.
    cache._pending_src_layer = layer_id

    cache.num_indices.fill_(num_indices)
    if num_indices:
        dev = torch.device("cuda")
        cache.evict_slots[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % cache_size
        # src_indices are layer-local expert rows (0..num_experts) under the new contract.
        cache.src_indices[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % num_experts

    # reference: legacy per-bank loop
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = False
    cache.copy_missing()
    torch.cuda.synchronize()
    ref = [c.clone() for _, c in cache.banks]

    # fused multi-bank launch
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = True
    cache.copy_missing()
    torch.cuda.synchronize()

    for b, (r, (_, c)) in enumerate(zip(ref, cache.banks)):
        assert torch.equal(r, c), f"bank {b} (feat={FEATS[b]}) fused != per-bank at num_indices={num_indices}"
