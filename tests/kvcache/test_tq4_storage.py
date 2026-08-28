"""TQ4-NC storage invariants reused from the proven RTX 2070 Ornith runner."""

from __future__ import annotations

import torch
import pytest


def setup_module() -> None:
    from freetoken.distributed.info import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def test_tq4_reference_round_trip_uses_packed_nibbles_and_bf16_scales():
    from freetoken.kvcache.tq4 import decode_tq4, encode_tq4

    values = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    packed, scales = encode_tq4(values)
    restored = decode_tq4(packed, scales, head_dim=8)

    assert packed.dtype is torch.uint8 and packed.shape == (3, 2, 4)
    assert scales.dtype is torch.bfloat16 and scales.shape == (3, 2)
    assert torch.isfinite(restored).all()


def test_tq4_pool_keeps_logical_head_dim_but_allocates_half_width_bytes():
    from freetoken.kvcache.mha_pool import MHAKVCache

    pool = MHAKVCache(
        num_kv_heads=2, num_layers=3, head_dim=8, num_pages=5, page_size=4,
        dtype=torch.bfloat16, device=torch.device("cpu"), kv_cache_dtype="tq4-nc",
    )

    assert pool.logical_head_dim == 8
    assert pool.is_packed
    assert pool.k_cache(0).shape == (5, 4, 2, 4)
    assert pool.k_cache(0).dtype is torch.uint8
    assert pool.k_scale(0).shape == (5, 4, 2)


def test_tq4_pool_scatter_matches_reference_packing():
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.kvcache.tq4 import encode_tq4, randomized_hadamard

    pool = MHAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=8, num_pages=2, page_size=4,
        dtype=torch.bfloat16, device=torch.device("cpu"), kv_cache_dtype="tq4-nc",
    )
    k = torch.tensor([[1.0] * 16, [2.0] * 16], dtype=torch.bfloat16)
    v = torch.tensor([[3.0] * 16, [4.0] * 16], dtype=torch.bfloat16)
    rows = torch.tensor([6, 1], dtype=torch.int32)
    pool.store_kv(k, v, rows, layer_id=1)

    expected, scales = encode_tq4(
        randomized_hadamard(k.view(2, 2, 8), layer_id=1, num_kv_heads=2)
    )
    assert torch.equal(pool.k_cache(1).view(-1, 2, 4)[rows.long()], expected)
    assert torch.equal(pool.k_scale(1).view(-1, 2)[rows.long()], scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton JIT")
def test_tq4_cuda_store_compiles_without_python_global_constants():
    """The CUDA implementation must compile on Triton 3.6, not only pass its CPU oracle."""
    from freetoken.kernel.triton.kv_quant import store_tq4_nc_kv

    device = torch.device("cuda")
    k_cache = torch.empty((4, 2, 4), dtype=torch.uint8, device=device)
    v_cache = torch.empty_like(k_cache)
    k_scale = torch.empty((4, 2), dtype=torch.bfloat16, device=device)
    v_scale = torch.empty_like(k_scale)
    rows = torch.tensor([3, 1], dtype=torch.int32, device=device)
    k = torch.arange(32, dtype=torch.float16, device=device).reshape(2, 16) / 16
    v = (k + 1).contiguous()

    store_tq4_nc_kv(
        k_cache=k_cache, v_cache=v_cache, k_scale=k_scale, v_scale=v_scale,
        indices=rows, k=k, v=v, layer_id=0, head_dim=8,
        inputs_are_transformed=True,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(k_scale[rows.long()].float()).all()
    assert torch.isfinite(v_scale[rows.long()].float()).all()
