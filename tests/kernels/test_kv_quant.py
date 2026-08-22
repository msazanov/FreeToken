from __future__ import annotations

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantized KV store needs CUDA")
@pytest.mark.parametrize(
    ("mode", "mean_abs_error"),
    [("fp8-e5m2", 0.08), ("int8", 0.03)],
)
def test_quantized_mha_store_round_trips_scattered_kv_slots(mode, mean_abs_error):
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_tokens, num_kv_heads, head_dim = 3, 2, 256
    pool = MHAKVCache(
        num_kv_heads=num_kv_heads,
        num_layers=1,
        head_dim=head_dim,
        num_pages=1,
        page_size=16,
        dtype=torch.bfloat16,
        device=device,
        kv_cache_dtype=mode,
    )
    k = torch.randn(num_tokens, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out_loc = torch.tensor([11, 2, 8], dtype=torch.int32, device=device)

    pool.store_kv(k, v, out_loc, layer_id=0)
    torch.cuda.synchronize()

    def restore(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        rows = data.view(-1, num_kv_heads, head_dim)[out_loc.long()].float()
        factors = scale.view(-1, num_kv_heads)[out_loc.long()].float()
        return rows * factors.unsqueeze(-1)

    actual_k = restore(pool.k_cache(0), pool.k_scale(0))
    actual_v = restore(pool.v_cache(0), pool.v_scale(0))
    expected_k = k.view(num_tokens, num_kv_heads, head_dim).float()
    expected_v = v.view(num_tokens, num_kv_heads, head_dim).float()

    assert (actual_k - expected_k).abs().mean() < mean_abs_error
    assert (actual_v - expected_v).abs().mean() < mean_abs_error

