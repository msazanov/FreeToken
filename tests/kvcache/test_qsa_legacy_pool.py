"""CPU-visible invariants for Qwen4 compressed sparse-attention cache storage."""

from __future__ import annotations

import torch


def setup_module() -> None:
    """The pool unit tests don't initialize distributed runtime state."""
    from freetoken.distributed.info import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _pool(*, kv_cache_dtype: str = "bf16"):
    from freetoken.kvcache.qsa_legacy_pool import QSAKVCache

    return QSAKVCache(
        num_kv_heads=2,
        num_layers=4,
        head_dim=8,
        num_pages=3,
        page_size=8,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_num_kv_heads=1,
        index_head_dim=4,
        compress_ratio=4,
        layer_ids=(1, 3),
        kv_cache_dtype=kv_cache_dtype,
    )


def test_qsa_pool_stores_compressed_index_keys_in_compute_dtype_with_int8_kv():
    """KV quantization saves full K/V only; QSA's scoring index remains BF16."""
    pool = _pool(kv_cache_dtype="int8")

    assert pool.k_cache(1).dtype is torch.int8
    assert pool.compressed_k_cache(1).dtype is torch.bfloat16
    assert pool.compressed_k_cache(1).shape == (3 * 2, 1, 4)

    keys = torch.arange(8, dtype=torch.bfloat16).view(2, 1, 4)
    pool.store_compressed_k(keys, torch.tensor([0, 5]), layer_id=1)
    assert torch.equal(pool.compressed_k_cache(1)[[0, 5]], keys)


def test_qsa_pool_rebuild_preserves_mapping_and_resizes_both_cache_tiers():
    pool = _pool()
    pool.rebuild(5)

    assert pool.k_cache(3).shape == (5, 8, 2, 8)
    assert pool.compressed_k_cache(3).shape == (5 * 2, 1, 4)
    assert pool.unit_bytes()[0] == 2 * 2 * 2 * 8 * 2 + 2 * 1 * 4 * 2 // 4


def test_factory_selects_qsa_pool_for_hybrid_linear_qwen4_config():
    from freetoken.kvcache import create_kvcache_pool, resolve_pool_class
    from freetoken.kvcache.qsa_legacy_pool import QSAKVCache
    from freetoken.models.config import (
        LinearGatedDeltaGroupConfig,
        ModelConfig,
        QSAAttentionGroupConfig,
        RotaryConfig,
    )

    rotary = RotaryConfig(head_dim=8, rotary_dim=8, max_position=4096, base=1e4, scaling=None)
    config = ModelConfig(
        num_layers=4, num_qo_heads=4, num_kv_heads=2, head_dim=8,
        hidden_size=64, vocab_size=64, intermediate_size=128, rms_norm_eps=1e-6,
        rotary_config=rotary, hidden_act="silu", tie_word_embeddings=False,
        num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
        norm_topk_prob=False, model_type="qwen4_exp", architectures=("Qwen4ExpForCausalLM",),
        attention_groups=(
            LinearGatedDeltaGroupConfig(
                name="linear", layer_ids=(0, 2), num_key_heads=4, num_value_heads=4,
                key_head_dim=8, value_head_dim=8, conv_kernel_dim=4, output_gate=True,
            ),
            QSAAttentionGroupConfig(
                name="qsa", layer_ids=(1, 3), num_kv_heads=2, head_dim=8,
                rotary_config=rotary, index_num_heads=4, index_num_kv_heads=1,
                index_head_dim=4, index_token_budget=8, index_compress_ratio=4,
            ),
        ),
    )

    assert resolve_pool_class(config) is QSAKVCache
    pool = create_kvcache_pool(
        config, num_pages=3, page_size=8, dtype=torch.bfloat16,
        device=torch.device("cpu"), kv_cache_dtype="int8",
    )
    assert isinstance(pool, QSAKVCache)
    assert pool.k_cache(1).dtype is torch.int8
    assert pool.compressed_k_cache(3).dtype is torch.bfloat16


def test_qsa_pool_allows_tq4_full_kv_without_quantizing_index_keys():
    pool = _pool(kv_cache_dtype="tq4-nc")

    assert pool.k_cache(1).dtype is torch.uint8
    assert pool.k_cache(1).shape[-1] == 4  # packed storage for logical head_dim=8
    assert pool.logical_head_dim == 8
    assert pool.compressed_k_cache(1).dtype is torch.bfloat16
