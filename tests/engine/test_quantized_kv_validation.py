from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.attention import AttnType
from freetoken.models.config import KVCacheGroupSpec


def _config(mode: str, backend: str, attn_type: AttnType = AttnType.FULL):
    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=(0,),
        num_kv_heads=2,
        head_dim=64,
        sliding_window=None,
        attn_type=attn_type,
    )
    model_config = SimpleNamespace(
        dsv4_args=None,
        kv_cache_group_specs=lambda: (spec,),
    )
    return SimpleNamespace(
        kv_cache_dtype=mode,
        attention_backend=backend,
        model_config=model_config,
    )


def test_quantized_kv_requires_triton_for_every_attention_phase():
    from freetoken.engine.engine import _validate_quantized_kv_config

    with pytest.raises(ValueError, match="kv-cache-dtype.*triton"):
        _validate_quantized_kv_config(_config("fp8-e5m2", "trtllm"))


def test_quantized_kv_requires_generic_mha_pool():
    from freetoken.engine.engine import _validate_quantized_kv_config

    with pytest.raises(ValueError, match="MHAKVCache"):
        _validate_quantized_kv_config(_config("int8", "triton", AttnType.BSA))


def test_bf16_remains_valid_for_other_attention_backends():
    from freetoken.engine.engine import _validate_quantized_kv_config

    _validate_quantized_kv_config(_config("bf16", "trtllm", AttnType.BSA))
