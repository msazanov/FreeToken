"""Contract tests for the dense Gemma-4 E2B GGUF path.

These tests intentionally use only GGUF metadata and meta-device module construction.  They
must remain runnable without a GPU or the 3.2 GiB checkpoint, while covering the architecture
features that distinguish E2B from the older Gemma-4 MoE parser.
"""

from __future__ import annotations

import pytest

from freetoken.models.gguf.config import GgufConfigShim


def _e2b_shim(monkeypatch: pytest.MonkeyPatch) -> GgufConfigShim:
    """Return the metadata shape emitted by google/gemma-4-E2B-it-qat-q4_0-gguf."""
    metadata = {
        "gemma4.block_count": 35,
        "gemma4.embedding_length": 1536,
        "gemma4.embedding_length_per_layer_input": 256,
        "gemma4.attention.head_count": 8,
        # E-series writes one scalar; the older MoE converter writes a per-layer list.
        "gemma4.attention.head_count_kv": 1,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.key_length_swa": 256,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.value_length_swa": 256,
        "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
        "gemma4.attention.shared_kv_layers": 20,
        "gemma4.attention.sliding_window": 512,
        "gemma4.attention.sliding_window_pattern": [
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
        ],
        "gemma4.feed_forward_length": [6144] * 15 + [12288] * 20,
        "gemma4.context_length": 131072,
        "gemma4.rope.freq_base": 1_000_000.0,
        "gemma4.rope.freq_base_swa": 10_000.0,
        "gemma4.rope.dimension_count_swa": 256,
        "gemma4.final_logit_softcapping": 30.0,
    }
    # The synthetic shim has no tensor table.  Keep the test about parser behavior rather than
    # filesystem I/O; _full_rotary_dim's metadata-only fallback is covered separately.
    monkeypatch.setattr(
        "freetoken.models.gemma4.gguf._full_rotary_dim",
        lambda shim, head_dim: head_dim // 4,
    )
    return GgufConfigShim(
        architectures=["Gemma4GGUFForCausalLM"],
        model_path="<synthetic-e2b>",
        model_type="gemma4",
        metadata=metadata,
        vocab_size=262144,
        tie_word_embeddings=True,
    )


def test_dense_e2b_metadata_does_not_require_expert_keys(monkeypatch: pytest.MonkeyPatch):
    from freetoken.models.gemma4.gguf import parse_gguf_config

    config = parse_gguf_config(_e2b_shim(monkeypatch))

    assert config.moe_enabled is False
    assert config.num_experts == 0
    assert config.per_layer_hidden_size == 256
    assert config.num_kv_shared_layers == 20
    assert config.intermediate_size == 6144
    assert config.use_double_wide_mlp is True


def test_e2b_scalar_kv_metadata_and_layer_geometry(monkeypatch: pytest.MonkeyPatch):
    from freetoken.models.gemma4.gguf import parse_gguf_config

    config = parse_gguf_config(_e2b_shim(monkeypatch))

    assert config.num_kv_heads == 1
    assert config.first_kv_shared_layer_idx() == 15
    assert config.is_kv_shared_layer(15)
    assert not config.is_kv_shared_layer(14)
    assert config.attention_group_for_layer(34).head_dim == 512
    assert config.attention_group_for_layer(34).num_kv_heads == 1


def test_shared_kv_layers_build_q_only_attention(monkeypatch: pytest.MonkeyPatch):
    from freetoken.models.gemma4.attention import Gemma4Attention
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.gemma4.gguf import parse_gguf_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = parse_gguf_config(_e2b_shim(monkeypatch))
    layer = Gemma4Attention(config, layer_id=34)

    assert hasattr(layer, "q_proj")
    assert not hasattr(layer, "qkv_proj")
    assert not hasattr(layer, "k_norm")
    assert not hasattr(layer, "v_norm")


def test_non_shared_e2b_layer_remains_fused_qkv(monkeypatch: pytest.MonkeyPatch):
    from freetoken.models.gemma4.attention import Gemma4Attention
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.gemma4.gguf import parse_gguf_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = parse_gguf_config(_e2b_shim(monkeypatch))
    layer = Gemma4Attention(config, layer_id=14)

    assert hasattr(layer, "qkv_proj")
    assert not hasattr(layer, "q_proj")
    assert hasattr(layer, "k_norm")
    assert hasattr(layer, "v_norm")
