from __future__ import annotations

from freetoken.attention.base import AttnType
from freetoken.models.config import ModelConfig, QSAAttentionGroupConfig, RotaryConfig


def test_qwen4_qsa_group_emits_compressed_index_cache_spec():
    rotary = RotaryConfig(
        head_dim=128,
        rotary_dim=128,
        max_position=262_144,
        base=10_000_000.0,
        scaling=None,
    )
    config = ModelConfig(
        num_layers=4,
        num_qo_heads=16,
        num_kv_heads=2,
        head_dim=128,
        hidden_size=2048,
        vocab_size=100,
        intermediate_size=0,
        rms_norm_eps=1e-6,
        rotary_config=rotary,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        norm_topk_prob=False,
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForCausalLM"],
        attention_groups=(
            QSAAttentionGroupConfig(
                name="qsa",
                layer_ids=(3,),
                num_kv_heads=2,
                head_dim=128,
                rotary_config=rotary,
                index_num_heads=4,
                index_num_kv_heads=1,
                index_head_dim=128,
                index_token_budget=2048,
                index_compress_ratio=4,
            ),
        ),
        qwen4_args=object(),
        requires_naive_cache=True,
        supports_cuda_graph=False,
    )

    spec = config.kv_cache_group_specs()[0]

    assert config.attn_type_for_layer(3) is AttnType.QSA
    assert spec.attn_type is AttnType.QSA
    assert spec.index_num_kv_heads == 1
    assert spec.index_compress_ratio == 4
    assert spec.index_token_budget == 2048
    assert config.requires_naive_cache
    assert not config.supports_cuda_graph
