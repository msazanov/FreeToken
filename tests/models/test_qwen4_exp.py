from __future__ import annotations

import inspect
from types import SimpleNamespace

from freetoken.models.register import get_model_spec


def test_qwen4_text_model_registry_entry():
    spec = get_model_spec("Qwen4ExpForCausalLM")

    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"


def test_qwen4_text_config_builds_qsa_and_linear_groups():
    from freetoken.models.qwen4_exp.config import parse_config

    text = SimpleNamespace(
        layer_types=["linear_attention", "linear_attention", "linear_attention", "qwen_sparse_attention"],
        head_dim=128,
        rope_parameters={
            "partial_rotary_factor": 1.0,
            "rope_theta": 10_000_000,
            "mrope_interleaved": True,
            "mrope_section": [22, 21, 21],
        },
        indexer_budget=2048,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        max_position_embeddings=262_144,
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        eos_token_id=100,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        split_ngram_parts=128,
        output_gate_type="sigmoid",
        hidden_act="silu",
        num_hidden_layers=4,
        num_attention_heads=16,
        hidden_size=2048,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        norm_topk_prob=False,
        tie_word_embeddings=False,
    )
    config = parse_config(
        SimpleNamespace(
            text_config=text,
            model_type="qwen4_exp",
            architectures=["Qwen4ExpForCausalLM"],
        )
    )

    assert config.is_linear_layer(0)
    assert config.attn_type_for_layer(3).value == "qsa"
    assert config.num_experts_per_tok == 10
    assert config.qwen4_args.ple_layer_ids == (1,)
    assert config.requires_naive_cache
    assert not config.supports_cuda_graph


def test_qwen4_mrope_can_supply_positions_to_shared_qwen35_projection():
    from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention

    assert "positions" in inspect.signature(Qwen3_5Attention._project).parameters
