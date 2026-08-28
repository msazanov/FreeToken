from __future__ import annotations

import torch

from freetoken.models.gguf.config import GgufConfigShim


def _real_qwen4exp_shim() -> GgufConfigShim:
    """Metadata surface verified from AtomicChat Q4_K_M shard 1."""
    return GgufConfigShim(
        architectures=["Qwen4ExpGGUFForCausalLM"],
        model_path="/models/Qwen3.8-Flash-Next-Q4_K_M.gguf",
        model_type="qwen4exp",
        vocab_size=248320,
        tie_word_embeddings=False,
        metadata={
            "qwen4exp.block_count": 48,
            "qwen4exp.context_length": 262144,
            "qwen4exp.embedding_length": 2560,
            "qwen4exp.embedding_length_per_layer_input": 160,
            "qwen4exp.attention.head_count": 24,
            "qwen4exp.attention.head_count_kv": 2,
            "qwen4exp.attention.key_length": 256,
            "qwen4exp.attention.value_length": 256,
            "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen4exp.attention.compress_ratios": [0, 0, 0, 4] * 12,
            "qwen4exp.attention.indexer.head_count": 4,
            "qwen4exp.attention.indexer.key_length": 128,
            "qwen4exp.attention.indexer.top_k": 2048,
            "qwen4exp.expert_count": 512,
            "qwen4exp.expert_used_count": 10,
            "qwen4exp.expert_feed_forward_length": 640,
            "qwen4exp.expert_shared_feed_forward_length": 640,
            "qwen4exp.hyper_connection.count": 4,
            "qwen4exp.hyper_connection.low_rank": 320,
            "qwen4exp.ple.layers": [1],
            "qwen4exp.ple.conv_kernel": 4,
            "qwen4exp.ple.eos_token_id": 248044,
            "qwen4exp.ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
            "qwen4exp.ple.head_vocab_sizes": [20000003 + i for i in range(16)],
            "qwen4exp.ple.head_offsets": [i * 20000003 for i in range(16)],
            "qwen4exp.ple.heads_per_ngram": 8,
            "qwen4exp.ple.ngram_size": 3,
            "qwen4exp.rope.dimension_count": 64,
            "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
            "qwen4exp.rope.freq_base": 10_000_000.0,
            "qwen4exp.ssm.conv_kernel": 4,
            "qwen4exp.ssm.group_count": 16,
            "qwen4exp.ssm.inner_size": 6144,
            "qwen4exp.ssm.state_size": 128,
            "qwen4exp.ssm.time_step_rank": 48,
        },
    )


def test_qwen4exp_gguf_parses_actual_qsa_gdn_and_ple_geometry():
    from freetoken.models.qwen4_exp.gguf import parse_gguf_config

    config = parse_gguf_config(_real_qwen4exp_shim())

    assert config.model_type == "qwen4exp"
    assert config.num_layers == 48
    assert config.num_experts_per_tok == 10
    assert config.qwen4_args.output_gate_type == "sigmoid"
    assert config.qwen4_args.ple_layer_ids == (1,)  # GGUF ids are already zero-based.
    assert config.qwen4_args.mrope_section == (11, 11, 10)
    assert config.qwen4_args.indexer_budget == 2048
    qsa_group = next(group for group in config.attention_groups if group.name == "qsa")
    assert qsa_group.layer_ids == tuple(range(3, 48, 4))
    assert config.linear_attention_group().layer_ids == tuple(
        layer for layer in range(48) if layer % 4 != 3
    )
    assert config.requires_naive_cache and not config.supports_cuda_graph


def test_qwen4exp_gguf_is_registered_with_the_native_weight_iterator():
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
    from freetoken.models.register import get_model_spec

    registry_key = GGUF_ARCH_TO_REGISTRY["qwen4exp"]
    spec = get_model_spec(registry_key)

    assert registry_key == "Qwen4ExpGGUFForCausalLM"
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.iter_weights == "iter_gguf_weights"


def test_qwen4exp_gguf_uses_qwen2_bpe_tokenizer_converter():
    """Transformers has no ``qwen4exp`` GGUF converter key yet.

    Qwen3.8's GGUF tokenizer is GPT2-style BPE, like Qwen3.x; routing it to the
    qwen2 converter keeps the embedded tokenizer self-contained.
    """
    from freetoken.models.gguf.tokenizer import _STOP_TOKENS, _TOKENIZER_ARCH

    assert _TOKENIZER_ARCH["qwen4exp"] == "qwen2"
    assert "<|im_end|>" in _STOP_TOKENS["qwen4exp"]


def test_qwen4exp_gguf_yields_packed_ple_key_and_value_projections(monkeypatch):
    """PLE's two projections are resident GGUF operators, not part of its mmap table."""
    from freetoken.distributed.info import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp import gguf
    import freetoken.models.gguf.reader as reader
    import freetoken.utils as hf_utils

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    class _Tensor:
        def __init__(self, name):
            self.name = name

        def packed(self):
            return torch.ones(2, 3, dtype=torch.uint8)

    monkeypatch.setattr(hf_utils, "cached_load_hf_config", lambda _path: _real_qwen4exp_shim())
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: iter((
        _Tensor("blk.1.ple_key.weight"),
        _Tensor("blk.1.ple_value.weight"),
        _Tensor("blk.1.ple_norm_key.weight"),
        _Tensor("blk.1.ple_norm_query.weight"),
        _Tensor("blk.1.ple_norm_conv.weight"),
    )))
    monkeypatch.setattr(gguf, "_scan_quant_types", lambda _path: {})
    monkeypatch.setattr(gguf, "_to_bf16", lambda _tensor: torch.ones(8, dtype=torch.bfloat16))

    weights = dict(gguf.iter_gguf_weights("/fixture.gguf", "cpu", include_moe_experts=False, include_non_moe=True))

    assert "model.layers.1.ple.key_proj.qweight" in weights
    assert "model.layers.1.ple.value_proj.qweight" in weights
    assert "model.layers.1.ple.norm_key.weight" in weights
    assert "model.layers.1.ple.norm_query.weight" in weights
    assert "model.layers.1.ple.norm_conv.weight" in weights
