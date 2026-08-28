from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from freetoken.models.register import get_model_spec


def test_qwen4_text_model_registry_entry():
    spec = get_model_spec("Qwen4ExpForCausalLM")

    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"


def test_qwen4_text_config_builds_qsa_and_linear_groups(monkeypatch):
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

    from freetoken.distributed.info import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp import Qwen4ExpForCausalLM

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    model = Qwen4ExpForCausalLM(config)
    assert model.model.layers.op_list[1].ple is not None
    assert "model.layers.1.ple.ple_embedding.layer_multipliers" in model.state_dict()
    assert not hasattr(model, "visual")

    from dataclasses import replace

    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear
    from freetoken.models.gguf.dequant import GGML_BF16, GGML_Q8_0
    from freetoken.models.qwen4_exp import gguf

    quant_types = {
        (-1, name): GGML_Q8_0
        for name in (
            "token_embd.weight", "output.weight", "output_hc_down.weight",
            "output_hc_up.weight", "output_hc_inject.weight",
        )
    }
    for layer_id in range(config.num_layers):
        for name in (
            "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
            "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
        ):
            quant_types[(layer_id, name)] = GGML_Q8_0
    for layer_id in (0, 1, 2):
        for name in ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"):
            quant_types[(layer_id, name)] = GGML_Q8_0
    for name in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"):
        quant_types[(3, name)] = GGML_Q8_0
    for name in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
        quant_types[(3, name)] = GGML_BF16
    for name in ("ple_key.weight", "ple_value.weight"):
        quant_types[(1, name)] = GGML_Q8_0
    monkeypatch.setattr(gguf, "_scan_quant_types", lambda _path: quant_types)

    gguf_model = Qwen4ExpForCausalLM(replace(config, gguf_model_path="/fixture.gguf"))
    assert isinstance(gguf_model.model.embed_tokens, GGUFEmbedding)
    assert isinstance(gguf_model.model.layers.op_list[3].self_attn.o_proj, GGUFLinear)
    assert isinstance(gguf_model.model.layers.op_list[1].ple.key_proj, GGUFLinear)


def test_qwen4_mrope_can_supply_positions_to_shared_qwen35_projection():
    from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention

    assert "positions" in inspect.signature(Qwen3_5Attention._project).parameters


def test_batch_exposes_optional_mrope_positions_for_text_only_qwen4():
    from freetoken.core import Batch

    field = Batch.__dataclass_fields__["rope_positions"]
    assert field.default is None


def test_qwen4_text_decoder_class_is_importable_without_vision():
    from freetoken.models.qwen4_exp import Qwen4ExpForCausalLM

    assert Qwen4ExpForCausalLM.__name__ == "Qwen4ExpForCausalLM"


def test_qwen4_ple_hash_uses_uint64_overflow_and_excludes_current_eos_from_reset():
    from freetoken.models.qwen4_exp.model import build_ngram_ids

    eos = 99
    tokens = torch.tensor([5, eos, 7, 8])
    multipliers = torch.tensor([(1 << 45) - 5, (1 << 45) + 7, (1 << 45) + 21])
    vocab_sizes = torch.tensor([97, 89])
    offsets = torch.tensor([0, 97])

    actual = build_ngram_ids(
        tokens,
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=eos,
        multipliers=multipliers,
        vocab_sizes=vocab_sizes,
        offsets=offsets,
    )

    mask = (1 << 64) - 1
    expected = []
    for index, token in enumerate(tokens.tolist()):
        ctx = [token]
        for shift in (1, 2):
            source = index - shift
            reset = source < 0 or any(tokens[j].item() == eos for j in range(source, index))
            ctx.append(eos if reset else tokens[source].item())
        rows = []
        for ngram, (size, offset) in enumerate(zip(vocab_sizes.tolist(), offsets.tolist()), start=2):
            mixed = (ctx[0] * multipliers[0].item()) & mask
            for position in range(1, ngram):
                mixed ^= (ctx[position] * multipliers[position].item()) & mask
            rows.append((mixed % size) + offset)
        expected.append(rows)

    assert actual.tolist() == expected
