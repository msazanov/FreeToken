"""GGUF metadata adapter for the text path of Qwen3.8 Flash Next."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
)

from .args import Qwen4ExpArgs

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


_ARCH = "qwen4exp"


def _kv(shim: "GgufConfigShim", key: str, default: Any = None) -> Any:
    value = shim.metadata.get(f"{_ARCH}.{key}", default)
    if value is None:
        raise ValueError(f"GGUF {shim.model_path}: missing required key {_ARCH}.{key}")
    return value


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    """Build Qwen4's text-only config from llama.cpp's ``qwen4exp`` metadata."""
    num_layers = int(_kv(shim, "block_count"))
    ratios = tuple(int(value) for value in _kv(shim, "attention.compress_ratios"))
    if len(ratios) != num_layers:
        raise ValueError(
            f"GGUF {shim.model_path}: qwen4exp.attention.compress_ratios has "
            f"{len(ratios)} entries for {num_layers} layers"
        )
    qsa_ids = tuple(index for index, ratio in enumerate(ratios) if ratio > 0)
    linear_ids = tuple(index for index, ratio in enumerate(ratios) if ratio == 0)
    if not qsa_ids or not linear_ids:
        raise ValueError("Qwen4Exp requires both QSA and linear GDN layers")
    qsa_ratios = {ratios[index] for index in qsa_ids}
    if len(qsa_ratios) != 1:
        raise ValueError(f"Qwen4Exp supports one QSA compression ratio, got {sorted(qsa_ratios)}")
    compress_ratio = qsa_ratios.pop()
    if compress_ratio < 2:
        raise ValueError(f"Qwen4Exp QSA compression ratio must be >= 2, got {compress_ratio}")

    head_dim = int(_kv(shim, "attention.key_length"))
    value_dim = int(_kv(shim, "attention.value_length"))
    if head_dim != value_dim:
        raise ValueError(
            f"Qwen4Exp only supports matching Q/K and V dimensions, got {head_dim} and {value_dim}"
        )
    rotary_dim = int(_kv(shim, "rope.dimension_count"))
    sections = tuple(int(value) for value in _kv(shim, "rope.dimension_sections"))
    if len(sections) != 4 or sections[-1] != 0:
        raise ValueError(
            "Qwen4Exp GGUF MRoPE must have [temporal, height, width, 0] sections"
        )
    mrope_section = sections[:3]
    if sum(mrope_section) * 2 != rotary_dim:
        raise ValueError(
            f"Qwen4Exp MRoPE sections {sections} do not cover rotary dim {rotary_dim}"
        )
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(_kv(shim, "context_length")),
        base=float(_kv(shim, "rope.freq_base")),
        scaling=None,
    )

    indexer_budget = int(_kv(shim, "attention.indexer.top_k"))
    if indexer_budget % compress_ratio:
        raise ValueError("Qwen4Exp QSA indexer top-k must divide by its compression ratio")
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=int(_kv(shim, "ssm.group_count")),
            num_value_heads=int(_kv(shim, "ssm.time_step_rank")),
            key_head_dim=int(_kv(shim, "ssm.state_size")),
            value_head_dim=int(_kv(shim, "ssm.state_size")),
            conv_kernel_dim=int(_kv(shim, "ssm.conv_kernel")),
            output_gate=True,
        ),
        QSAAttentionGroupConfig(
            name="qsa",
            layer_ids=qsa_ids,
            num_kv_heads=int(_kv(shim, "attention.head_count_kv")),
            head_dim=head_dim,
            rotary_config=rotary,
            index_num_heads=int(_kv(shim, "attention.indexer.head_count")),
            index_num_kv_heads=1,
            index_head_dim=int(_kv(shim, "attention.indexer.key_length")),
            index_token_budget=indexer_budget,
            index_compress_ratio=compress_ratio,
        ),
    )
    ple_vocab_sizes = tuple(int(value) for value in _kv(shim, "ple.head_vocab_sizes"))
    ple_offsets = tuple(int(value) for value in _kv(shim, "ple.head_offsets"))
    args = Qwen4ExpArgs(
        hc_count=int(_kv(shim, "hyper_connection.count")),
        hc_lowrank=int(_kv(shim, "hyper_connection.low_rank")),
        # llama.cpp writes actual decoder indexes; unlike HF config, no -1 offset applies.
        ple_layer_ids=tuple(int(value) for value in _kv(shim, "ple.layers")),
        ple_embed_dim=int(_kv(shim, "embedding_length")),
        ple_conv_kernel_size=int(_kv(shim, "ple.conv_kernel")),
        ngram_size=int(_kv(shim, "ple.ngram_size")),
        heads_per_ngram=int(_kv(shim, "ple.heads_per_ngram")),
        ngram_vocab_size_base=0,
        split_ngram_parts=1,
        eos_token_id=int(_kv(shim, "ple.eos_token_id")),
        indexer_n_heads=int(_kv(shim, "attention.indexer.head_count")),
        indexer_kv_heads=1,
        indexer_head_dim=int(_kv(shim, "attention.indexer.key_length")),
        indexer_budget=indexer_budget,
        indexer_compress_ratio=compress_ratio,
        output_gate_type="silu",
        mrope_section=mrope_section,
        mrope_interleaved=True,
        ple_layer_multipliers=tuple(int(value) for value in _kv(shim, "ple.layer_multipliers")),
        ple_head_vocab_sizes=ple_vocab_sizes,
        ple_head_offsets=ple_offsets,
    )
    if len(ple_vocab_sizes) != (args.ngram_size - 1) * args.heads_per_ngram:
        raise ValueError("Qwen4Exp PLE vocabulary geometry does not match n-gram heads")
    if len(ple_offsets) != len(ple_vocab_sizes):
        raise ValueError("Qwen4Exp PLE offsets do not match vocabulary geometry")

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=int(_kv(shim, "attention.head_count")),
        num_kv_heads=int(_kv(shim, "attention.head_count_kv")),
        head_dim=head_dim,
        hidden_size=int(_kv(shim, "embedding_length")),
        vocab_size=shim.vocab_size,
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(_kv(shim, "attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=shim.tie_word_embeddings,
        rotary_config=rotary,
        num_experts=int(_kv(shim, "expert_count")),
        num_experts_per_tok=int(_kv(shim, "expert_used_count")),
        moe_intermediate_size=int(_kv(shim, "expert_feed_forward_length")),
        shared_expert_intermediate_size=int(_kv(shim, "expert_shared_feed_forward_length")),
        norm_topk_prob=False,
        model_type=shim.model_type,
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant="gguf",
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
        gguf_model_path=shim.model_path,
        use_qk_norm=True,
        vision_config=None,
        image_token_id=None,
        attention_groups=groups,
        qwen4_args=args,
        requires_naive_cache=True,
        supports_cuda_graph=False,
    )


def _scan_quant_types(model_path: str) -> dict[tuple[int, str], int]:
    """Return GGML types keyed by ``(layer, suffix)``; globals use layer ``-1``."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    result: dict[tuple[int, str], int] = {}
    for tensor in iter_gguf_tensors(model_path):
        if tensor.name.startswith("blk."):
            _, raw_layer, suffix = tensor.name.split(".", 2)
            result[(int(raw_layer), suffix)] = tensor.ggml_type
        else:
            result[(-1, tensor.name)] = tensor.ggml_type
    return result


def is_gguf_model(config: ModelConfig) -> bool:
    return config.gguf_model_path is not None


def convert_qwen4_to_gguf(model, config: ModelConfig, *, model_path: str) -> None:
    """Replace Qwen4 resident projections with native packed GGUF operators.

    Routed experts and the 38-GiB PLE table are deliberately excluded: they need
    their own offload/mmap providers and must never become resident tensors.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GGUFLMHead, gguf_merged_or_plain

    quant = _scan_quant_types(model_path)

    def qt(layer_id: int, suffix: str) -> int:
        try:
            return quant[(layer_id, suffix)]
        except KeyError as exc:
            prefix = suffix if layer_id < 0 else f"blk.{layer_id}.{suffix}"
            raise ValueError(f"GGUF {model_path}: missing required tensor {prefix}") from exc

    def swap_linear(owner, attr: str, quant_type: int) -> None:
        linear = getattr(owner, attr)
        setattr(
            owner,
            attr,
            GGUFLinear(
                linear.weight.shape[1], linear.weight.shape[0], quant_type,
                has_bias=linear.bias is not None,
            ),
        )

    def swap_hc(residual, layer_id: int | None, prefix: str) -> None:
        source_layer = -1 if layer_id is None else layer_id
        name = "output_hc" if layer_id is None else prefix
        swap_linear(residual, "input_mix_weight_down", qt(source_layer, f"{name}_down.weight"))
        swap_linear(residual, "input_mix_weight_up", qt(source_layer, f"{name}_up.weight"))
        if residual.block_inject_weight is not None:
            swap_linear(residual, "block_inject_weight", qt(source_layer, f"{name}_inject.weight"))

    inner = model.model
    embed = GGUFEmbedding(config.vocab_size, config.hidden_size, qt(-1, "token_embd.weight"))
    inner.embed_tokens = embed
    qkv_sizes = [
        config.num_qo_heads * config.head_dim * 2,
        config.num_kv_heads * config.head_dim,
        config.num_kv_heads * config.head_dim,
    ]
    linear = config.linear_attention_group()
    assert linear is not None
    gdn_sizes = [
        2 * linear.num_key_heads * linear.key_head_dim
        + linear.num_value_heads * linear.value_head_dim,
        linear.num_value_heads * linear.value_head_dim,
        linear.num_value_heads,
        linear.num_value_heads,
    ]
    shared_size = config.shared_expert_intermediate_size
    for layer_id, layer in enumerate(inner.layers.op_list):
        swap_hc(layer.attn_hyper_connection, layer_id, "hc_attn")
        swap_hc(layer.mlp_hyper_connection, layer_id, "hc_ffn")
        if layer._is_linear:
            layer.linear_attn.in_proj = gguf_merged_or_plain(
                config.hidden_size,
                gdn_sizes,
                [
                    qt(layer_id, "attn_qkv.weight"),
                    qt(layer_id, "attn_gate.weight"),
                    qt(layer_id, "ssm_beta.weight"),
                    qt(layer_id, "ssm_alpha.weight"),
                ],
                has_bias=False,
            )
            # ``ssm_out`` needs a column-level V-head reorder and is loaded dense.
        else:
            attn = layer.self_attn
            attn.qkv_proj = gguf_merged_or_plain(
                config.hidden_size,
                qkv_sizes,
                [
                    qt(layer_id, "attn_q.weight"),
                    qt(layer_id, "attn_k.weight"),
                    qt(layer_id, "attn_v.weight"),
                ],
                has_bias=False,
            )
            swap_linear(attn, "o_proj", qt(layer_id, "attn_output.weight"))
            indexer = attn.indexer
            indexer.index_qk_proj = gguf_merged_or_plain(
                config.hidden_size,
                [indexer.q_dim, indexer.k_dim],
                [qt(layer_id, "indexer.q_proj.weight"), qt(layer_id, "indexer.k_proj.weight")],
                has_bias=False,
            )

        mlp = layer.mlp
        mlp.shared_expert.gate_up_proj = gguf_merged_or_plain(
            config.hidden_size,
            [shared_size, shared_size],
            [qt(layer_id, "ffn_gate_shexp.weight"), qt(layer_id, "ffn_up_shexp.weight")],
            has_bias=False,
        )
        swap_linear(mlp.shared_expert, "down_proj", qt(layer_id, "ffn_down_shexp.weight"))
        if layer.ple is not None:
            ple = layer.ple
            swap_linear(ple, "key_proj", qt(layer_id, "ple_key.weight"))
            swap_linear(ple, "value_proj", qt(layer_id, "ple_value.weight"))

    swap_hc(inner.hyper_connection_mixer, None, "")
    model.lm_head = GGUFLMHead(
        config.hidden_size, config.vocab_size, qt(-1, "output.weight"), has_bias=False
    )


__all__ = ["convert_qwen4_to_gguf", "is_gguf_model", "parse_gguf_config"]
