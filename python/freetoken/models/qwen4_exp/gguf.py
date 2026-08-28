"""GGUF metadata adapter for the text path of Qwen3.8 Flash Next."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

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


_EXPERT_SUFFIXES = frozenset({
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
})


def _ungroup_v(
    tensor: torch.Tensor,
    dim: int,
    num_key_heads: int,
    num_v_per_key: int,
    head_dim: int,
) -> torch.Tensor:
    """Inverse llama.cpp's grouped-to-tiled GDN V-head permutation."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    view = shape[:dim] + [num_v_per_key, num_key_heads, head_dim] + shape[dim + 1 :]
    result = tensor.reshape(*view)
    permutation = list(range(len(view)))
    permutation[dim], permutation[dim + 1] = permutation[dim + 1], permutation[dim]
    return result.permute(*permutation).contiguous().reshape(*shape)


def _ungroup_packed_rows(
    packed: torch.Tensor, num_key_heads: int, num_v_per_key: int, head_dim: int
) -> torch.Tensor:
    """Undo tiled V heads while keeping each GGUF quantization row intact."""
    return _ungroup_v(packed, 0, num_key_heads, num_v_per_key, head_dim)


def _to_bf16(tensor) -> torch.Tensor:
    """Materialize a small F32/F16/BF16 GGUF tensor as BF16."""
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(tensor.packed().reshape(-1), tensor.ggml_type, torch.bfloat16).reshape(
        tensor.shape
    )


def _to_f32(tensor) -> torch.Tensor:
    """Materialize a small F32/F16/BF16 GGUF tensor as F32."""
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(tensor.packed().reshape(-1), tensor.ggml_type, torch.float32).reshape(
        tensor.shape
    )


def _dequant_any(tensor) -> torch.Tensor:
    """Materialize the exceptional packed matrix whose columns need reordering."""
    from freetoken.models.gguf.dequant import GGML_NAME, GGML_UNQUANTIZED, row_bytes

    if tensor.ggml_type in GGML_UNQUANTIZED:
        return _to_bf16(tensor)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{tensor.name}: CUDA is required to dequantize "
            f"{GGML_NAME.get(tensor.ggml_type, tensor.ggml_type)} before GDN output reordering"
        )
    from freetoken.kernel.gguf import ggml_dequantize

    out_features, in_features = tensor.shape
    packed = tensor.packed().reshape(
        out_features, row_bytes(in_features, tensor.ggml_type)
    ).cuda()
    return ggml_dequantize(
        packed, tensor.ggml_type, out_features, in_features, torch.bfloat16
    ).cpu()


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"Qwen4Exp GGUF {what} supports TP=1 only; packed GGUF operators and "
            "expert banks are not tensor-parallel sharded"
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield Qwen4Exp GGUF weights, keeping every safe projection packed.

    The source format uses tiled V heads for GDN.  FreeToken's GDN uses the HF
    grouped order, hence all V-indexed rows are inverted together.  ``ssm_out``
    is the only column-wise permutation and is deliberately materialized once.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    assert not include_moe_experts, (
        "Qwen4Exp GGUF routes experts through a native offload bank; the ordinary "
        "weight iterator must not materialize them."
    )
    assert include_non_moe
    _require_tp1("weight loading")
    config = parse_gguf_config(cached_load_hf_config(model_path))
    linear = config.linear_attention_group()
    assert linear is not None
    key_heads = linear.num_key_heads
    value_heads = linear.num_value_heads
    value_per_key = value_heads // key_heads
    value_dim = linear.value_head_dim
    qk_rows = 2 * key_heads * linear.key_head_dim
    untile = key_heads != value_heads
    qsa_layers = {layer for layer in range(config.num_layers) if not config.is_linear_layer(layer)}
    quant = _scan_quant_types(model_path)
    gdn_parts: dict[int, dict[str, torch.Tensor]] = {}
    qsa_parts: dict[int, dict[str, torch.Tensor]] = {}
    index_parts: dict[int, dict[str, torch.Tensor]] = {}
    shared_parts: dict[int, dict[str, torch.Tensor]] = {}

    def emit_fused(
        base: str,
        target: str,
        layer_id: int,
        slots: dict[str, torch.Tensor],
        order: tuple[str, ...],
        sources: tuple[str, ...],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        types = [quant[(layer_id, source)] for source in sources]
        if len(set(types)) == 1:
            yield f"{base}.{target}.qweight", torch.cat([slots[key] for key in order], dim=0)
        else:
            for index, key in enumerate(order):
                yield f"{base}.{target}.qweight_{index}", slots[key]

    # PLE constants live in GGUF metadata, not the tensor table.  They are model
    # buffers nevertheless, so load them before the mmap provider is initialized.
    for layer_id in config.qwen4_args.ple_layer_ids:
        base = f"model.layers.{layer_id}.ple.ple_embedding"
        yield f"{base}.layer_multipliers", torch.tensor(
            config.qwen4_args.ple_layer_multipliers, dtype=torch.long
        )
        yield f"{base}.ngram_heads_vocab_sizes", torch.tensor(
            config.qwen4_args.ple_head_vocab_sizes, dtype=torch.long
        )
        yield f"{base}.ngram_heads_offsets", torch.tensor(
            config.qwen4_args.ple_head_offsets, dtype=torch.long
        )

    for tensor in iter_gguf_tensors(model_path):
        name = tensor.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", tensor.packed()
            continue
        if name == "output.weight":
            if not config.tie_word_embeddings:
                yield "lm_head.qweight", tensor.packed()
            continue
        if name == "output_hc_norm.weight":
            yield "model.hyper_connection_mixer.hc_norm.weight", _to_bf16(tensor) - 1
            continue
        if name in ("output_hc_down.weight", "output_hc_up.weight"):
            target = "input_mix_weight_down" if name.endswith("down.weight") else "input_mix_weight_up"
            yield f"model.hyper_connection_mixer.{target}.qweight", tensor.packed()
            continue
        if not name.startswith("blk."):
            continue
        layer_id = int(name.split(".")[1])
        if layer_id >= config.num_layers:
            continue
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer_id}"
        if suffix in _EXPERT_SUFFIXES:
            continue

        hc_packed_targets = {
            "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down",
            "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up",
            "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight",
            "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down",
            "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up",
            "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight",
        }
        if suffix in hc_packed_targets:
            yield f"{base}.{hc_packed_targets[suffix]}.qweight", tensor.packed()
            continue
        if suffix in ("hc_attn_norm.weight", "hc_ffn_norm.weight"):
            target = "attn_hyper_connection" if suffix.startswith("hc_attn") else "mlp_hyper_connection"
            yield f"{base}.{target}.hc_norm.weight", _to_bf16(tensor) - 1
            continue
        if suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _to_bf16(tensor)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(tensor).reshape(1, -1)
            continue
        if suffix == "ssm_norm.weight":
            yield f"{base}.linear_attn.norm.weight", _to_bf16(tensor)
            continue
        if suffix == "ssm_a":
            value = _to_f32(tensor)
            if untile:
                value = _ungroup_v(value, 0, key_heads, value_per_key, 1)
            if not bool((value < 0).all()):
                raise ValueError(f"{name}: expected llama.cpp's pre-transformed negative A")
            yield f"{base}.linear_attn.A_log", torch.log(-value)
            continue
        if suffix == "ssm_dt.bias":
            value = _to_f32(tensor)
            if untile:
                value = _ungroup_v(value, 0, key_heads, value_per_key, 1)
            yield f"{base}.linear_attn.dt_bias", value
            continue
        if suffix == "ssm_conv1d.weight":
            value = _to_bf16(tensor).reshape(qk_rows + value_heads * value_dim, linear.conv_kernel_dim)
            if untile:
                value = torch.cat(
                    [value[:qk_rows], _ungroup_v(value[qk_rows:], 0, key_heads, value_per_key, value_dim)],
                    dim=0,
                )
            yield f"{base}.linear_attn.conv1d.weight", value.unsqueeze(1)
            continue
        if suffix in ("ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight"):
            # Names already include the ``norm_`` prefix.  Prepending it again
            # produced e.g. ``norm_norm_key`` and left the real PLE parameters
            # absent from the state dict at load time.
            target = suffix.removeprefix("ple_").removesuffix(".weight")
            yield f"{base}.ple.{target}.weight", _to_bf16(tensor) - 1
            continue
        if suffix == "ple_conv1d.weight":
            yield f"{base}.ple.conv1d.weight", _to_bf16(tensor).reshape(
                config.hidden_size * config.qwen4_args.hc_count,
                1,
                config.qwen4_args.ple_conv_kernel_size,
            )
            continue
        if suffix in ("ple_key.weight", "ple_value.weight"):
            target = "key_proj" if suffix == "ple_key.weight" else "value_proj"
            yield f"{base}.ple.{target}.qweight", tensor.packed()
            continue
        if suffix in ("attn_q_norm.weight", "attn_k_norm.weight"):
            target = "q_norm" if suffix.startswith("attn_q") else "k_norm"
            yield f"{base}.self_attn.{target}.weight", _to_bf16(tensor) - 1
            continue
        if suffix in ("indexer.q_norm.weight", "indexer.k_norm.weight"):
            target = "q_layernorm" if suffix.startswith("indexer.q") else "k_layernorm"
            yield f"{base}.self_attn.indexer.{target}.weight", _to_bf16(tensor) - 1
            continue
        if suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.qweight", tensor.packed()
            continue
        if suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            parts = shared_parts.setdefault(layer_id, {})
            parts["gate" if suffix.startswith("ffn_gate") else "up"] = tensor.packed()
            if len(parts) == 2:
                yield from emit_fused(
                    base, "mlp.shared_expert.gate_up_proj", layer_id, parts,
                    ("gate", "up"), ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
                )
                del shared_parts[layer_id]
            continue

        if layer_id in qsa_layers:
            if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
                parts = qsa_parts.setdefault(layer_id, {})
                parts[suffix[5]] = tensor.packed()
                if len(parts) == 3:
                    yield from emit_fused(
                        base, "self_attn.qkv_proj", layer_id, parts, ("q", "k", "v"),
                        ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
                    )
                    del qsa_parts[layer_id]
                continue
            if suffix == "attn_output.weight":
                yield f"{base}.self_attn.o_proj.qweight", tensor.packed()
                continue
            if suffix in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
                parts = index_parts.setdefault(layer_id, {})
                parts["q" if suffix.startswith("indexer.q") else "k"] = tensor.packed()
                if len(parts) == 2:
                    yield from emit_fused(
                        base, "self_attn.indexer.index_qk_proj", layer_id, parts, ("q", "k"),
                        ("indexer.q_proj.weight", "indexer.k_proj.weight"),
                    )
                    del index_parts[layer_id]
                continue
        else:
            if suffix in ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"):
                key = {
                    "attn_qkv.weight": "qkv", "attn_gate.weight": "gate",
                    "ssm_beta.weight": "beta", "ssm_alpha.weight": "alpha",
                }[suffix]
                value = tensor.packed()
                if untile:
                    if key == "qkv":
                        value = torch.cat([value[:qk_rows], _ungroup_packed_rows(value[qk_rows:], key_heads, value_per_key, value_dim)], dim=0)
                    elif key == "gate":
                        value = _ungroup_packed_rows(value, key_heads, value_per_key, value_dim)
                    else:
                        value = _ungroup_packed_rows(value, key_heads, value_per_key, 1)
                parts = gdn_parts.setdefault(layer_id, {})
                parts[key] = value
                if len(parts) == 4:
                    yield from emit_fused(
                        base, "linear_attn.in_proj", layer_id, parts,
                        ("qkv", "gate", "beta", "alpha"),
                        ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"),
                    )
                    del gdn_parts[layer_id]
                continue
            if suffix == "ssm_out.weight":
                value = _dequant_any(tensor)
                if untile:
                    value = _ungroup_v(value, 1, key_heads, value_per_key, value_dim)
                yield f"{base}.linear_attn.out_proj.weight", value
                continue

    assert not gdn_parts, f"incomplete Qwen4 GDN projections: {sorted(gdn_parts)}"
    assert not qsa_parts, f"incomplete Qwen4 QSA projections: {sorted(qsa_parts)}"
    assert not index_parts, f"incomplete Qwen4 indexer projections: {sorted(index_parts)}"
    assert not shared_parts, f"incomplete Qwen4 shared-expert projections: {sorted(shared_parts)}"


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


__all__ = [
    "convert_qwen4_to_gguf",
    "is_gguf_model",
    "iter_gguf_weights",
    "parse_gguf_config",
]
