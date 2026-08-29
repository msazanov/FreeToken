"""Qwen3.8-Flash-Next runtimes.

The upstream HF/NVFP4/FP8 implementation and the GGUF/Turing compatibility
implementation intentionally use separate model classes. They share one registry
package but keep their different QSA, PLE-state, and weight-loading contracts explicit.
"""

from freetoken.models.qwen3_5_moe.weight import setup_offload_expert_banks

from .args import Qwen4ExpArgs as Qwen4ExpGGUFArgs
from .config import (
    PLE_CONV_STATE,
    PLE_NGRAM_STATE,
    Qwen4ExpArgs,
    parse_config,
    ple_slot_states,
)
from .gguf import iter_gguf_weights, parse_gguf_config
from .gguf_experts import gguf_expert_types, load_gguf_expert_sources
from .gguf_model import Qwen4ExpGGUFForCausalLM
from .model import Qwen4ExpForCausalLM
from .weight import (
    iter_weights,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
    load_ple_table,
)

__all__ = [
    "PLE_CONV_STATE",
    "PLE_NGRAM_STATE",
    "Qwen4ExpArgs",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpGGUFArgs",
    "Qwen4ExpGGUFForCausalLM",
    "gguf_expert_types",
    "iter_gguf_weights",
    "iter_weights",
    "load_gguf_expert_sources",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
    "parse_config",
    "parse_gguf_config",
    "ple_slot_states",
    "setup_offload_expert_banks",
]
