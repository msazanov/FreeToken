# Ornith Q4_K_M on Turing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the official Ornith 1.5 35B-A3B Q4_K_M GGUF through FreeToken on RTX 2070 without unpacking experts or transferring cache padding.

**Architecture:** Extend PR #131's GGUF expert cache so quant types may vary by layer. Host sources stay compact, GPU slots use a maximum bank stride, copy kernels move each layer's native bytes, and MoE kernels receive explicit slot stride plus per-layer quant types.

**Tech Stack:** Python 3.12, PyTorch 2.11/CUDA 13, C++/CUDA JIT kernels, pytest, FreeToken qwen35moe GGUF adapter.

**Spec:** `docs/superpowers/specs/2026-08-26-ornith-q4km-turing-design.md`

## Global Constraints

- Runtime is FreeToken only.
- Target GPU is NVIDIA RTX 2070, compute capability SM 7.5.
- Expert weights remain in native packed Q4_K/Q6_K layouts.
- The official GGUF file is used unchanged.
- Existing NVFP4 service and `kv-quant` worktree remain untouched until benchmark handoff.

---

### Task 1: Mixed-bank model metadata

**Files:**
- Modify: `python/freetoken/models/config.py`
- Modify: `python/freetoken/models/qwen3_5_moe/gguf.py`
- Modify: `python/freetoken/models/qwen3_5_moe/gguf_experts.py`
- Test: `tests/models/test_qwen35moe_gguf.py`

**Interfaces:**
- Consumes: `gguf_expert_types(model_path, num_layers) -> dict[str, list[int]]`
- Produces: per-layer `ModelConfig.gguf_expert_types` for `gate_up` and `down`

- [ ] Add a test whose down types are `[GGML_Q6_K, GGML_Q4_K]` and assert metadata preserves both layers instead of raising.
- [ ] Run the test and verify the current uniform-bank assertion fails.
- [ ] Store immutable per-layer type tuples in `ModelConfig` and remove only the GPU-offload uniformity rejection.
- [ ] Run the focused model tests and commit.

### Task 2: Variable source bytes with fixed GPU slot stride

**Files:**
- Modify: `python/freetoken/moe/offload_cache.py`
- Modify: `python/freetoken/engine/cache_budget.py`
- Modify: `python/freetoken/kernel/fast_index_copy.py`
- Modify: `python/freetoken/kernel/csrc/fast_index_copy.cuh`
- Test: `tests/moe/test_offload_cache.py`
- Test: `tests/kernels/test_fast_index_copy.py`

**Interfaces:**
- Produces: per-layer copy feature sizes and fixed per-bank destination strides
- Preserves: existing `fast_index_copy_multi_jit` callers when source and destination strides match

- [ ] Add a CPU geometry test with two contiguous source layers of unequal trailing byte width.
- [ ] Verify `set_bank_sources` currently rejects the unequal shapes.
- [ ] Allocate the cache using the maximum bytes per expert and record native bytes per layer.
- [ ] Add a CUDA copy test proving a smaller source reaches the requested padded slot while adjacent slots remain unchanged.
- [ ] Extend the fused copy kernel with destination stride bytes and per-layer feature bytes.
- [ ] Run uniform and mixed copy tests, then commit.

### Task 3: Explicit MoE slot stride and per-layer quant dispatch

**Files:**
- Modify: `python/freetoken/kernel/gguf.py`
- Modify: `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu`
- Modify: `python/freetoken/kernel/csrc/gguf/moe_vec.cuh`
- Modify: `python/freetoken/moe/fused_q4_0.py`
- Modify: `python/freetoken/layers/moe.py`
- Test: `tests/models/test_gguf_dispatch.py`
- Test: `tests/moe/test_mixed_gguf_experts.py`

**Interfaces:**
- `fused_experts_gguf(..., quant_type: int, down_quant_type: int, expert_stride_bytes: tuple[int, int] | None)`
- CUDA expert addressing uses `slot * expert_stride_bytes`, then existing row/block addressing.

- [ ] Add a synthetic test with a Q4_K layer and Q6_K layer sharing one padded down cache.
- [ ] Verify the current compact-stride kernel reads the wrong slot or rejects the geometry.
- [ ] Thread per-layer quant types and cache strides through the layer dispatch.
- [ ] Compare both layers against CUDA dequantization plus BF16 matmul with cosine similarity above 0.9999.
- [ ] Run the GGUF kernel suite on SM75 and commit.

### Task 4: Prefill, rebuild, and budgeting invariants

**Files:**
- Modify: `python/freetoken/moe/offload_cache.py`
- Modify: `python/freetoken/engine/cache_budget.py`
- Test: `tests/moe/test_offload_cache.py`
- Test: `tests/engine/test_cache_budget.py`

**Interfaces:**
- All prefill paths consume the same per-layer copy descriptor as decode.
- `expert_bytes_per_slot` returns allocated maximum-stride GPU bytes.

- [ ] Add failing tests for mixed-bank materialization, prefill double buffers, cache rebuild, and budget accounting.
- [ ] Make all paths copy only native layer bytes into fixed-stride destinations.
- [ ] Run offload/cache tests and commit.

### Task 5: Turing integration

**Files:**
- Merge/cherry-pick: Turing PR #24 changes not already superseded by PR #131
- Modify only conflict sites required by current upstream
- Test: relevant attention, GGUF, and launch tests

**Interfaces:**
- CUDA JIT/AOT target includes SM75.
- Ampere-only paths remain disabled or bypassed on RTX 2070.

- [ ] Apply the Turing changes and resolve conflicts one subsystem at a time.
- [ ] Build native helpers for SM75.
- [ ] Run synthetic Q4_K/Q6_K mixed-bank kernels on the RTX 2070.
- [ ] Commit the verified compatibility layer.

### Task 6: Official checkpoint serving and benchmark

**Files:**
- Use: `/home/random/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-GGUF/snapshots/fbbaed45c2f0e200276ffa51701a24d45dc7f57e/Ornith-1.5-35B-Q4_K_M.gguf`
- Create: `benchmarks/2026-08-26-ornith-35b-q4km-turing.md`

**Interfaces:**
- OpenAI-compatible FreeToken server endpoint
- Benchmark schema matches `benchmarks/2026-08-26-ornith-35b-nvfp4-baseline.md`

- [ ] Reconcile every served tensor against the model state without loading MTP block 40.
- [ ] Stop the NVFP4 service only immediately before the real Q4_K_M load.
- [ ] Start at small context and verify deterministic factual and code prompts.
- [ ] Restore INT8 KV and 122880 context after the smoke test.
- [ ] Measure cold/warm prefill, decode tokens/s, RAM, VRAM, swap, and expert-cache statistics.
- [ ] Save the comparison, push the branch to `msazanov/FreeToken`, and report remaining limitations.
