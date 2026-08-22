# Quantized Paged KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Serve Ornith-1.5-35B-A3B-NVFP4 through FreeToken while its paged full-attention KV cache is stored as dynamically scaled FP8 E5M2 or INT8.

**Architecture:** Keep Ornith weights as NVFP4 and retain BF16 activations, queries, and GatedDeltaNet state. Extend only the generic MHA cache with one-byte K/V slabs and separate BF16 scales per \`(layer, token, KV head)\`; quantize on store and dequantize in the existing Triton attention loads. The BF16 default must retain its current tensors and kernel specialization.

**Tech Stack:** Python 3.12, PyTorch 2.11 CUDA 13, Triton 3.6, pytest, NVIDIA Turing SM 7.5.

**Spec:** User objective in the current thread: quantized FP8/INT8 KV cache, Ornith 35B NVFP4 weights, FreeToken runner.

## Global Constraints

- Preserve the safetensors NVFP4 checkpoint; do not switch to GGUF or requantize weights.
- Expose exactly \`bf16\` (default), \`fp8-e5m2\`, and \`int8\`. Do not expose E4M3 because this Turing Triton target rejects it.
- First implementation supports only \`MHAKVCache\` and \`--attention-backend triton\`; other cache families and backends must fail before allocation.
- K and V each have independent BF16 per-vector scales. FP8 uses \`amax / 57344\`; INT8 uses \`amax / 127\`; clamp each scale to a positive epsilon.
- Storage cost includes data plus two scale bytes for each K/V vector. For Ornith's 256-dimensional head this is \`(256 + 2) / 512 = 50.39%\` of BF16 KV.
- Tests must run before production code for every behavior. Stop the 1919 server only for the final full-model run, and restore a known-good command if a candidate fails.

---

### Task 1: Public configuration and memory accounting

**Files:**
- Modify: \`python/freetoken/engine/config.py\`
- Modify: \`python/freetoken/server/args.py\`
- Modify: \`python/freetoken/kvcache/base.py\`
- Modify: \`tests/kvcache/test_pool_sizing_surface.py\`

**Interfaces:**
- Produces \`EngineConfig.kv_cache_dtype: str = "bf16"\`.
- Produces CLI option \`--kv-cache-dtype {bf16,fp8-e5m2,int8}\`.
- Produces pure helper \`kv_cache_storage_bytes_per_vector(head_dim, mode)\` used by \`spec_kv_bytes_per_token\`.

- [ ] **Step 1: Write the failing sizing test**

\`\`\`python
def test_int8_mha_cost_counts_one_byte_values_and_bf16_scales():
    config = _generic_config(kv_cache_dtype="int8")
    (spec,) = config.model_config.kv_cache_group_specs()
    assert spec_kv_bytes_per_token(spec, config) == 2 * 2 * 2 * (64 + 2)
\`\`\`

Also test that a parser-created config defaults to \`bf16\` and rejects \`fp8-e4m3fn\`.

- [ ] **Step 2: Run it red**

Run: \`PYTHONPATH=.test-deps:python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/kvcache/test_pool_sizing_surface.py -q\`

Expected: a failure caused by the absent \`kv_cache_dtype\` contract.

- [ ] **Step 3: Implement only the config and exact byte arithmetic**

Add the frozen config field and parser choice. Let tests' duck-typed configs omit the field by reading \`getattr(config, "kv_cache_dtype", "bf16")\`. For MHA data use one byte plus one BF16 scale for each head of each K or V slab; leave index and non-MHA arithmetic unchanged.

- [ ] **Step 4: Run green**

Run the command from step 2. Expected: PASS including existing BF16 parity assertions.

- [ ] **Step 5: Commit**

\`\`\`bash
git add python/freetoken/engine/config.py python/freetoken/server/args.py python/freetoken/kvcache/base.py tests/kvcache/test_pool_sizing_surface.py
git commit -m "feat: add quantized KV cache configuration"
\`\`\`

### Task 2: MHA pool allocation, rebuild, and scaled store

**Files:**
- Create: \`python/freetoken/kernel/triton/kv_quant.py\`
- Modify: \`python/freetoken/kvcache/mha_pool.py\`
- Modify: \`python/freetoken/kvcache/__init__.py\`
- Modify: \`tests/kvcache/test_kv_cache_rebuild.py\`
- Create: \`tests/kernels/test_kv_quant.py\`

**Interfaces:**
- Produces \`MHAKVCache(..., kv_cache_dtype=...)\`, \`k_scale(layer_id)\`, and \`v_scale(layer_id)\`.
- Produces a CUDA store kernel which writes quantized K/V and BF16 scales at \`out_loc\`.
- Preserves existing \`k_cache\`, \`v_cache\`, and \`store_kv\` behavior in BF16 mode.

- [ ] **Step 1: Write failing layout and round-trip tests**

\`\`\`python
def test_int8_pool_allocates_one_byte_values_and_two_scale_slabs():
    pool = MHAKVCache(2, 2, 64, 4, 1, torch.bfloat16, torch.device("cpu"),
                      kv_cache_dtype="int8")
    assert pool.k_cache(0).dtype is torch.int8
    assert pool.k_scale(0).shape == (4, 1, 2)
    assert pool.unit_bytes() == (2 * 2 * 2 * (64 + 2), 0)
\`\`\`

On CUDA, store random BF16 K/V at non-contiguous output locations, dequantize with the stored scale, and assert mean absolute error below \`0.08\` for FP8 E5M2 and \`0.03\` for INT8.

- [ ] **Step 2: Run it red**

Run: \`PYTHONPATH=.test-deps:python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/kvcache/test_kv_cache_rebuild.py tests/kernels/test_kv_quant.py -q\`

Expected: missing constructor argument or scale accessor.

- [ ] **Step 3: Implement minimum correct storage**

For BF16 keep the existing \`_kv_buffer\` allocation exactly. For quantized mode allocate K data, V data, K scales, and V scales separately; recreate all four in \`rebuild\`; use the existing JIT store only for BF16 and the new Triton store for quantized paths. Make \`unit_bytes\` report actual live data plus scales.

- [ ] **Step 4: Run green**

Run the command from step 2. Expected: CPU layout/rebuild tests and CUDA round-trip tests pass.

- [ ] **Step 5: Commit**

\`\`\`bash
git add python/freetoken/kvcache/mha_pool.py python/freetoken/kvcache/__init__.py python/freetoken/kernel/triton/kv_quant.py tests/kvcache/test_kv_cache_rebuild.py tests/kernels/test_kv_quant.py
git commit -m "feat: store MHA KV as scaled FP8 or INT8"
\`\`\`

### Task 3: Triton paged-attention dequantization

**Files:**
- Modify: \`python/freetoken/attention/triton.py\`
- Modify: \`python/freetoken/kernel/triton/attention.py\`
- Modify: \`tests/kernels/test_triton_attention.py\`

**Interfaces:**
- Extends paged decode stage 1, extend attention, and split extend attention launches with optional K/V scale pointers and a compile-time \`KV_QUANT\` mode.
- Produces attention outputs compared to a BF16 reference.

- [ ] **Step 1: Write a failing parameterized parity test**

\`\`\`python
@pytest.mark.cuda
@pytest.mark.parametrize(("mode", "atol"), [("fp8-e5m2", 8e-2), ("int8", 3e-2)])
def test_paged_attention_reads_scaled_kv(mode, atol):
    reference = run_paged_attention(q, k_bf16, v_bf16, page_table)
    actual = run_paged_attention(q, k_data, v_data, page_table, k_scale, v_scale, mode)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=atol)
\`\`\`

Cover single-token decode and multi-token prefill; run the existing split helper when the prompt crosses its split threshold.

- [ ] **Step 2: Run it red**

Run: \`PYTHONPATH=.test-deps:python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/kernels/test_triton_attention.py -q\`

Expected: failure due to absent scale launch parameters, not a collection error.

- [ ] **Step 3: Implement load-site dequantization**

Pass MHA scale views from \`TritonAttentionBackend.forward\`. At every cache K/V \`tl.load\`, multiply the one-byte value by its per-vector BF16 scale before \`tl.dot\`. Never cast Q to an FP8/INT8 type. Keep the current launch path when the mode is BF16 so its generated kernel signature and CUDA graphs do not change.

- [ ] **Step 4: Run green**

Run: \`PYTHONPATH=.test-deps:python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/kernels/test_triton_attention.py tests/kernels/test_kv_quant.py -q\`

Expected: BF16, FP8 E5M2 and INT8 attention tests pass on the RTX 2070.

- [ ] **Step 5: Commit**

\`\`\`bash
git add python/freetoken/attention/triton.py python/freetoken/kernel/triton/attention.py tests/kernels/test_triton_attention.py tests/kernels/test_kv_quant.py
git commit -m "feat: dequantize KV cache in Triton attention"
\`\`\`

### Task 4: Capability gate and full Ornith NVFP4 verification

**Files:**
- Modify: \`python/freetoken/engine/engine.py\`
- Create: \`tests/engine/test_quantized_kv_validation.py\`
- Modify: the existing \`ft serve\` documentation section in \`README.md\`
- Create: \`benchmarks/kv-quant-ornith-35b.json\`

**Interfaces:**
- Produces early errors for non-Triton backends or non-MHA cache pools paired with quantized KV.
- Produces a selected production command for the existing Ornith NVFP4 checkpoint, with measured rather than estimated results.

- [ ] **Step 1: Write failing validation tests**

\`\`\`python
def test_quantized_kv_requires_triton_backend():
    with pytest.raises(ValueError, match="kv-cache-dtype.*triton"):
        validate_quantized_kv_config(config_with("fp8-e5m2", attention_backend="trtllm"))

def test_bf16_remains_valid_for_nontriton_backend():
    validate_quantized_kv_config(config_with("bf16", attention_backend="trtllm"))
\`\`\`

- [ ] **Step 2: Run it red, implement early validation, then run green**

Run: \`PYTHONPATH=.test-deps:python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/engine/test_quantized_kv_validation.py tests/kvcache -q\`

Expected before implementation: missing validation failure. Expected after implementation: PASS.

- [ ] **Step 3: Capture the current BF16 120K baseline**

Record the exact command, \`/health\`, \`/v1/models\`, \`/v1/cache/status\`, \`nvidia-smi\`, a fixed 384-token prompt's TTFT, decode token/s, and repeat-request cached-token count.

- [ ] **Step 4: Test the actual NVFP4 model sequentially**

Stop only \`freetoken-ornith-nvfp4.service\`. Start the worktree version on port 1919 with the same model path/offload flags and \`--kv-cache-dtype fp8-e5m2 --attention-backend triton\`; verify all baseline probes. Repeat for INT8 with identical prompt and seed. Restore the prior known-good command immediately if either server cannot start or generate correctly.

- [ ] **Step 5: Probe 262144 native context only after a stable 120K winner**

Use \`--max-seq-len-override 262144 --num-tokens 262144 --kv-reserve-tokens 262144\`, record allocation result, MoE slots, cache bytes, TTFT and decode token/s. Leave the verified stable configuration running and commit the JSON measurements.

\`\`\`bash
git add python/freetoken/engine/engine.py tests/engine/test_quantized_kv_validation.py README.md benchmarks/kv-quant-ornith-35b.json
git commit -m "feat: validate quantized KV serving"
\`\`\`
