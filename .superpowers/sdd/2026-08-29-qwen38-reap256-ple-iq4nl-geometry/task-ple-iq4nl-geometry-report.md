# Task report: IQ4_NL PLE geometry invariant

Date: 2026-08-29

Implementation commit: `f98eaef` (`fix: enforce IQ4_NL PLE geometry`)

## Scope

This task closes the single Important review gap after `e2f0659`.  The
`IQ4_NL` dequantization entry point computes `ceil(k / 256)` blocks and each
CUDA block writes 256 values.  Qwen3.8's real PLE geometry is 16 n-gram heads
with 160 values per head, or 2,560 flattened values per token.  That total is
divisible by 256; an arbitrary `ple_embed_dim` is not necessarily safe.

No model, benchmark, download, subagent, cache, or CUDA-kernel implementation
was changed or run as part of this task.

## TDD evidence

### RED

Tests were written before the production change and run with:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4exp_ple.py::test_qwen4_ple_accepts_iq4_nl_table \
  tests/models/test_qwen4exp_ple.py::test_qwen4_ple_rejects_iq4_nl_non_block_aligned_embedding_dim \
  tests/models/test_qwen4exp_ple.py::test_iq4_nl_dequantizes_real_ple_geometry_on_cuda -q
```

Result: `1 failed, 1 passed, 1 skipped`.

The invalid-dimension test reached the old `row_bytes(150, IQ4_NL)` assertion
(`150 not a multiple of block 32`) instead of a loader-level `RuntimeError`.
The real-geometry acceptance test passed, and the CUDA test skipped because
CUDA was unavailable in the test environment.

### GREEN

The minimal production change imports `GGML_IQ4_NL` and, before calling
`row_bytes`, rejects an IQ4_NL PLE configuration unless
`ple_embed_dim % 256 == 0`.  The error names the value, the required divisor,
and the `ggml_dequantize` complete-block reason.

The realistic loader test now uses:

```text
heads_per_ngram = 16
ngram_heads = 16
ple_embed_dim = 2560
head_dim = 160
table shape = (16, 160)
IQ4_NL row_bytes = 90
```

The CUDA-gated test packs 16 rows × 5 IQ4_NL blocks per row (16 × 90 bytes),
calls `ggml_dequantize(..., m=16, n=160)`, and compares the result against a
deterministic CPU nibble-table reference.  It also checks shape, dtype,
finiteness, device, and repeatability.

Fresh results after the fix:

```text
tests/models/test_qwen4exp_ple.py: 6 passed, 1 skipped
Qwen4Exp/GGUF/PLE suite: 28 passed, 3 skipped
```

The CUDA test was skipped cleanly.  `nvidia-smi` reported that it could not
communicate with the NVIDIA driver and `torch.cuda.is_available()` was false;
the test will execute the real kernel automatically in a CUDA-enabled
environment.

## Review conclusion

The loader now rejects the unsafe IQ4_NL configuration before the old
block-width assertion and accepts the actual REAP PLE geometry.  This proves
the configuration invariant and provides a real-GPU kernel gate, but does not
claim a live GPU result in this environment.
