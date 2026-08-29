# Qwen3.8 REAP-256 PLE `IQ4_NL` Compatibility Fix

Date: 2026-08-29
Scope: `Qwen4ExpPLE` GGUF PLE table type gate only

## Root cause

The REAP-256 GGUF header identifies `per_layer_token_embd.weight` as
`GGML_IQ4_NL` (enum 20), with logical shape `(160, 320001536)`. The existing
`_HostNGramEmbedding.load_host_weights` path accepted only `GGML_Q5_1` even
though the native `ggml_dequantize` dispatch and Python `DEQUANT_TYPES` already
include `IQ4_NL`. PLE forward already passes the table's actual `_gguf_type`,
so no runtime or kernel change was necessary.

## RED

Added `tests/models/test_qwen4exp_ple.py` before changing production code. The
tests cover:

- `Q5_1` acceptance;
- `IQ4_NL` acceptance;
- rejection of unsupported `F16`;
- rejection of an incorrect logical row width;
- rejection of an incorrect packed `row_bytes` value.

Command:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python \
/home/random/freetoken-turing/.venv/bin/python -m pytest \
tests/models/test_qwen4exp_ple.py -q
```

Result against the old implementation:

```text
.FF..                                                                    [100%]
2 failed, 3 passed in 4.07s
```

The `IQ4_NL` test failed with the old `must be Q5_1` error. The unsupported
type test also failed because the old error did not describe the supported
dispatch set. The row-shape tests and Q5_1 acceptance passed, confirming that
the regression targeted the intended gate.

## GREEN

The minimal production change imports `DEQUANT_TYPES` and replaces the
`GGML_Q5_1` equality check with membership in that exact dispatch set. The
error reports the supported native quantized types and the received type.
The existing logical-width, `row_bytes`, packed-table and minimum-row checks
were left unchanged.

Focused command result:

```text
.....                                                                    [100%]
5 passed in 4.07s
```

Relevant regression command:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python \
/home/random/freetoken-turing/.venv/bin/python -m pytest \
tests/models/test_qwen4exp_ple.py \
tests/models/test_qwen4_exp.py \
tests/models/test_qwen4exp_gguf.py \
tests/models/test_qwen4exp_gguf_experts.py -q
```

Result:

```text
27 passed, 2 skipped in 4.14s
```

## Scope and non-goals

- Changed: `python/freetoken/models/qwen4_exp/model.py` PLE type validation.
- Added: focused PLE regression tests and this report.
- Updated: `README.md`, `TESTLOG.md`, and `CHANGELOG.md` with the evidence.
- Not changed: experts, offload runtime, cache policy, kernels, model files,
  benchmark code, downloads, or server state.
- Not run: model serving, benchmark, download, or subagent.
