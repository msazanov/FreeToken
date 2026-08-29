# Task 2 Report: Decode-only protected/transient admission

## Design

Added protected-layer admission for file-backed Qwen GGUF decode only. Dispatch requires protected policy, Qwen GGUF format, a file-backed layer, and decode trace phase. Prefill continues on the existing dynamic hybrid-LRU path.

The CPU reference and Triton kernel build unique routes in first-seen rank order. The first P routes use [L * P, (L + 1) * P) and the remaining routes use [num_layers * P, cache_size). Each target range evicts only non-active slots using the deterministic (usage, slot) key. Mapping cleanup, in-place route rewriting, and num_indices/evict_slots/src_indices staging preserve the existing copy contract.

The scheduler now updates trace phase for protected policy even when moe_collect_stats=False; stats reset and collection remain gated by moe_collect_stats.

## Changed files

- python/freetoken/moe/offload_cache.py: protected decode dispatch predicate.
- python/freetoken/moe/offload_kernels.py: public protected admission API, CPU reference, and Triton implementation.
- python/freetoken/scheduler/scheduler.py: protected-policy trace phase update when stats are disabled.
- tests/moe/test_offload.py: CPU range, pressure, repeat, prefill-dispatch, and scheduler regression coverage.
- tests/models/test_qwen4exp_gguf_experts.py: Qwen-sized 48x512 top-10 CUDA-gated parity.

## Tests

### RED: protected admission

Command:

~~~text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_offload.py tests/models/test_qwen4exp_gguf_experts.py -q -k protected_layer
~~~

Output:

~~~text
.......F.s                                                               [100%]
1 failed, 8 passed, 1 skipped, 36 deselected in 3.89s
~~~

The expected failure showed the old global file-backed hybrid-LRU placed rank-tail/rank-1 routes without protected-layer geometry.

### RED: scheduler phase regression

Command:

~~~text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_offload.py -q -k scheduler_sets_protected_trace_phase_without_collect_stats
~~~

Output:

~~~text
F                                                                        [100%]
1 failed, 37 deselected in 3.59s
~~~

### GREEN

Command:

~~~text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_offload.py tests/models/test_qwen4exp_gguf_experts.py -q -k 'protected_layer or qwen4_file_backed_lru_admission'
~~~

Post-amend output:

~~~text
...........s                                                             [100%]
11 passed, 1 skipped, 36 deselected in 4.27s
~~~

Scheduler regression green command:

~~~text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_offload.py -q -k scheduler_sets_protected_trace_phase_without_collect_stats
~~~

Output:

~~~text
.                                                                        [100%]
1 passed, 37 deselected in 4.01s
~~~

Syntax verification command:

~~~text
PYTHONDONTWRITEBYTECODE=1 /home/random/freetoken-turing/.venv/bin/python -m py_compile python/freetoken/moe/offload_kernels.py python/freetoken/moe/offload_cache.py python/freetoken/scheduler/scheduler.py tests/moe/test_offload.py tests/models/test_qwen4exp_gguf_experts.py
~~~

Exit code: 0.

The CUDA parity test skipped because this environment has no CUDA. No model, server, or benchmark was run.

## Commit

Implementation SHA: 5586884e60ed813a4516973145d53e542ff0dc13

## Concerns

- CUDA/Triton execution was unavailable locally; only the parity test skip path was exercised.
- An exploratory full run of both changed test files had two unrelated baseline/environment failures: one requires unavailable flashinfer, and one expects an older num_experts validation message. The targeted Task 2 commands are green.
