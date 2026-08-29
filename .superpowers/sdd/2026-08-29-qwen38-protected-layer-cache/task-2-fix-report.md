# Task 2 fix report: protected-layer Critical regressions

## Scope

This change addresses only the two Critical findings from the independent review of
`5586884e60ed813a4516973145d53e542ff0dc13` (`feat: protect Qwen decode experts by
layer`):

1. Triton protected admission must not stage a copy or eviction for a route whose
   experts are already resident.
2. Direct engine prefill warmup must not inherit the cache's initial `decode` trace
   phase and accidentally use protected decoder admission.

No model, server, benchmark, download, LRU policy, or ordinary prefill semantics were
changed. The Important/Minor review findings remain open.

## Root-cause confirmation at `5586884`

### Critical 1: Triton resident-route corruption

`_ensure_experts_protected_layer_kernel` computed `needs` correctly, but then iterated
through every `rank < unique_count`. For a resident rank, `is_candidate` was all false;
`tl.argmax` therefore selected a dummy expert and `tl.argmin` selected a victim even
though no copy was needed. The kernel then incremented `copy_count`, staged
`evict_slots/src_indices`, and could invalidate a resident mapping.

The CPU reference already skips a resident target. The fix adds the same missing-route
guard in the Triton loop: victim selection and copy staging run only when that rank has
at least one `needs` lane.

### Critical 2: direct warmup phase leak

`OffloadMoeCache` initialized `_trace_phase` to `"decode"`. Normal scheduler forwards
call `_set_moe_trace_phase`, but `Engine._warmup_prefill()` directly calls
`model.forward()` before that scheduler seam. Therefore the warmup entered
`uses_protected_layer_admission()` as a decoder batch.

The fix scopes `set_trace_phase("prefill")` to the direct warmup and restores the prior
phase in its existing `finally` cleanup. The initial cache phase, scheduler phase
handling, LRU path, and normal prefill dispatch are unchanged.

## TDD evidence

### RED

The direct warmup regression initially failed for the observed reason:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/moe/test_offload.py -q \
  -k engine_warmup_prefill_disables_protected_decoder_admission

1 failed, 38 deselected
assert [True] == [False]
```

The CUDA regression was added as a real Triton test and is gated because this sandbox
has no CUDA:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4exp_gguf_experts.py -q \
  -k protected_layer_cuda_repeat_resident_route_has_no_copy_or_eviction

1 skipped, 10 deselected
```

The pre-fix phase predicate was also directly reproduced: initial `decode` returned
`uses_protected=True`; the CPU resident-route reference returned `num_indices=0` for a
fresh raw repeated route, isolating the remaining implementation defect to Triton.

### GREEN

Focused CPU regressions and existing protected-layer tests:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/moe/test_offload.py -q \
  -k 'engine_warmup_prefill_disables_protected_decoder_admission or \
      protected_layer_repeated_route_stays_resident or \
      protected_layer_prefill_keeps_dynamic_hybrid_admission or \
      scheduler_sets_protected_trace_phase_without_collect_stats'

4 passed, 35 deselected
```

CUDA-gated Triton regression:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4exp_gguf_experts.py -q \
  -k protected_layer_cuda_repeat_resident_route_has_no_copy_or_eviction

1 skipped, 10 deselected
```

Syntax check:

```text
PYTHONDONTWRITEBYTECODE=1 /home/random/freetoken-turing/.venv/bin/python -m py_compile \
  python/freetoken/engine/engine.py \
  python/freetoken/moe/offload_kernels.py \
  python/freetoken/moe/offload_cache.py \
  tests/moe/test_offload.py \
  tests/models/test_qwen4exp_gguf_experts.py
```

Relevant full suite:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/moe/test_offload.py tests/models/test_qwen4exp_gguf_experts.py -q

43 passed, 5 skipped, 2 failed
```

The two failures are pre-existing/environmental and unrelated to this diff:

- explicit `fi` backend requires unavailable `flashinfer`;
- an older test expects `num_experts`, while current validation reports
  `cache_size 3 < required slots 4`.

## CUDA limitation

`torch.cuda.is_available()` is false in the local sandbox, so Triton compilation and
CUDA parity could not be executed here. The CUDA test is an executable gated test, not
a mock; it must be run on an SM75/Turing machine before treating CUDA parity as
empirically confirmed. No model or benchmark was launched.

## SHA

- Root-cause baseline reviewed: `5586884e60ed813a4516973145d53e542ff0dc13`.
- Fix implementation: the atomic commit containing this report; its exact SHA is
  recorded in the final handoff after commit creation.

