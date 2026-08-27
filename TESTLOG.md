# RTX 2070 Test Log

This is an append-only evidence log for the FreeToken RTX 2070 fork. Do not edit
or delete past rows: corrected interpretations belong in a later dated entry.

## Recording contract

For every experiment record:

- date, git revision/worktree, model, exact launch flags and hardware state;
- workload input/output shape, warm-up and repetition policy;
- TTFT, prefill throughput, decode throughput, cache fields, GPU/RAM/swap data;
- result or failure, plus the path to the raw JSON artifact in `benchmarks/results/`;
- whether the result is kernel-only, server-level, or an end-to-end quality result.

For context-speed curves, also append a plot-ready point to the sibling
`slices.jsonl`: each point records the commit, dirty-state, manual parameter
labels, runtime cache geometry, sampling mode, actual context, TTFT, prefill and
decode. Use `temperature=0` / `greedy-argmax` until FreeToken wires API seeds
through to the sampler; a nominal seed would currently be misleading.

## 2026-08-27 — DeepSeek Harness composition inspection

Read-only inspection of the live `dsh web --no-open` installation and
`dsh --profile web --dump-config` found that the active web patch mounts only
`@local/dsh-freetoken-telemetry` and `@local/dsh-ornith-compat`. Both
`compaction-basic` and `tool-result-pruner` remain explicitly disabled. Their
available defaults are respectively automatic compaction at 80% of routed
capacity with a 16% retained tail, and 8192-character tool-result pruning
(4096-character head plus 1024-character tail), but neither policy currently
runs. The active FreeToken geometry is 122880 TQ4-NC KV pages, 1429 MoE cache
slots (14% residency), and 8 Mamba slots. No configuration was changed during
inspection. This establishes the cache benchmark prerequisite: use the active
system/tool prefix before enabling or tuning any retention policy.

## 2026-08-27 — existing TQ4-NC evidence

| Experiment | Method | Result | Scope / limitation |
|---|---|---|---|
| SM75 TQ4 split-K decode | CUDA-event microbenchmark, Ornith geometry (16 Q / 2 KV heads, D=256) | 18 fixed splits versus 8: 1K `-20.0%`, 4K `-13.8%`, 16K `-28.7%`, 32K `-30.9%` kernel time | Kernel-only; graph-safe scratch was resized to 18 for TQ4 on SM75. |
| Mixed long-prefill attention | 18,000-token packed prefix + 1,024-token current block | `1768.77 ms` all-packed → `551.32 ms` mixed, `3.21x` faster | Attention kernel only; not full TTFT. |
| FP16 Tensor-Core route | PTX inspection on RTX 2070 | No `mma.sync`; route disabled | Ornith's Q/KV grouping exposes M=8, not a useful Turing Tensor-Core tile. |
| Paired packed-byte load | 16K split-decode CUDA-event A/B | `1.475 ms` one-load vs `1.479 ms` two-load control | Noise-level 0.3% delta; experiment reverted. |
| Live smoke request | `:1919`, 3,088 input tokens, single request | completed in `19.50 s`; chunk logs `1024×3 + 16` | Functional check only, not a matched old/new A/B. |

## 2026-08-27 — pending context matrix

The next entries will use the repository-context compression scenario at requested
1K, 16K, 64K and 112K input-token tiers. Raw data will be written below
`benchmarks/results/` outside tmpfs.

## 2026-08-27 — invalid runner validation retained

The first live runner attempt requested the 1K tier but reserved a 512-token
margin and consumed that body budget with the leading generic README. Server usage
was 660 prompt / 255 completion tokens, TTFT 3.812 s and decode 28.63 tok/s, but
the compression score was only 2/5 because `control_api.py` and its `build_stats`
evidence were absent from the prompt. This is a **methodology failure**, not a
quality measurement. Raw artifact retained at
`benchmarks/results/2026-08-27-ornith-context-matrix/compression-1024.json`.

## 2026-08-27 — repository-compression cold-context matrix, corrected runner

| Requested tier | Actual prompt | TTFT | Server-level prefill estimate | Decode | Quality anchors | Runtime peak / floor | Raw artifact |
|---:|---:|---:|---:|---:|---:|---|---|
| 1K | 1,011 | 5.545 s | 182.32 tok/s | 26.77 tok/s | 4/5 | validation run | `benchmarks/results/2026-08-27-ornith-context-matrix-v2/compression-1024.json` |
| 16K | 16,373 | 134.259 s | 121.95 tok/s | 20.56 tok/s | 4/5 | GPU 100%, 87 C, 136.46 W; RAM available min 2.00 GiB; swap free min 10.99 GiB | `benchmarks/results/2026-08-27-ornith-context-matrix-v2/compression-16384.json` |
| 64K | 65,524 | 1125.087 s | 58.24 tok/s | 12.69 tok/s | 5/5 | GPU 100%, 88 C, 135.09 W; RAM available min 1.77 GiB; swap free min 10.13 GiB | `benchmarks/results/2026-08-27-ornith-context-matrix-v2/compression-65536.json` |
| 112K | 114,678 | 2915.792 s | 39.33 tok/s | 9.12 tok/s | 4/5 | GPU 100%, 89 C, 140.98 W; RAM available min 2.15 GiB; swap free min 9.96 GiB | `benchmarks/results/2026-08-27-ornith-context-matrix-v2/compression-114688.json` |

All corrected rows deliberately use a distinct leading tier marker, so radix cache
reported zero cached input tokens: these are cold-context measurements. The 64K
and 112K requests released their KV pages after completion; no persistent allocator
leak was observed. The 112K result is an end-to-end repository-compression task:
the model produced 255 tokens after a 48m 35.8s TTFT; it found 4/5 source anchors
(`build_stats` was the missing literal), so the result remains a valid runtime
measurement but not a perfect quality pass.

## 2026-08-27 — DeepSeek Harness-shaped radix-cache reuse

The active FreeToken server was read-only tested at `:1919` with its live
`--enable-cache-report`, 122880-page TQ4-NC radix cache, 1429 MoE slots and
8 Mamba slots. Each tier preserves the deployed DSH coding-agent system prompt
and the local Ornith tool-protocol section, then runs (a) a cold unique request,
(b) a byte-identical replay, and (c) the original turn as assistant history plus
a tiny new user request. The telemetry plugin does **not** expose the private
active-session request body, so this is deliberately labelled
*Harness-shaped*, rather than an exact session replay. `temperature=0`,
`reasoning_effort=off` and `max_tokens=64` isolate prefix reuse from extended
reasoning quality.

| Requested tier | Actual prompt | Scenario | Cached / new tokens | Cache hit | TTFT | Wall | Raw artifact |
|---:|---:|---|---:|---:|---:|---:|---|
| 1K | 1,148 | cold | 0 / 1,148 | 0.00% | 7.346 s | 8.50 s | `benchmarks/results/2026-08-27-ornith-harness-cache/cache-1024.json` |
| 1K | 1,148 | exact warm | 1,088 / 60 | 94.77% | 1.630 s | 2.72 s | same artifact |
| 1K | 1,222 | append | 1,187 / 35 | 97.14% | 1.616 s | 3.26 s | same artifact |
| 16K | 16,510 | cold | 0 / 16,510 | 0.00% | 135.558 s | 136.45 s | `benchmarks/results/2026-08-27-ornith-harness-cache/cache-16384.json` |
| 16K | 16,510 | exact warm | 16,448 / 62 | 99.62% | 2.013 s | 2.94 s | same artifact |
| 16K | 16,570 | append | 16,534 / 36 | 99.78% | 1.700 s | 2.72 s | same artifact |

This is an end-to-end server result, not a kernel microbenchmark. It establishes
that long agent histories are interactive **when the prefix is stable**. The
remaining cold-miss bottleneck is still long prefill, and must be profiled
separately; do not average warm and cold numbers into one claimed throughput.

## 2026-08-27 — runtime attribution without service reconfiguration

The already-running service was observed without a restart or cache-pool change.
The 16,510-token cold Harness-shaped request spent 131 of 133 one-second samples
in prefill, with GPU utilisation averaging 98.6% (peak 100%), a 87 C peak, and
at least 2.28 GiB RAM / 10.28 GiB swap still available. Per-1024-token blocks
slowed from 143.8 tok/s at 4,096 processed tokens to 96.7 tok/s at 14,336, while
the GPU remained saturated: this attributes the growing cold TTFT to
context-dependent GPU work, principally attention/prefill, not to host-memory
pressure or a KV capacity failure.

For a decode-only diagnostic, `ignore_eos=true` held a 1K warm-prefix request to
255 generated tokens. The three client decode measurements were 28.98, 28.87,
and 26.93 tok/s (the lower third row had a larger uncached append). The live
MoE counters reported 8 active experts per MoE layer, 3.46–3.49 missing experts
per layer (43.25–43.59% miss-rate), `fetched_per_layer=0`, and the same amount
in `cpu_per_layer`. Thus a sizeable decode component is CPU expert execution;
there was no evidence of a PCIe fetch component in this window. This diagnostic
uses forced post-EOS output, so it is **not** a response-quality result and its
append cache result must not be generalized to normal turns. Raw artifact:
`benchmarks/results/2026-08-27-ornith-harness-cache-decode-profile/cache-1024.json`.
