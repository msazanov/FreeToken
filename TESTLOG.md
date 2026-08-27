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
