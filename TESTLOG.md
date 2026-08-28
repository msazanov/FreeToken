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

## 2026-08-28 — Qwen4Exp real-GGUF non-expert loader smoke

Worktree: `feat/qwen4exp-gguf-turing`, post-`bb0dde7` dirty implementation.
Model: local AtomicChat `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`, 33 GGUF
shards. Hardware: RTX 2070 SM75; no inference service active. The smoke
initialized FreeToken TP=1 exactly as the engine does, then exhausted
`iter_gguf_weights(... include_moe_experts=False, include_non_moe=True)`.

Result: completed successfully after one cold CUDA compilation of the borrowed
GGUF kernel, yielding 885 non-expert tensors. Required sentinels had these
layouts: GDN input `(16480, 2720)` `uint8`; GDN output `(2560, 6144)`
`bfloat16`; PLE conv `(10240, 1, 4)` `bfloat16`; QSA QKV `(13312, 2720)`
`uint8`; QSA indexer `(640, 5120)` `uint8`; embedding/LM head `(248320,
2720)` `uint8`. The iterator's final fusion assertions passed.

This is a loader-level integration test, **not** a TTFT, tok/s or quality result.
It deliberately skipped both the 38 GiB PLE Q5_1 table and every routed-expert
bank; model construction/service startup remains blocked on their providers.

## 2026-08-28 — Qwen4Exp PLE Q5_1 mmap and row-dequant probe

Worktree: `feat/qwen4exp-gguf-turing`, dirty PLE-provider implementation.
Model/table: the same local 33-shard AtomicChat Q4_K_M GGUF. The probe parsed
real metadata, instantiated only `_HostNGramEmbedding`, loaded its three PLE
metadata buffers, opened `per_layer_token_embd.weight`, then selected rows
`[0, 1, 2, 100, 1000]` and called the existing CUDA `ggml_dequantize` path.

Result: table view `(320001536, 120)` Q5_1; gathered packed rows `(5, 120)`;
dequantized output `(5, 160)` BF16 with all finite values. RSS delta while
opening and sampling was `1.95 MiB`; therefore the table is mmap-backed rather
than resident in system RAM. The first call reported `ninja: no work to do`, so
the prior GGUF kernel compilation was not charged to this probe.

This proves table opening, row selection and dequantization only. It is neither
a complete prefill/decode benchmark nor an end-to-end PLE quality validation;
the routed-expert bank remains absent.

## 2026-08-28 — Qwen3.8 separate GGUF expert source (unit)

- **Hypothesis:** Qwen4's gate, up, and down expert stacks can remain direct
  GGUF views rather than being concatenated into a 50-GiB anonymous host bank.
- **Test:** `PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/models/test_qwen4exp_gguf_experts.py -q`
- **Result:** `1 passed`. The test builds one synthetic expert layer and proves
  each returned `[expert, row, packed-bytes]` tensor has the same storage
  address as its source. This is a loader-layout check, not inference or
  throughput evidence.

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

## 2026-08-27 — live MoE-cache expansion, preserving 122K KV

`POST /v1/cache/rebuild` was issued idle-only with `moe_cache_size=1700`.
FreeToken accepted the request in 3.6 s and remained serving; KV stayed at
122,880 pages/tokens and Mamba at 8 slots. This was an MoE-only rebuild, so the
scheduler intentionally retained the radix prefix cache. GPU allocation grew
from 6,646 MiB to 7,084 MiB. The fixed 0.85 memory-ratio cache budget leaves
about 135 MiB after the target geometry; no model weights or CPU expert banks
were reloaded.

| Configuration | Decode samples | Mean client decode | Mean MoE miss-rate | Missing experts / layer |
|---|---:|---:|---:|---:|
| 1429 slots (14.0% residency) | two 255-token warm-prefix diagnostic turns | 28.93 tok/s | 43.37% | 3.47 / 8 |
| 1700 slots (16.6% residency) | same shape, two turns | 31.27 tok/s | 35.07% | 2.81 / 8 |

The paired diagnostic therefore shows **+8.1% decode speed** and **-8.30
percentage points** MoE miss-rate (about -19% relative). It uses
`temperature=0`, `reasoning_effort=off`, and `ignore_eos=true`; it measures
runtime only, not answer quality. Artifact:
`benchmarks/results/2026-08-27-ornith-harness-cache-moe1700/cache-1024.json`.

A normal end-to-end repository compression smoke test after the rebuild used a
1,012-token cold prompt, generated 383 tokens at 28.44 tok/s, had 5.514 s TTFT,
and retained 4/5 required source anchors. It completed normally and is a
functional/quality guard, not a strict pre/post quality comparison:
`benchmarks/results/2026-08-27-ornith-moe1700-smoke/compression-1024.json`.

## 2026-08-27 — active DeepSeek Harness retention policy

The web-profile patch now enables `tool-result-pruner` before
`compaction-basic`, verified by `dsh --profile web --dump-config` and a clean
restart of `deepseek-harness-local.service` (`HTTP 200` at port 3080). For the
exact `freetoken / Ornith 1.5 35b` route: automatic compaction starts at
`floor(122880 × 0.88) = 108134` tokens, retains the newest 49,152 tokens
verbatim, limits a compaction summary to 4,096 tokens, and has no repeated
normal-pressure retry. Tool results above 32,768 characters are reduced to a
24,576-character head plus a 4,096-character tail only once compaction pressure
qualifies. This preserves large working context while preventing a repeat of a
hard 122,880-token overflow. The policy loads and the Harness is functional;
its actual >108K compaction path has **not** been invoked yet and remains a
future quality/latency measurement.

## 2026-08-28 — Qwen3.8 real file-backed expert probes

Worktree: `feat/qwen4exp-gguf-turing`, post-`f4b4825` then dirty three-bank
cache implementation. Model: local 33-shard AtomicChat Q4_K_M, RTX 2070 SM75,
no inference service active.

- Opening all 48 layers produced direct expert views with shapes gate/up
  `[512, 640, 1100]` and down `[512, 2560, 360]`; gate/up types are IQ3_S or
  IQ2_S by layer and down is IQ4_NL. The virtual mapped ranges are 13.037 GiB,
  13.037 GiB and 21.094 GiB respectively. The process RSS delta was 889.79 MiB,
  not a 47–50 GiB anonymous-bank allocation; all views were contiguous.
- A fresh 512-slot GPU cache copied ten routed layer-0 experts from the
  file-backed source in 1.0142 s while allocating 1138.5 MiB GPU cache. After
  source pages were warm, three disjoint ten-expert selections took 56.361,
  23.925 and 22.393 ms. This is a selected-row transfer microprobe, not a
  decoder tok/s result.
- A real layer-0 IQ3_S/IQ3_S/IQ4_NL separate-projection MoE GEMV returned a
  finite `[1, 2560]` output. The first 88.9278-s measurement included fresh CUDA
  extension compilation; after that, three GPU-only calls measured 0.470, 0.399
  and 0.395 ms. Therefore transfer/cache misses, rather than the expert GEMV,
  are the current bottleneck.
- Unit coverage after the implementation:
  `PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/models/test_qwen4exp_gguf_experts.py -q` → `4 passed`.

## 2026-08-28 — Qwen3.8 file-backed routed-prefill regression

The first implementation of `qwen4_gguf` still inherited the generic prefill
contract, which calls `materialize_layer(layer_id)` before the MoE router has
selected experts. The new focused regression test intentionally made that method
raise. It failed at `python/freetoken/layers/moe.py:400`, proving the full-layer
path was still active. The corrected path calls `ensure_experts` and
`copy_missing`, then runs the compact GPU slots; the test also verifies rewritten
LRU slot ids, the three-bank views, `n=None`, and `is_prefill=True`.

Verification after the change:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4exp_gguf_experts.py tests/models/test_qwen4exp_gguf.py \
  tests/models/test_qwen4_exp.py tests/models/test_gguf_type_tables.py \
  tests/moe/test_fused_copy.py -q
22 passed in 6.81s
```

This establishes that file-backed prefill will not **unconditionally** copy all
512 experts. It does not yet establish prompt throughput: the next result must
come from a real Qwen request.

## 2026-08-28 — Qwen3.8 first real service-load attempts

Test geometry: current Qwen branch, RTX 2070 8 GiB, `float16`, 512 file-backed
expert slots, 2,048-token KV, one request, port 1920 loopback. This small KV
budget was intentional: it isolates compatibility/loading before any long-context
claim. Ornith was not restarted or altered.

1. The first command used the venv's older editable checkout and failed before
   engine load: `AutoConfig` looked for a nonexistent `config.json` and the
   tokenizer worker could not construct a HF tokenizer. This was a launch-path
   error, not a Qwen weight error. The corrected command runs the current branch
   with `PYTHONPATH=.../freetoken/python python -m freetoken.cli`.
2. The corrected engine resolved `qwen4exp`, selected the QSA backend, disabled
   CUDA graphs (expected for host-side PLE work), and intentionally selected a
   naive cache because Qwen-owned runtime state cannot safely resume a generic
   radix prefix. It then failed in the tokenizer worker with
   `KeyError: 'qwen4exp'` from Transformers' GGUF converter. A red regression
   test captured that missing mapping; `qwen4exp -> qwen2` made the relevant
   tokenizer/adapter suite pass.
3. The same launch had progressed into weight loading and stopped at
   `KeyError: model.layers.1.ple.key_proj.qweight`. Direct inspection of the
   actual first shard found `blk.1.ple_key.weight` and
   `blk.1.ple_value.weight`, proving that the checkpoint is not missing them.
   The iterator lacked their output mapping. A red synthetic-GGUF test captured
   it; both now yield the expected packed projection names.

Post-fix verification:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4exp_gguf.py tests/models/test_qwen4_exp.py \
  tests/models/test_qwen4exp_gguf_experts.py tests/models/test_gguf_dispatch.py -q
32 passed in 9.85s
```

No answer-generation or tok/s result exists yet. The next live launch is the
required evidence for that stage.

## 2026-08-28 — Qwen3.8 PLE normalization mapping correction

After the tokenizer/projection corrections, the live scheduler reached the
next PLE field and failed at `model.layers.1.ple.norm_key.weight`. The real GGUF
contains `blk.1.ple_norm_key.weight`, `ple_norm_query.weight`, and
`ple_norm_conv.weight`; the iterator was spelling their destinations as
`norm_norm_*`. This is a local adapter typo, not missing checkpoint data.

The PLE iterator regression was expanded with all three real source names and
first failed on the expected `norm_key` absence. The destination is now
`model.layers.1.ple.norm_{key,query,conv}.weight`.

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4exp_gguf.py tests/models/test_qwen4_exp.py \
  tests/models/test_qwen4exp_gguf_experts.py tests/models/test_gguf_dispatch.py -q
32 passed in 8.10s
```

The service again stopped cleanly after the failed child scheduler. No Qwen
server remains running, and no tok/s result is claimed at this point.

## 2026-08-28 — Qwen3.8 runtime geometry findings

With the full 512-slot file-backed expert cache and a 2,048-token KV cache, the
server completed initialization but retained only 0.49 GiB VRAM. A first
one-line request then reached the GDN forward and failed with CUDA OOM in the
Triton `chunk_fwd_o` kernel. This is a runtime scratch-budget failure; it does
not prove a model or weight failure.

The first FP16 attempt had failed even earlier because GGUF embeddings and small
GGUF tensors currently compute in BF16 while the GDN state pool followed the
requested FP16 dtype. Triton rejected that mixed `bf16`/`fp16` conv state. A
consistent BF16 diagnostic launch passed that type boundary and reached the OOM,
which isolates the two problems. Native FP16 remains a required later adapter
task for SM75; BF16 is used only to expose the remaining functional path.

Trying 256 slots exposed a second generic assumption: engine/cache validation
required `cache_size >= num_experts` even though the Qwen4 file-backed prefill
no longer materializes the full layer. No matching FreeToken issue/PR was found
in the GitHub search. The guard is now parameterized: generic formats still need
512 slots, while file-backed Qwen4 accepts its router working set (`top_k=10`).

Focused verification after that change:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/engine/test_cache_budget.py::test_guard_passes_when_size_covers_one_expert_per_layer \
  tests/engine/test_cache_budget.py::test_guard_raises_actionable_error_when_too_small \
  tests/engine/test_cache_budget.py::test_file_backed_expert_cache_only_needs_one_routed_token_working_set \
  tests/models/test_qwen4exp_gguf_experts.py -q
9 passed in 4.02s
```

The broader cache-budget file had two pre-existing environment failures because
its `fi` backend needs FlashInfer, which is not installed. Those are not counted
as evidence for or against this patch.

## 2026-08-28 — Qwen3.8 QSA/MRoPE first-forward correction

With 256 slots the server initialized with 1.06 GiB free VRAM and began the
actual short prefill. It then compiled the router (falling back to pure PyTorch
because optional `triton_kernels` is absent) and reached the first QSA layer,
where it stopped with `AttributeError: Batch has no attribute rope_positions`.
The attention code was meant to fall back to ordinary text positions when MRoPE
positions are absent, but direct attribute access prevented that fallback.

`Batch.rope_positions` is now an optional, default-`None` field; Qwen4 attention
uses defensive lookup before falling back to `positions`. This is a text-only
correctness fix. Multimodal callers may later set actual `[3, tokens]` MRoPE
coordinates.

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4_exp.py tests/models/test_qwen4exp_gguf.py \
  tests/models/test_qwen4exp_gguf_experts.py -q
16 passed in 4.61s
```

## 2026-08-28 — optional fused router dependency

During the first real Qwen forward, FreeToken reported a pure-PyTorch top-k
router fallback because `triton_kernels` was absent. We installed the official
Triton-Lang source package. Main initially failed to import against the pinned
Triton 3.6.0, so it was replaced with the matching upstream `v3.6.0` commit.

```text
numpy 2.4.6
triton 3.6.0
from triton_kernels.topk import topk  # OK
```

This verifies package compatibility, not speed. The next Qwen request must
verify that the fused kernel itself compiles and runs on SM75.

The real request showed that it does not: `triton_kernels.topk` raised a Triton
`arange` geometry compilation error. The package remains installed for future
compatible shapes, but Qwen must safely use the existing PyTorch fallback. A
red/green regression now verifies that any optional router exception yields the
same normalized top-k result rather than crashing the scheduler:

```text
13 passed in 6.76s
```

## 2026-08-28 — Qwen3.8 FP16 activation-boundary repair

The 256-slot, `--dtype float16` live launch reached the first GDN
causal-convolution with BF16 embeddings but FP16 recurrent state. That mismatch
was produced by `GGUFEmbedding`, which historically dequantized embeddings to
BF16 regardless of the engine dtype. It is not evidence that Q4_K_M expert
weights should be expanded to FP16.

The Qwen4 GGUF adapter now passes its requested runtime dtype to the embedding,
so an FP16 engine emits FP16 embeddings. The model's packed GGUF weights retain
their original Q4_K_M/IQ types; FP16 is only the activation/state math format
native to this Turing experiment.

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4_exp.py tests/models/test_qwen4exp_gguf.py \
  tests/models/test_qwen4exp_gguf_experts.py \
  tests/moe/test_fused_moe.py::test_fused_topk_falls_back_when_optional_triton_kernel_rejects_geometry -q
18 passed in 4.47s
```

No performance figure is claimed by this unit-level repair. The next required
measurement is a real FP16 server request, followed by warm decode and context
tests.
