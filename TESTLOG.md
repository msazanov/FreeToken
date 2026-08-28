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

## 2026-08-28 — Qwen3.8 first complete FP16 QSA/MRoPE request

After the FP16 embedding repair, the first scheduler request reached QSA and
then failed on a missing FreeToken MRoPE Triton entry point. GitHub search found
no existing FreeToken patch. We compared the SGLang Qwen MRoPE kernel with the
fresh `tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark` operational patch. The
latter identifies Qwen3.8's partial rotary width (32 half-lanes in a padded
128-lane head) and adds the required bounds check. The local kernel now follows
that implementation rather than replacing MRoPE with host-side PyTorch.

Focused regression after the port:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4_exp.py tests/models/test_qwen4exp_gguf.py \
  tests/models/test_qwen4exp_gguf_experts.py \
  tests/moe/test_fused_moe.py::test_fused_topk_falls_back_when_optional_triton_kernel_rejects_geometry -q
19 passed in 6.35s
```

Live server configuration: RTX 2070 8 GiB, `--dtype float16`, Q4_K_M GGUF,
256 file-backed LRU expert slots, `--num-tokens 2048`, QSA page size 4, naive
cache, one request. It initialized with 1.06 GiB free VRAM.

| request | prompt / completion | result | measured time / server metric |
| --- | --- | --- | --- |
| cold smoke | 43 / 1 | HTTP 200 | 78.74 s total; server reported 0.41 input tok/s |
| repeat | 47 / 7 | HTTP 200 | 112.22 s total; server reported 0.40 input tok/s |
| second repeat | 47 / 7 | HTTP 200 | 100.60 s total; server reported 0.35 input tok/s, decode line 0.02 tok/s |

The API completion proves that FP16 GDN, QSA, MRoPE, packed Q4_K_M GGUF reads
and the offloaded three-bank expert path can execute together on SM75. It does
**not** establish a usable runner: both seven-token responses were nonsensical
`reasoning_content` fragments rather than the requested `pong`, and the warm
repeat did not improve materially. Candidate causes to investigate before any
long-context benchmark are adapter weight mapping/tokenizer-template correctness
and the still synchronous NVMe expert-miss path. Do not treat these figures as
a Qwen3.8 performance claim.

## 2026-08-28 — Qwen3.8 official tokenizer parity audit

To rule out a malformed chat prompt, we downloaded only the official Qwen
`config.json`, `chat_template.jinja`, `tokenizer_config.json` and 13 MiB
`tokenizer.json` into `~/dev/qwen/reference-qwen38` (not `/tmp`). The GGUF's
embedded template hash is exactly the official one:

```text
c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041
```

The first comparison nevertheless found an exact tokenization failure: the
official prompt for `Reply with exactly: pong` at `reasoning_effort=low` had 45
IDs and ended in `<think>` ID `248068`; the generic GGUF Qwen2 converter emitted
47 IDs and split `<think>` as IDs `13314, 741, 29`. This explains the invalid
generation prefix and makes the earlier nonsense output non-diagnostic of model
quality.

The loader now registers Qwen3.8's template-only non-special controls as added
tokens. It deliberately uses `add_tokens`, because the official tokenizer marks
`<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>` and
`</tool_response>` as `special: false`; they must be single IDs but remain
visible to the reasoning/tool parsers.

Verification:

```text
focused local suite: 20 passed in 4.48s
official-vs-GGUF render same = True
official-vs-GGUF IDs: 45 / 45, same = True
tail: [..., 248045, 74455, 198, 248068, 198] on both sides
```

The currently running server initialized its tokenizer before this correction.
It must be restarted before the next quality/speed request; no conclusion about
Qwen's answer quality is valid until that run completes.

## 2026-08-28 — Qwen3.8 GDN gate and RMSNorm audit

The tokenizer-corrected, thinking-disabled request still produced deterministic
garbage (`Packagefinois已经是 symbol];`, HTTP 200, 17 input / 7 output tokens,
144.40 s). This ruled out the template/parsing explanation but is not a quality
result.

We investigated a fresh external Qwen4 RMSNorm report. It correctly describes
ones-centered Qwen weights being broken by a runtime that also applies `1 + w`.
Actual values in this GGUF confirm the same raw convention:

```text
blk.0.ssm_norm.weight       mean 0.9668, std 0.0326
blk.0.hc_attn_norm.weight   mean 0.9365, std 0.4729
blk.0.hc_ffn_norm.weight    mean 0.8905, std 1.0008
blk.3.indexer.q_norm.weight mean 0.9628, std 0.0651
```

This runner's GGUF iterator already applies `raw - 1` for those norm parameters
before `GemmaPlusOneRMSNorm` performs `1 + w`; its effective scale is therefore
the original raw value. No RMSNorm change was made.

The same official `config.json` exposes a direct, independent mismatch:
`output_gate_type` is `"sigmoid"`, while `parse_gguf_config()` was hard-coded
to `"silu"`. The GGUF format does not include this architecture value. The
adapter now uses sigmoid and its metadata regression passes:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4_exp.py tests/models/test_qwen4exp_gguf.py \
  tests/models/test_qwen4exp_gguf_experts.py \
  tests/moe/test_fused_moe.py::test_fused_topk_falls_back_when_optional_triton_kernel_rejects_geometry -q
20 passed in 7.68s
```

The next required experiment is a clean server restart and the identical
thinking-disabled request. Only a coherent answer permits speed/context
benchmarking.

## 2026-08-28 — First valid Qwen3.8 FP16/Q4_K_M generation

After changing the Qwen4 GGUF GDN output gate from the incorrect SiLU default
to the official sigmoid, the exact same thinking-disabled smoke prompt became
coherent:

```text
prompt: Reply with exactly: pong
response: pong
usage: 17 prompt tokens, 2 completion tokens
HTTP 200, first valid request: 56.656689 s
HTTP 200, same-process repeat: 34.992677 s
server prefill metric on both: 0.26 input tok/s
```

This is the first quality-valid point for this runner: Qwen3.8 Q4_K_M executes
through FP16 GDN, QSA, MRoPE, native GGUF projections, PLE mmap and the
file-backed expert LRU on RTX 2070 SM75. The repeated answer is also correct.

The timing is **not** a decode throughput benchmark: the generation ends after
two tokens, and every 17-token request reports zero prompt-cache reuse because
the Qwen4 experiment currently uses the required naive cache. It does show that
cold-JIT/initial-cache work drops from 56.66 s to 34.99 s, while the remaining
dominant cost is still routed expert fetch/prefill. Long-context and sustained
decode measurements may now proceed from this commit.

## 2026-08-28 — Initial context-scaling probe and host-side stall

All requests below used the same valid FP16/Q4_K_M server configuration,
thinking disabled, `max_tokens=1`, temperature 0 and an `x ` repeated user
payload. Prompt-token count is the API's reported count, not an estimate.

| prompt tokens | completion | result | wall time / runtime observation |
| ---: | ---: | --- | --- |
| 28 | 1 | completed, coherent first token `The` | 30.185 s; prefill 0.23 tok/s |
| target 56 | — | cancelled after >5 min | no prefill completion; GPU 0%, scheduler CPU ~50%, API still reported active=1 |

The second request is not treated as a throughput datapoint. It is a
pathological host-side stall: the active request survived client cancellation,
the GPU remained idle with 7.07 GiB allocated, and the scheduler had to be
stopped to release it. At the time, RAM had 17 GiB available but pre-existing
system swap usage was 22 GiB; this does not prove swap was the sole cause.

The evidence does prove that the present synchronous file-backed expert/PLE
preparation path is not safe to extrapolate to 1K/16K/64K contexts. The server
was shut down and its remaining frontend process explicitly terminated after a
graceful shutdown left it waiting on a dead worker; VRAM returned to 9 MiB.
Future context benchmarking must first add request cancellation/timeout handling
and profile the CPU/mmap/Pinned-DMA stages instead of treating this as a GPU
throughput limit.

## 2026-08-28 — Batched Qwen4 file-backed expert-copy correction

Before changing the runtime, a direct real-weight microprofile of ten selected
experts from each of the three layer-0 banks measured the transfer mechanism,
with CUDA synchronized after each trial:

| transfer mechanism | selected rows | best steady-state time |
| --- | ---: | ---: |
| former per-row copies | 10 × 3 banks | 2.0979 s |
| gather once + scatter once per bank | 10 × 3 banks | 0.6009 s |

This is a **3.49× reduction** for the measured selected-expert transfer
primitive. It is not yet an end-to-end token/s claim: a real prefill also
contains PLE lookups, GDN/QSA, router work and all decoder layers.

The runtime now applies the batched path only if a bank's source and cache rows
have equal shapes. Otherwise it keeps the previous prefix-copy semantics.
Regression and focused architecture suite after the change:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/models/test_qwen4_exp.py tests/models/test_qwen4exp_gguf.py \
  tests/models/test_qwen4exp_gguf_experts.py tests/moe/test_fused_copy.py \
  tests/moe/test_fused_moe.py::test_fused_topk_falls_back_when_optional_triton_kernel_rejects_geometry -q
27 passed in 8.62s
```

Next measurement: restart the server from this commit, repeat the known 28- and
56-token probes, and only then extend the context ladder. The entry deliberately
does not claim a context-speed win until that end-to-end evidence exists.

## 2026-08-28 — 56-token JIT root cause and dynamic LRU admission

The 56-token request did not reach GPU execution before its 180-second client
timeout. Process inspection showed the scheduler worker waiting for one
single-core `ptxas` child, not blocked on NVMe or memory pressure:

```text
ptxas --gpu-name sm_75 /tmp/tmp1xqcu6bo.ptx
PTX size: 9.7 MiB, 188166 lines
entry point: _lru_ensure_kernel
```

That entry is flashlib's sequential LRU-admission strategy. Qwen's routed
prefill passes a `[tokens, top_k]` expert-ID tensor, so at 56 tokens it can
present 560 IDs; the static victim loop makes compilation proportional to that
shape. The cancelled old server was stopped after the client timed out, with
VRAM returning to 9 MiB.

The replacement applies only to `qwen4_gguf` file-backed layers: it calls
FreeToken's dynamic hybrid LRU kernel with `max_fetch = cache_size`, which is
equivalent to ordinary all-miss admission whenever the normal LRU capacity
precondition holds. Its CPU reference and GPU implementation are already
cross-tested in `tests/moe/test_hybrid_fetch.py`.

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest \
  tests/moe/test_hybrid_fetch.py tests/models/test_qwen4_exp.py \
  tests/models/test_qwen4exp_gguf.py tests/models/test_qwen4exp_gguf_experts.py \
  tests/moe/test_fused_copy.py \
  tests/moe/test_fused_moe.py::test_fused_topk_falls_back_when_optional_triton_kernel_rejects_geometry -q
33 passed in 13.20s
```

No end-to-end 56-token result is claimed yet. The next server run is the first
validation that the dynamic GPU kernel compiles quickly and preserves Qwen's
actual routed prefill behaviour.

## 2026-08-28 — Dynamic-admission context ladder (provisional)

With `bbdba12`, no `ptxas` process was created for the 56-token request. The
server completed the following synthetic `x ` prompt probes (temperature 0,
thinking disabled, one completion token):

| API prompt tokens | server prefill tok/s | result |
| ---: | ---: | --- |
| 28 | 0.32 | HTTP 200 |
| 56 | 0.94 first / 0.73 repeat | HTTP 200; former version timed out while compiling LRU |
| 128 | 1.56 | HTTP 200 |
| 256 | 2.71 | HTTP 200 |
| 512 | 5.90 | HTTP 200 |
| 1024 | 11.56 | HTTP 200 |
| 1536 | 13.94 | HTTP 200 |
| 2047 | 13.99 | HTTP 200 |

These figures establish that the static-LRU JIT stall is removed and that wider
prefill batches use the GPU better. They are **not final quality benchmarks**:
the 256-slot MoE cache can be smaller than the distinct expert set of a long
prefill. The subsequent capacity-safe chunking correction must be used for the
final ladder.

## 2026-08-28 — Detailed 2K file-backed prefill profile

A 20-second sample during a 2047-token prefill on the RTX 2070 Mobile measured:

| metric | observed value |
| --- | ---: |
| scheduler-worker CPU | 22.80 CPU-s / 20 wall-s = 114% of one core |
| worker `read_bytes` | 6.49 GB = 324.5 MB/s |
| NVMe read sectors | 6.68 GB = 334.0 MB/s |
| system major page faults | 20,016 = ~1,001/s |
| GPU utilisation | 1–81%, ~26% arithmetic sample mean |
| GPU clock | mostly 1215 MHz, briefly 1590 MHz |
| GPU temperature | 74–78 C during sample |

The same NVMe read-only direct-I/O test reached about 1.50 GB/s for a 512 MiB
GGUF shard slice (0.341 s). The live path therefore receives only ~22% of the
drive's sequential-read ceiling. The missing throughput is explained by
mmap-major-fault/random expert-row access and active swap/I/O wait, not a
compute-bound GPU kernel. CPU utilisation was also far below machine-wide
saturation; the scheduler coordinates one dominant host thread while the GPU
waits between bursts.

Next optimisation direction: retain/prefetch the selected file-backed expert
rows in a bounded host staging cache and overlap their H2D copy with GPU work.
MTP remains a later decode-only project; it cannot improve this prefill trace.

## 2026-08-28 — Capacity-safe Qwen3.8 prefill ladder (warm expert cache)

Commit `a28c68e` adds the required route-capacity boundary: every Qwen GGUF
MoE prefill fragment contains at most `moe_cache_size / top_k = 25` tokens, so
all possible ten-expert routes have a valid GPU-cache slot.  This replaces the
earlier provisional ladder for quality/performance comparisons.

The server was run with `--moe-backend offload --moe-cache-size 256`,
`--dtype float16`, `--max-seq-len-override 2048`, `temperature=0`, `seed=42`,
one completion token, and the same synthetic `x ` user payload.  The table
uses the API/server-reported prompt length, not the requested character count.
All calls returned HTTP 200.

| prompt tokens | server prefill tok/s | note |
| ---: | ---: | --- |
| 128 | 1.22 | first capacity-safe probe |
| 296 | 1.17 | tokenizer-expanded synthetic payload |
| 512 | 4.57 | capacity-safe |
| 1024 | 9.70 | capacity-safe |
| 2047 | 12.19 | capacity-safe maximum for this server configuration |

These are **warm-LRU, sequential-process measurements**: the 256 expert slots
were deliberately retained between requests, while KV prompt reuse remained
zero.  They measure the steady agent-like case, not cold-start latency.  A
future comparison must either clear/rebuild the MoE LRU before every row or
explicitly retain this same warm-up protocol.

### 1K capacity-safe runtime profile

During the 1024-token request, the actual scheduler worker (PID 679044, not
the lightweight HTTP parent) was sampled for 20 seconds:

| metric | observed value |
| --- | ---: |
| worker CPU time | 49.01 CPU-s / 20 wall-s = 245% (about 2.45 cores) |
| worker `read_bytes` | 12.07 GB = ~575 MiB/s |
| NVMe read sectors | 12.12 GB = ~578 MiB/s |
| system major page faults | 34,068 = ~1,703/s |
| GPU utilisation | 1–45%, ~25% arithmetic sample mean |
| GPU memory | 7.35 GiB / 8.00 GiB |
| GPU temperature | 76–80 C |

The worker and NVMe therefore improved relative to the prior provisional 2K
trace, but the GPU is still starved between short compute bursts.  The limiting
pipeline remains page-faulted expert rows plus host-to-device staging, not
FP16 arithmetic throughput.  Other active user processes (not stopped for this
measurement) are a background-noise caveat.

## 2026-08-28 — 16K BF16-KV allocation boundary

The same capacity-safe Qwen4 GGUF server was restarted with
`--max-seq-len-override 16384 --num-tokens 16384`, FP16 compute, BF16 KV,
256 MoE slots and `cache_type=naive`. Startup itself succeeded:

```text
Allocating 16384 tokens for KV cache, K + V = 0.42 GiB
Free memory after initialization: 0.71 GiB
```

A deterministic 16,383-token synthetic prompt then failed at the first GDN
prefill convolution, before an HTTP response, with:

```text
torch.OutOfMemoryError: Tried to allocate 160.00 MiB
100.62 MiB free; 7.35 GiB allocated by PyTorch
```

Thus 16K is a **KV-allocation success but end-to-end prefill failure** at
`moe_cache_size=256`: the remaining 0.71 GiB reported after setup is not all
available to the forward pass once allocator/accounting and GDN activation
workspace are considered. The scheduler worker exited; its waiting frontend
and the outstanding local curl were terminated cleanly, and GPU usage returned
to 9 MiB.

This establishes the next test direction. Do not reduce the cache to ten slots
merely because Qwen routes ten experts per token: that would serialise prefill
into one-token chunks and destroy expert reuse. First reserve bounded
activation headroom by moderately reducing the MoE LRU, and expose the
already-implemented QSA `tq4-nc` KV storage through the CLI so 64K/112K can be
tested without BF16 KV consuming the remaining VRAM.

## 2026-08-28 — TQ4-NC QSA KV: live validation and 16K workspace limit

The QSA pool already implemented packed `tq4-nc` storage, but the server CLI
did not expose it. Commit `f5341ac` adds the explicit CLI choice; commit
`54bfaa8` adds a CUDA regression and fixes a Triton 3.6 compilation defect
where the TQ4 store kernel read a Python global epsilon.

The exact Qwen4 Q4_K_M server configuration, with 256 MoE slots and 16,384 KV
tokens, now starts with:

| KV format | reported 16K K+V allocation | free VRAM after initialisation |
| --- | ---: | ---: |
| BF16 | 0.42 GiB | 0.71 GiB |
| TQ4-NC | 0.14 GiB | 1.00 GiB |

Thus TQ4-NC reduces this model's QSA KV allocation by about threefold. The
first live QSA request previously crashed at Triton compile time; the new CUDA
test reproduces that exact JIT invocation on SM75 and the focused suite passes
after the fix. A thinking-disabled OpenAI request returned the exact expected
answer:

```text
prompt: Reply with exactly: pong
reasoning_effort: off
response: pong
HTTP 200; 17 prompt tokens, 2 completion tokens
```

The default-thinking repeat is not a quality failure but demonstrates the
runtime cost: its 32-token cap contained the coherent reasoning that it must
answer `pong`, then stopped before the final answer; the decode log settled
around 0.3 tok/s after first-JIT. This configuration is therefore currently a
long-context feasibility path, not a recommended agent decode configuration.

Two end-to-end 16,383-token TQ4 attempts did **not** complete:

| MoE slots | failure point | allocation requested | free at failure |
| ---: | --- | ---: | ---: |
| 256 | Qwen hyper-connection mix | 160 MiB | 166.62 MiB |
| 192 | GDN chunk-gated-delta FLA state | 192 MiB | 58.62 MiB |

The 192-slot server did increase post-init free VRAM to 1.13 GiB, but the
later FLA temporary allocation still exceeded its peak headroom. These are not
KV-allocation failures and do not justify reducing the LRU to ten slots: that
would cap routed prefill fragments at a single token. The next experiment must
determine whether FreeToken's existing `max_extend_tokens` scheduler chunking
preserves Qwen PLE/GDN/QSA state; if so it can bound activation workspace
without discarding a useful expert cache.

## 2026-08-28 — 16K stateful scheduler-chunking and thermal boundary

Source review and the existing scheduler/QSA/Qwen geometry suite establish that
`--max-prefill-length 2048` makes a 16,383-token request into eight sequential
forwards while retaining one request table row, GDN state, QSA pending
compression state and PLE convolution state. The relevant local verification
passed before the live trial:

```text
tests/scheduler/test_scheduler_chunked_prefill.py
tests/kvcache/test_qsa_pool.py
tests/models/test_qwen4_exp.py
tests/models/test_qwen4exp_gguf.py
21 passed in 16.75s
```

Live configuration: FP16 Q4_K_M, TQ4-NC KV, 16,384 KV tokens, 256 expert
slots, `--max-prefill-length 2048`, naive cache, one request. The first three
real continuation chunks completed without OOM:

| completed scheduler chunk | tokens | reported prefill tok/s |
| ---: | ---: | ---: |
| 1 | 2048 | 11.26 |
| 2 | 2048 | 36.93 |
| 3 | 2048 | 20.42 |

The request was intentionally stopped before completion because the mobile RTX
2070 reached 91–92 C. `nvidia-smi` reported `SW Thermal Slowdown: Active`, a
780 MHz SM clock (versus about 1215 MHz after cooling), and the GPU has a 94 C
slowdown / 99 C shutdown threshold. This is therefore **not** a full 16K speed
result; it proves the stateful chunking path and records three partial chunks.
After a 30-second idle cooldown the GPU returned to 77 C, 1215 MHz and 9 MiB
used. Future long benchmarks must use a stable thermal/power envelope or cool
between chunks; otherwise apparent runtime regressions are thermal throttling,
not FreeToken performance.

## 2026-08-28 — 16K TQ4-QSA full end-to-end prefill (normal cooling)

This entry supersedes the operational conclusion of the preceding partial run:
the later repeat used the laptop's normal cooling configuration with **no GPU
power, clock, or thermal limit imposed by the benchmark**.  The model stayed at
about 1215 MHz during the useful part of the run and completed normally.

Configuration: `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`, FP16 compute,
`tq4-nc` KV, 16,384-token KV budget, 256 MoE LRU slots, naive cache,
one request, and `--max-prefill-length 2048`.  The request used a deterministic
synthetic 16,331-word prompt with `reasoning_effort=off`, `temperature=0`,
`seed=42`, and `max_tokens=1`.  The server returned HTTP 200 with 16,343 prompt
tokens and one completion token.

| scheduler chunk | tokens | server-reported prefill tok/s |
| ---: | ---: | ---: |
| 1 | 2048 | 2.15 |
| 2 | 2048 | 42.95 |
| 3 | 2048 | 43.09 |
| 4 | 2048 | 43.47 |
| 5 | 2048 | 43.22 |
| 6 | 2048 | 42.24 |
| 7 | 2048 | 36.31 |

The last 2,007-token residual log line reported `64149.15 tok/s`.  That is an
accounting artefact at scheduler completion (the request is already removed
from the timed prefill batch), not a performance measurement, and is excluded.
The first 2,048-token value includes cold per-request setup/JIT and is likewise
kept separate from the steady chunks.  The usable steady-state interval is
about 42--43 tok/s, with a late-chunk decline to 36.31 tok/s.  The request's
semantic smoke test remains separately verified by the exact `pong` response;
the synthetic one-token completion is only a speed workload.

Observed during this full run: approximately 83--85 C, 95--103 W and 1215 MHz;
after completion the card cooled to 68 C and idled.  No fixed clock or power
limit was applied.  A plot point is deliberately not appended to
`slices.jsonl` yet: the external wall-clock JSON emitted by this manually
interrupted/re-attached control command was not retained, so an aggregate tok/s
would be invented.  The next automated context point must use the benchmark
runner, which writes both its raw JSON and the slice row atomically.
