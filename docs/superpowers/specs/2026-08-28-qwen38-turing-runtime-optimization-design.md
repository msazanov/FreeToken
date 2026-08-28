# Qwen3.8 Flash Next Q4_K_M on RTX 2070 — Runtime Optimisation Design

Status: proposed for review. This document defines the first optimisation cycle;
it does not authorise implementation by itself.

## Objective

Make the existing execution-valid Qwen3.8 Flash Next Q4_K_M path useful as a
single-agent DeepSeek Harness runtime on this exact machine. The current
evidence consists of one 16K synthetic speed completion and one short
exact-output smoke. Long-context answer quality and a complete 64K request are
not established yet.

- RTX 2070 Mobile, Turing SM75, 8 GiB VRAM;
- Intel i7-8750H, 6 cores / 12 threads, AVX2;
- 32 GiB DDR4;
- 1 TiB NVMe;
- FreeToken fork branch `feat/qwen4exp-gguf-turing`;
- unchanged AtomicChat `AD-4.27bpw-Q4_K_M-M64` weights.

The first cycle changes scheduling, caching, I/O and state handling only. It
does not requantize experts, introduce mixed-precision fallback weights, enable
MTP, or replace FreeToken with llama.cpp. The quality target is therefore the
current Q4_K_M model, not a smaller approximation.

The practical target is not merely a high isolated token/s number. It is a
coding agent that can ingest a long repository context once, reuse it across
turns, and decode without making NVMe reads the serial critical path.

## Measured baseline and corrected geometry

The current runtime has already established these facts:

| Property | Qwen3.8 Q4_K_M | Ornith 1.5 35B Q4_K_M |
| --- | ---: | ---: |
| MoE layers | 48 | 40 |
| routed experts/layer | 512 | 256 |
| active experts/token/layer | 10 | 8 |
| layer-expert pairs | 24,576 | 10,240 |
| expert activations/token | 480 | 320 |
| measured VRAM expert slots | 256 | 1,429 |
| global expert residency | 1.04% | 13.96% |
| mean slots/layer | 5.33 | 35.73 |
| active parameters/token | about 6B | about 3B |

One Qwen layer-expert record occupies 2,329,600 packed bytes across gate, up
and down. Qwen therefore has both 1.5 times as many routed expert activations
per generated token and a much smaller resident fraction. Exact Ornith speed is
not a physically sound promise even after software overhead is removed.

The avoidable gap is nevertheless large:

- a complete 16,343-token request, using 2,048-token scheduler chunks, TQ4-NC
  KV and 256 expert slots, sustained 42–43 prefill tok/s in the useful steady
  chunks;
- a 64K configuration allocated its 0.57 GiB TQ4-NC KV and started correctly;
- 2,048-token chunks OOMed immediately in GDN/FLA workspace;
- 1,024-token chunks sustained about 8.8–10.0 prefill tok/s and reached roughly
  45K tokens before a 78 MiB QSA score allocation OOMed with 84.81 MiB free;
- the 1K runtime profile used about 2.45 CPU cores, read about 575 MiB/s from
  NVMe, incurred about 1,700 major faults/s and averaged about 25% GPU
  utilisation;
- direct sequential reading from the same NVMe reached about 1.50 GB/s.

The approximately 9.4 tok/s value is a partial 64K boundary observation up to
roughly 45K tokens, not a completed 64K benchmark. The raw one-second sampler
stream was not retained. It is an engineering control without confidence
intervals, not an acceptance denominator.

This combination says that the packed expert GEMV is not the primary prefill
bottleneck. The current token-major file-backed path creates page-faulted,
repeated expert traffic and leaves the GPU waiting. Long-context QSA workspace
is an independent correctness/capacity blocker.

## Prior-art audit

The design deliberately adopts mechanisms that already have public evidence.
Performance numbers are treated as evidence only for the hardware/model on
which they were measured.

| Source | Demonstrated result or mechanism | Decision for this fork |
| --- | --- | --- |
| [FreeToken PR #257](https://github.com/FlashML-org/FreeToken/pull/257) | Official text-only Qwen3.8 implementation with bounded sparse QSA, capture-safe PLE/GDN/QSA decode, hybrid-radix state snapshots and CUDA graphs; 36 tok/s on RTX 4090 and 65 tok/s on RTX 5090, but with about 111 GiB pinned host RAM | Use as the semantic/runtime reference. Port components, not its impossible 111 GiB memory policy. |
| [FreeToken PR #232](https://github.com/FlashML-org/FreeToken/pull/232) | Alternative Qwen3.8 path, including 250K context validation. A matched 4090 report measured 14.896 tok/s offload versus 12.753 tok/s hybrid despite the bandwidth profiler recommending hybrid | Do not enable hybrid from a microbenchmark. Require an end-to-end A/B on the i7-8750H. |
| [FreeToken PR #226](https://github.com/FlashML-org/FreeToken/pull/226) | Earlier exact model path capped at 2K because sparse QSA was absent; first cold request took 212 seconds | Not an implementation base for long context. |
| [FreeToken PR #231](https://github.com/FlashML-org/FreeToken/pull/231) | Exposes realized miss rate, per-layer misses, routing working set, entropy and stationary-oracle hit rate. Ornith measured 74.1% realized versus 76.4% oracle | Port this telemetry before changing eviction policy. |
| [FreeToken issue #174](https://github.com/FlashML-org/FreeToken/issues/174) | On DSV4, global LRU achieved only 8–18% hits while a frequency oracle reached 46% at the same capacity | Qwen policy work is justified only if its own realized/oracle gap is material. |
| [FreeToken issue #111](https://github.com/FlashML-org/FreeToken/issues/111) | Repeated full expert-bank streaming can dominate each prefill chunk, including small appended prefixes | Prefill and decode must use different expert policies. |
| [FreeToken PR #199](https://github.com/FlashML-org/FreeToken/pull/199) | Exact-geometry GGUF pools improved decode 23.13% at the same byte budget on Laguna-S | Budget cache in bytes and preserve native row geometry; do not count nominal slots that are padded differently. |
| [FreeToken issue #239](https://github.com/FlashML-org/FreeToken/issues/239) | RTX 2050 4 GiB sustained about 20–21 decode tok/s on Qwen3.6 after shrinking the prefill buffer, but failed around 600 prompt tokens in GDN/FLA workspace | Treat prefill workspace and decode residency as separate VRAM demands. |
| [Qwen hardware-fit draft](https://github.com/MaxKerkula/FreeToken/pull/1) | Implements fixed-record file-backed experts, but its real-geometry analysis requires 2.30–4.59 GB/s at plausible miss rates and its reader remains synchronous | Reuse its `ExpertSource` boundary and accounting ideas; do not copy the synchronous SSD path or claim the sidecar solves bandwidth. |
| [Flash-MoE paper](https://arxiv.org/abs/2601.17063) and [layer-major implementation notes](https://github.com/Anemll/anemll-flash-llama.cpp/blob/master/docs/flashmoe-layer-major-dedup-single-sidecar.md) | For prefill, route a whole layer slab, deduplicate experts, sort by file offset, load each unique expert once, and overlap next-stage reads with current compute. The paper reports only about 7% end-to-end gain from its learned cache on Qwen3-30B, despite larger gains on another model | Adopt layer-major expert-major prefill. Treat learned eviction as a later, trace-gated option. |
| [S-MoE](https://github.com/melasistema/s-moe) | Independently implements aligned expert files, asynchronous unbuffered reads, a retained LRU ring and layer-major deduplicated prefill | Confirms the same I/O architecture independently; its Metal kernels are not reusable on CUDA. |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | Provides SSD/RAM/GPU expert tiers, dedicated I/O threads, cache policies, route tracing and priority-banded prefetch | Reuse concepts and metrics, not its PyTorch/HF execution stack. Its public code differs from the paper's high-performance runtime. |
| [Who Should Own the Expert Cache?](https://arxiv.org/abs/2608.12103) | Under an equal-memory wall on GH200, kernel page-cache LRU matched a same-domain frequency oracle closely; lookahead advice added only 0.3% | Do not duplicate many GiB into a pinned user-space L2 by default. Measure page-cache residency and use explicit L2 only when our trace disproves the kernel-managed option. |
| [SpecPrefetch](https://arxiv.org/abs/2607.24787) | A transfer-only route predictor preserves model outputs and improved decode by up to 20% on Snapdragon | Keep prediction outside the correctness path and defer it until the native-router trace establishes achievable recall. |
| [Qwen3.8 official repository](https://github.com/QwenLM/Qwen3.8-Flash-Next) and [DGX mmap PLE implementation](https://github.com/blazux/qwen3.8-Flash-DGX) | PLE is intentionally host/off-accelerator storage. The mmap implementation deduplicates and sorts row IDs and uses bounded worker parallelism | Keep the Q5_1 PLE on NVMe; optimize sparse row gathering instead of preloading the 38 GiB table. |

No public report was found for Qwen3.8 Flash Next on RTX 2070, Turing SM75, or
an equivalent 8 GiB Turing GPU. Upstream FreeToken officially targets Ampere
and newer. Our existing SM75 kernels and measurements remain necessary original
work rather than a configuration copied from an established result.

## Architecture decision

The runtime becomes phase-aware. Prefill is an expert-major streaming problem;
decode is a latency-sensitive cache problem; prefix reuse is a model-state
problem. One global token-major LRU cannot solve all three.

The existing GGUF readers, mixed-IQ kernels, TQ4-NC KV implementation and Qwen
correctness fixes remain the model-provider layer. The official PR #257 becomes
the reference for Qwen state semantics, QSA and hybrid-radix behavior. A blind
merge is rejected: our branch is 103 commits ahead of the common base and the
official PR is one large commit with a different weight provider. Each adopted
component needs a focused differential test against the current GGUF path.

PR #257 is therefore ported semantically, not merged or cherry-picked as a
model-wide change. Its QSA scoring/top-k tests, declarative slot-state semantics
and scheduler alignment are allowed references. Its unpacked equal-dtype paged
attention, pinned PLE provider, NVFP4/block-FP8 weight path and branch topology
are out of scope for the local GGUF/TQ4 port.

### 1. Measurement comes before policy

Inventory and wire the telemetry already present in `EngineConfig` and
`OffloadMoECache`; port only the missing CLI/report surface from PR #231. Add a
separate bounded route/copy trace that records, for each phase and MoE layer:

- selected expert IDs and routing weights;
- unique experts per prefill slab and references per expert;
- L1 hits, misses, evictions and bytes transferred;
- page-cache residency samples, major faults and physical NVMe bytes;
- H2D bytes and exposed wait time;
- GPU compute time, I/O overlap time and queue depth.

The trace format must be bounded and replayable offline. Raw hidden states and
prompt text are not required; layer, token ordinal, expert IDs and timing are
sufficient for cache simulation.

An offline simulator initially evaluates the same trace under byte-equal
policies:

- current global LRU;
- one static per-layer protected policy;
- global byte-budget LFU;
- Belady's finite-trace upper bound;
- capacity sweeps for VRAM L1 and RAM/page-cache L2.

The existing `oracle_hit_at_slots` statistic is renamed
`static_per_layer_topC_hit`: it is a stationary top-C frequency score, not
Belady and not a global optimum. Decayed frequency, segmented LRU and ARC remain
deferred until the simple controls show a material policy gap.

The eviction policy changes only if the measured oracle gap is meaningful. If
global LRU is already within five percentage points of the best trace oracle,
policy complexity is rejected and capacity/I/O remains the target.

### 2. Remove the 64K memory failure first

The local scorer is already row-chunked at a nominal 128 MiB score budget. It
still materializes a per-head FP32 dot tensor and does not bound the total live
score, reduced-logit, top-k, block-ID, expanded-index and allocator footprint.
The failure is therefore not fixed by adding another Python row loop.

Port PR #257's fused head-reduced paged scorer and split/merge top-k semantics
into the existing page-size-4 logical-selection layer while retaining the local
scale-aware TQ4-NC physical attention path. Validate equal 128 MiB behavior on
short contexts first, then sweep 8, 16 and 32 MiB under complete peak-memory and
latency telemetry. Test 48 MiB only if 32 MiB remains score-launch-bound and
measured headroom proves it safe; at 64K it needs the same two score passes as
32 MiB while leaving much less allocator margin.

Scratch is considered bounded only when one accounted arena covers logits,
top-k scratch, block IDs, expanded indices and q-index. Upstream's prefill
helper may allocate a fresh `torch.empty`, so reuse is a requirement of this
port rather than an upstream guarantee. GDN/QSA prefill remains scheduler-
chunked at 1,024 tokens until a fresh sweep proves a larger chunk safe.

At 64K with page size 4, the compressed score width is 16,384 and one FP32
score row is 64 KiB. The lower-bound workspace geometry, before live model
activations, allocator slack and TQ4 attention, is:

| Score budget | Rows/chunk | Score passes/QSA layer | Lower-bound temporary tensors |
| ---: | ---: | ---: | ---: |
| 8 MiB | 128 | 8 | about 19.3 MiB |
| 16 MiB | 256 | 4 | about 29.5 MiB |
| 32 MiB | 512 | 2 | about 50.0 MiB |
| 48 MiB | 768 | 2 | about 70.5 MiB |

Any budget that holds at least one row can be exact; smaller budgets trade more
score/top-k passes for capacity. The 48 MiB option is dominated initially
because it does not reduce the pass count relative to 32 MiB at 64K.

The first gate is a complete deterministic 64K request with no OOM. This stage
is a capacity fix, not a speed claim.

### 3. Expert-major layer-major prefill

The GGUF source is already expert-major and contiguous. This section changes
execution scheduling, not tensor storage. The first reference implementation
may reuse the existing mixed-IQ MMVQ kernels. A fused grouped kernel is a
separate gated optimisation because the current kernel still launches by
`(token, route)` and invokes gate, up and down separately.

For one scheduler slab and one MoE layer:

1. Compute the native router for all tokens in the slab.
2. Flatten `(token, top-k)` references and group them by expert.
3. Sort unique experts by their physical GGUF offsets within gate, up and down.
4. Process experts in bounded stage batches, initially 8, 16 and 32 experts.
5. Gather source bytes into one of two reusable pinned host buffers.
6. Transfer the current stage on a dedicated CUDA stream while the CPU prepares
   the next stage.
7. Gather the token rows for each resident expert, execute its packed mixed-IQ
   projection, and scatter weighted outputs back to the layer result.

The actual banks are not homogeneous Q4_K: gate/up include IQ3_S or IQ2_S and
down uses IQ4_NL. Any future grouped kernel must cover that mixed type matrix,
the three separate projections, prequantized inputs and token maps without
materializing BF16 expert copies or issuing a second host copy.

Let `U(L,S)` be the number of unique routed experts in layer `L` for reuse slab
`S`. The minimum one-copy expert payload is
`record_bytes * sum_L U(L,S) / S`. At `S=1024` and full `U=512`, it is 53.32
MiB/token, versus the observed 61.17--62.77 MiB/token. A 50% reduction from the
observed physical bytes requires mean `U <= 293` or equivalent cross-slab
RAM/VRAM residency. No fixed physical-byte reduction is therefore assumed
before route and page-residency measurement.

Initially the MoE reuse slab equals the 1,024-token scheduler slab. A reuse slab
of 2,048 or 4,096 requires an explicit layer-wise model-forward design that
retains the larger hidden-state slab while internally microchunking QSA/GDN. It
is not an independent configuration knob and must prove state and memory
correctness before being used for performance.

Prefill stages use transient scratch and do not automatically evict the decode
hot set. Only frequency evidence gathered during prefill may be admitted into
the protected decode set at the phase transition.

The first implementation continues to read the existing GGUF shards. The
source abstraction first gains behavior-neutral `shard_id`, `file_offset`,
`row_bytes`, `expert_stride`, bank name and layer metadata. Gate, up and down
receive independent physical schedules because they need not share shard
ordering. A new
sidecar, `O_DIRECT` and `io_uring` are explicitly deferred until the mmap
backend has measured unique-expert, page-fault and sequential-read efficiency.
The source abstraction must nevertheless expose physical offsets and batched
reads so a direct-I/O backend can be added without changing MoE scheduling.

### 4. Decode cache: protected per-layer set plus transient reserve

At 256 slots, strict global LRU lets successive layers evict one another. Every
candidate geometry must satisfy `48P + T <= 256`, where `P` is the protected
slots per layer and `T` covers the maximum number of unique active experts in
one supported decode layer. Top-10 only bounds `T` for batch size one, one
decode token and no MTP/speculation.

The initial candidate layout reserves:

- five protected slots for each of 48 layers: 240 slots;
- sixteen shared transient slots: a candidate only for the guarded batch-size-
  one, one-token, no-speculation mode.

Protected slots survive the full 48-layer sweep and are selected from measured
frequency. Transient slots serve the current layer's cold tail and are reused
by the next layer. The first implementation is hard-guarded to one running
request, one decode token, no MTP/speculation and a protected set frozen for the
request. Dynamic promotion requires an explicit spare/swap invariant rather
than silently evicting a protected slot.

The exact split remains a simulator output, not a hard-coded belief. Capacity
variants include `4×48 + 64`, `5×48 + 16`, and larger caches made possible by
phase-specific workspace release. The current global LRU remains available as
an A/B control.

After routing, asynchronous miss transfer may overlap with the always-executed
shared expert. Cross-layer prediction is not required for this first overlap;
the model's native router remains authoritative and output semantics do not
change.

### 5. RAM and NVMe tiers

The default L2 is the existing file-backed mmap plus the Linux page cache.
`mincore` is only a sampled residency indicator, never hit attribution.
Physical I/O is measured primarily through process `read_bytes`, cgroup/device
counters and major faults. No `MADV_DONTNEED` is issued in the default path:
advisory page-aligned `posix_fadvise`/madvise experiments come last and are
rejected if they increase cross-slab reads or faults. System-wide caches are
never dropped.

A small pinned staging pool, not a duplicate model cache, enables asynchronous
H2D. Stage size and pool size are one geometry: double-buffered stage 8 needs
at least 36 MiB, stage 16 at least 72 MiB and stage 32 at least 143 MiB, plus
alignment and metadata reserve. Host pinned bytes and device scratch are
reported separately and the total is included in the 32 GiB memory budget.

An explicit RAM expert cache is a gated experiment. It is tested at 2, 4 and
8 GiB only if the route replay predicts a material disk-miss reduction and the
system remains out of destructive swap pressure. It may store compact expert
records or lock selected mmap pages, but it must not silently coexist with an
equal duplicate in page cache. Page-cache-only and explicit-L2 runs use the
same total memory wall.

NVMe is L3. Reads are sorted and batched. A fixed-record sidecar is considered
only if three-shard scatter remains a measured limit after layer-major dedup;
the public Qwen hardware-fit draft already shows that a synchronous sidecar is
not sufficient.

### 6. PLE remains sparse and file-backed

PLE touches 16 packed Q5_1 rows per token, about 1.9 KiB of logical payload,
whereas one expert miss is about 2.33 MiB. The physical footprint can be much
larger because sparse rows fault whole pages and trigger readahead. PLE is not
allowed to consume 38 GiB of RAM merely to remove sparse faults.

Within each prefill slab, PLE row IDs are deduplicated and sorted, gathered by
a bounded worker pool sized for the six-core CPU, then dequantized into fixed
GPU buffers and scattered back into the exact original token/head order.
Packed payload, unique pages, physical bytes, faults and PLE wall time are
reported separately. Worker counts 1, 2, 4 and 6 are considered only after PLE
has a measured independent share; the 32-worker DGX setting is not copied.

### 7. Correct hybrid-radix state for the Harness

The highest-value agent optimisation is to avoid repeating a correct cold
prefill. Migration order is explicit: page-64 TQ4 storage parity; QSA logical
and physical row parity; chunked versus one-shot parity at unaligned cuts;
PLE convolution and n-gram sibling state; QSA pending state; snapshot/COW
tests; hybrid-radix; and only then CUDA graph. Partial migration must not expose
cached-token accounting before all recurrent state is restorable. Only after
differential tests pass may Qwen stop forcing `requires_naive_cache=True`.

The DeepSeek Harness validation has three distinct cases:

- byte-identical replay;
- normal assistant-history append;
- fan-out from the same system/tool prefix with a different user tail.

Usage-reported cached tokens must match actual TTFT reduction. Tool calls,
reasoning blocks and compaction summaries are included because a cache that
only works on synthetic strings does not satisfy the agent objective.

Whole-model CUDA graph decode is a separate final project, not a consequence of
page-64 or hybrid-radix success. The current
GGUF PLE and expert providers perform host-side file work. Graph capture is
enabled only after host staging is moved outside the captured compute region or
fixed-address buffers make that region capture-safe.

### 8. Phase-adaptive VRAM

Prefill and decode need different temporary memory. All QSA/GDN and I/O scratch
is bounded, preallocated and reusable. Pinned host staging cannot be overlaid
with VRAM expert slots. Only a preallocated device arena may change
interpretation at a request-safe phase barrier, without invalidating live KV,
GDN, QSA, PLE state or cache tags. The existing idle-only MoE-cache rebuild is
not used for a live prefill-to-decode transition.

This is evaluated after the fixed 256-slot policy because dynamic expansion can
hide a bad eviction design. The useful metric is additional protected slots and
decode tok/s, not merely lower reported free VRAM.

## Performance expectations and structural limits

Deduplication, offset-sorted reads and pinned overlap act on the same I/O
critical path and must not be multiplied as independent gains. Under the rough,
optimistic assumption that 25% of wall time is unavoidable non-I/O work, a 2×
I/O improvement yields about 1.60× end-to-end and even 1.50 GB/s versus the
observed 575 MiB/s yields about 1.86×. A 2× end-to-end result needs roughly a 3×
improvement in the exposed I/O component.

Planning ranges, not acceptance promises:

| Change | Realistic planning effect | Important qualification |
| --- | ---: | --- |
| Bounded/fused QSA | -15% to +10% throughput | Primarily makes complete 64K possible |
| Layer-major at reuse slab 1,024 | 0% to +35% end-to-end | Physical-byte headroom is likely only 0--15% without cache reuse |
| Sorted/pinned overlap | 0% to +20% inside the same I/O envelope | Do not multiply with the full layer-major gain |
| Grouped mixed-IQ kernel | 0% to +15% after I/O starvation falls | Current sampled GPU share is only about 25% |
| Protected decode cache | 0% to +15% | Requires a Qwen decode trace and material oracle gap |
| Hybrid-radix exact/append reuse | Large warm-TTFT reduction | Does not increase cold-prefill tok/s |
| Reuse slab 2,048/4,096 | Potentially +50% to +120% | Requires a new layer-wise execution design |

The resulting 64K planning bands are 8--10 tok/s after the QSA capacity fix,
10--15 tok/s for the lower-risk slab-1,024 stack with stretch around 17, and
15--21 tok/s only for a successful larger reuse slab plus overlap. These values
use the partial 9.4 tok/s boundary and must be replaced by completed matched
runs.

Qwen cannot be treated as Ornith with a better cache. It routes 480 versus 320
expert activations/token, has about 6B versus 3B active parameters, 48 layers
with 512 experts/top-10 versus 40 with 256/top-8, only 1.04% versus 13.96%
global expert residency, context-growing QSA, and a roughly 38 GiB sparse PLE
table. On 8 GiB SM75 and 32 GiB RAM, absolute Ornith cold-64K speed is not a
credible software-only target. The objective is to remove avoidable idle/I/O
and make cached agent turns useful without changing model quality.

## Minimum-risk experiment order

Each stage changes one variable and keeps the prior accepted path as a runtime
rollback. General artifacts include the exact git revision and dirty-diff hash,
model-shard hashes, CUDA/PyTorch/Triton/driver versions, allocator settings,
prompt token IDs, actual token counts, temperature, chunk/page/cache geometry,
kernel warm state and thermal state.

For 64K, run one correctness attempt before three matched A/B repetitions. For
1K and 16K, use five alternating control/candidate repetitions. Performance
comparisons are invalid when GPU clocks differ by more than 5% or thermal
throttling is active.

| Stage | One changed variable | Acceptance evidence | Abort and rollback |
| --- | --- | --- | --- |
| E0 | Reproducible runner and artifact manifest only | Repeat 1K/16K; retain the existing 64K OOM boundary with exact prompt tokens | Abort if raw artifacts or actual token counts differ; use the current runner |
| E1 | Wire existing MoE counters | Non-zero counters reconcile with route counts; output and timing remain within noise | Abort on output difference or more than 2% overhead; disable collection |
| E2 | Fused head-reduced QSA scorer at page-4/TQ4 and 128 MiB | Exact selected logical blocks and attention output within stated tolerance on short contexts | Abort on selection difference, NaN or more than 5% 16K regression; restore local `torch.mm` scorer |
| E3a-c | QSA workspace only: 8, then 16, then 32 MiB | Complete 48K/64K with score/top-k timing and full peak accounting | Abort on OOM/retry or insufficient margin for the next known allocation plus 64 MiB; keep the previous safe budget |
| E4 | Bounded route/copy trace | `U(layer,S)`, reuse distance, H2D copies and current-LRU/LFU/Belady replay | Abort above 3% overhead or on dropped events; keep counters only |
| E5 | GGUF physical-offset metadata only | Offsets match mmap ranges and existing outputs remain bit-identical | Abort if any view becomes a materialized copy; restore the old tensor metadata |
| E6 | Reference expert-grouped scheduler, stage 8, synchronous pageable input | Per-route and accumulated layer outputs pass differential tests; copy behavior is measured | Abort on numerical failure or more than 5% end-to-end regression; restore token-major execution |
| E7 | Pinned double buffer for stage 8 only | Exposed H2D wait and end-to-end time improve without RAM/swap pressure | Abort on growing swap/PSI, physical bytes or more than 5% wall regression; restore synchronous staging |
| E8a-b | Stage size only: 16, then 32 | Find a queue-depth optimum under the linked 72/143 MiB pool geometry | Abort without statistically significant gain or on memory pressure; retain stage 8 |
| E9 | Grouped mixed-IQ kernel only if E6/E7 become launch/quantize-bound | IQ2_S/IQ3_S/IQ4_NL differential suite and at least 5% end-to-end gain | Abort on any unsupported bank/type or smaller gain; retain existing MMVQ reference scheduler |
| E10a-b | MoE reuse slab only: 2,048, then 4,096; QSA/GDN microchunk stays 1,024 | Physical/H2D bytes per token fall while state and workspace remain correct | Abort on state divergence, activation OOM or more than 5% 16K regression; return to slab 1,024 |
| E11 | Protected/transient cache at total 256, batch size one | At least 256 Qwen decode tokens; miss/activation and p50/p95 latency beat global LRU | Abort if unique routes exceed transient capacity, a tag invariant fails or decode regresses; restore global LRU |
| E12 | TQ4 page size 64 with naive cache only | Storage, logical/physical selection and chunk cuts 1001/4096/4097 match page 4 | Abort on any chunk/one-shot divergence; restore page 4 |
| E13 | PLE/QSA sibling state in the state pool | Cold, exact replay, append, fan-out and COW tests match | Abort on stale state, output divergence or cached usage without TTFT gain; keep naive state |
| E14 | Hybrid-radix only | At least 95% usable prefix reuse with correct output and matching TTFT reduction | Abort on contamination or state divergence; restore naive cache |
| E15 | Page-cache advice only, with eviction advice last | Process/device bytes, faults, residency and wall time improve together | Abort on more than 5% physical-read/fault growth or lost cross-slab reuse; issue no advice |
| E16 | Preallocated device phase arena only | Request-safe transition preserves tags/state and improves decode | Abort if it requires idle-only rebuild or cold-starts the cache; keep fixed 256 slots |
| E17 | Whole-model CUDA graph only | Eager/replay differential passes with no host file operation or dynamic allocation in capture | Abort on dynamic addresses, file work or output difference; retain eager decode |

## Benchmark and quality contract

Every stage keeps a same-revision control and writes artifacts below
`benchmarks/results/`. Prefill, decode and cache reuse are never averaged into
one throughput number.

The benchmark matrix contains:

- cold 1K, 16K and 64K deterministic repository-context prompts;
- 112K only after 64K completes without OOM;
- one exact warm replay and one appended turn per context tier;
- at least 256 forced decode tokens for a stable decode window;
- a real repository-analysis/tool-call task through DeepSeek Harness;
- cold and warm expert/page-cache states labelled separately.

Each artifact records TTFT, per-chunk prefill, client and server decode,
realized/static-top-C/Belady expert hits, unique experts per layer slab, NVMe and H2D
bytes/token, major faults, CPU core use, GPU utilization/clock/power/temperature,
VRAM peak, RAM and swap pressure, QSA scratch peak and prefix cached tokens.

Qwen cache-policy experiments begin only after the QSA capacity fix produces a
stable window of at least 256 forced decode tokens. Ornith is a matched positive
control for telemetry and prefix reuse, not a raw throughput target. It uses the
same prompt construction, actual token counts, cold/warm protocol, decode
length, repetitions and thermal rules. Comparisons report activations,
misses/activation, bytes/activation and slots/layer alongside tok/s.

Runtime reordering can slightly change floating-point accumulation even with
identical weights. Greedy token parity alone is both too strict near a tiny
top-1 logit margin and too weak to localize a systematic error. Correctness is
therefore checked at four levels:

- exact router IDs and weights;
- exact QSA selected logical blocks;
- stated numerical tolerances for per-route, layer and final logits, including
  KL divergence and top-1 margin;
- a deterministic greedy corpus, perplexity/long-context anchors and valid
  reasoning/tool-call structure through the Harness.

The first-cycle performance gates are:

1. 64K completes and returns a valid answer without OOM.
2. Layer-major proceeds beyond its reference implementation only if matched
   end-to-end prefill improves at least 10% with a confidence interval excluding
   zero, 16K does not regress more than 5%, and measured H2D/copy reduction
   explains the gain. Physical-byte reduction is a reported outcome rather
   than a fixed 50% gate before route/page-cache measurement. A 2× result over
   the partial 9.4 tok/s boundary remains a stretch objective and requires a
   measured at least 3× improvement in the exposed I/O service or a validated
   MoE reuse slab larger than 1,024.
3. A decode-cache change is accepted only when matched end-to-end decode
   improves at least 10% and its realized hit rate moves toward the measured
   oracle; a hit-rate-only win is insufficient.
4. Hybrid-radix warm replay/append reports at least 95% prefix reuse at 16K and
   64K and produces a corresponding TTFT reduction, with correct state and
   output.

These are acceptance thresholds, not forecasts. Failure at a gate remains a
recorded result and selects the next branch of the design.

## Feature isolation and rollback

New behavior is independently selectable:

- routed token-major versus layer-major prefill;
- global LRU versus layer-aware protected/transient decode cache;
- page-cache-only versus explicit RAM L2;
- naive versus validated hybrid-radix cache;
- mmap versus a future direct-I/O expert source.

The current 16K execution-valid configuration remains the rollback path; it is
not a completed long-context quality baseline. No
experiment may delete or rewrite the existing Q4_K_M checkpoint, alter the
Ornith service, impose a GPU clock/power limit, or drop unrelated system caches.

## Deferred ideas

The following are technically plausible but outside the first quality-preserving
cycle:

- mixed Q2/Q4 copies for cold experts, as in HOBBIT;
- a trained route predictor or speculative cross-layer prefetch;
- CPU execution of cache-miss GGUF experts on the i7-8750H;
- the model's MTP head and multi-token verification;
- a repacked fixed-record expert sidecar with `io_uring`/`O_DIRECT`;
- quantizing or pruning the model beyond the current checkpoint.

Their priority is determined by the route trace and the measured residual
critical path, not by headline results from different hardware.
