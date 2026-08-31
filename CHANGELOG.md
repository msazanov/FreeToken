# RTX 2070 Fork Changelog

This changelog is append-only for experiments as well as code. Keep successful,
failed and inconclusive hypotheses; do not rewrite history.

## 2026-08-28

### Accepted — Qwen3.8 Flash Next GGUF base

- Added the text-only Qwen4Exp model descriptor for the local AtomicChat
  `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64` checkpoint: 48 decoder layers,
  512 routed experts with top-10 selection, GDN/QSA layer layout, MRoPE and the
  PLE metadata are read from GGUF rather than guessed. Commit `e243dda`.
- Added a native Turing-validated packed TQ4 QSA KV path. Its CUDA test passed
  on the actual RTX 2070 SM75 with the Qwen-width head geometry; it is a KV
  storage/attention-cache format, not a claim that all model weights use native
  INT4 tensor-core arithmetic. Commits `22ad4da` through `1fa2163`.
- Replaced resident Qwen4Exp projections with native packed GGUF operators:
  token embedding, output head, hyper-connection projections, GDN/QSA
  projections, QSA indexer, shared experts and PLE key/value projections.
  The focused construction/configuration suite passed `5/5` before commit
  `14785c7`.

### Upstream findings adopted as requirements

- llama.cpp's merged Qwen4Exp implementation (PR #27742) is the behavioural
  reference for this port. PLE must remain a single mmap-backed row table; only
  the host-selected rows may be transferred and dequantized. The PLE hash uses
  exact 64-bit multiplier/XOR arithmetic, EOS-aware history and persistent
  dilated-convolution state.
- The same upstream implementation requires one consistent inverse V-head
  permutation for every GDN V-indexed parameter (QKV V rows, gate, alpha/beta,
  A/dt, convolution V channels and output columns). Reordering only QKV can
  preserve shapes and throughput while producing incorrect text.
- The open follow-up PR #27774 establishes that QSA must cache the rotated K/V
  representation and undo the V rotation after attention; its indexer needs a
  separate raw-key cache that shares attention-cache slots during rewrites and
  restoration.

### Deliberately not accepted yet

- The 38 GiB Q5_1 PLE table and routed-expert stacks are **not** loaded into
  RAM or VRAM by the resident-GGUF commit. They require respectively a mmap row
  provider and a native packed expert-offload bank. No Qwen inference claim is
  made until both are implemented and a real request completes.

### Loader smoke evidence

- The real 33-shard AtomicChat Q4_K_M checkpoint completed a full non-expert
  iterator pass (885 yielded parameters) with no incomplete GDN/QSA/indexer or
  shared-expert fusions. The observed packed/dense boundaries were intentional:
  GDN input `16480 x 2720` bytes stays Q8 packed; the column-reordered GDN
  output is `2560 x 6144` BF16; QSA QKV is packed; and the BF16 indexer uses
  its native packed row width. This proves the loader mapping, not generation.

### Accepted — PLE mmap provider

- The Q5_1 `per_layer_token_embd.weight` is now opened as an NVMe-backed GGUF
  view rather than allocated as a model tensor. The exact PLE uint64 hash and
  EOS rule are retained before only selected rows are copied to GPU and
  dequantized. A real-table probe mapped `320,001,536 x 120` packed bytes,
  gathered/dequantized rows to BF16 correctly and increased process RSS by only
  about 2 MiB.
- Added the engine lifecycle hook immediately after `load_state_dict`, so PLE
  metadata buffers are populated before the host table opens. This closes the
  former dead-code path where `load_host_weights` existed but was never called.

## 2026-08-27

### Accepted

- TQ4-NC decode on SM75 uses a fixed 18-way split-K configuration instead of 8;
  CUDA-graph scratch capacity matches the selected split count.
- Long prefill uses packed TQ4 only for the historical KV prefix and reads the
  transformed current block directly, avoiding an immediate pack/unpack cycle.

### Rejected

- FP16 Tensor-Core operands for Ornith attention on SM75: emitted PTX used scalar
  FMA, not `mma.sync`, because the model's per-KV-head group is only M=8.
- Pairwise TQ4 byte loading: 16K decode timing changed by only 0.3%, so the code
  was reverted rather than increasing kernel complexity.
- A custom FWHT writer: existing K+V Hadamard work was `0.313 ms` and the full
  writer `1.159 ms` for 1,024 tokens, not material against long-prefill attention.
- First repository-compression runner filling strategy: a 512-token reserve made
  the nominal 1K prompt only 660 tokens and the generic README crowded out required
  source evidence. The raw result is retained in `TESTLOG.md`; the runner now
  prioritizes bounded real-source evidence windows and uses a 160-token margin.
- Cold long-context runtime profile: 64K is memory-feasible with TQ4-NC but not
  interactive on this laptop (18.75 minute TTFT, 12.69 decode tok/s). The limiting
  evidence is sustained 100% GPU utilization plus 86–88 C thermals and growing
  attention work, not KV exhaustion; the complete JSON trace is in TESTLOG.md.
- Cold 112K repository-compression profile: it completes without OOM or retained
  KV pages, but is decisively non-interactive (48m 35.8s TTFT, 9.12 decode tok/s).
  The full-request prefill estimate is 39.33 tok/s; the server's final-block rate
  was about 22.8 tok/s, showing that one average number hides the long-context tail.

### Added experiment discipline

- The repository declares its RTX 2070 Mobile / i7-8750H / 32 GiB / NVMe goal.
- `TESTLOG.md` is the measured-evidence log; `CHANGELOG.md` is the hypothesis and
  decision log. `AGENTS.md` requires both to be updated for every experiment.
- Context-speed artifacts now carry revision and parameter identity, and append a
  concise `slices.jsonl` point for direct curves: context on one axis, decode
  tok/s on the other, with TTFT/prefill retained as separate series.

### Discovered deployment gap

- The active DeepSeek Harness web profile does not enable either its bundled
  tool-result pruner or basic compaction provider. Do not attribute the current
  long-context behaviour to those packages until a local Ornith policy is
  explicitly mounted and benchmarked.

### Accepted cache evidence

- FreeToken's live radix cache delivers the intended agent behaviour with a
  stable DeepSeek Harness-shaped prefix: a 16,510-token exact replay reports
  16,448 cached tokens (99.62%) and reduces TTFT from 135.558 s cold to 2.013 s;
  an assistant-history append reaches 99.78% with 1.700 s TTFT. The repeatable
  runner and raw artifacts are `benchmarks/ornith_harness_cache_bench.py` and
  `benchmarks/results/2026-08-27-ornith-harness-cache/`.
- The cache experiment is intentionally a Harness-shaped proxy, not an assertion
  that its request bytes are identical to a private active DSH session. The
  existing telemetry plug-in exports runtime metrics but not prompt bodies.

### Runtime attribution

- Long cold prompts are GPU/attention-bound in the observed 16K run: 98.6% mean
  GPU utilisation, decreasing 1024-token block rate as the prefix grows, and no
  host-memory exhaustion. Cache reuse is therefore the dominant practical
  long-context speed lever.
- Decode has a separately measured CPU-MoE cost: 43.25–43.59% of eight routed
  experts per layer missed the 1429-slot cache and were handled by the CPU path.
  This is evidence to evaluate a MoE-residency/cache-size experiment next, not
  proof that changing it will improve end-to-end agent latency.

### Accepted live configuration changes

- Increased live FreeToken MoE cache from 1429 to 1700 slots through its
  idle-only runtime rebuild API while preserving the 122880-token KV cache.
  The matched forced-decode measurement improved from 28.93 to 31.27 tok/s
  (+8.1%) and reduced MoE misses from 43.37% to 35.07%. This is the active
  serving geometry.
- Enabled a conservative local DeepSeek Harness compaction policy for the
  FreeToken Ornith route: 88% trigger, 49152-token verbatim tail, 4096-token
  summarizer cap, and 32K-character tool-result pruning. It is deliberately
  not yet called an accepted quality policy until its first real long-session
  compaction is measured.

## 2026-08-28 — Qwen3.8 GGUF NVMe-source investigation

- Confirmed against FreeToken issues #55 and #122 that the existing generic
  GGUF MoE provider uses anonymous host banks and creates a combined `gate_up`
  copy. This is not file-backed lazy loading; for Qwen3.8 Q4_K_M it would
  require about 50 GiB of additional host memory and is rejected for the
  32-GiB target.
- Added the first Qwen4-specific primitive: `gguf_experts` retains `gate`,
  `up`, and `down` as separate, zero-copy `GGUFReader` views. The storage
  addresses are covered by a regression test, preventing accidental `cat()` or
  host-bank materialization in a later refactor.
- Reviewed FreeToken PR #211 before adapting it. The branch already contains
  an earlier stride-preserving z-grid chunker and regression test. Directly
  replacing it with #211 would lose padded-bank stride support, so the existing
  implementation will be runtime-verified rather than blindly cherry-picked.

### Accepted implementation boundary

- Added `qwen4_gguf`: a three-bank MoE cache (`gate`, `up`, `down`) and a
  separate-projection GGUF SwiGLU path. File-backed layers deliberately disable
  async full-layer prefill overlap and copy only LRU-selected rows into GPU
  slots; they do not register every mapped page with CUDA.
- Real SM75 probes proved the source mapping, selected-row copy and IQ2/IQ3/IQ4
  GPU GEMV work. The cold first expert selection is still far too expensive for
  a useful agent runtime, so the next accepted work is a bounded pinned staging
  buffer plus routed (rather than full-layer) file-backed prefill.

### Accepted routed prefill for file-backed experts

- `OffloadMoELayer._prefill_routed` now distinguishes ordinary RAM banks from
  `qwen4_gguf` file-backed banks. The former retain the existing full-layer DMA
  choreography. The latter route tokens first, promote only selected experts to
  LRU GPU slots, and invoke the compact-slot GEMV path. This removes an otherwise
  unavoidable NVMe sweep of all 512 experts for every Qwen MoE layer.
- This is a correctness and capacity change, not a claimed long-prompt speedup:
  a large prompt can still route many distinct experts. The upcoming live server
  test must measure how many are actually selected and its prefill/TTFT cost.

### Qwen3.8 first live-load blockers, fixed locally

- The pure GGUF directory deliberately has no HF `config.json` or tokenizer files.
  This is supported by the current branch: its first shard carries the model
  configuration and tokenizer. An initial launch accidentally used an older
  editable checkout, which bypassed that code; all subsequent launches use the
  current branch explicitly through `PYTHONPATH` and `python -m freetoken.cli`.
- Current `transformers` has no GGUF converter named `qwen4exp`. Qwen3.8 retains
  Qwen's GPT2-style BPE, so the adapter now maps it to the existing `qwen2`
  converter and its ordinary Qwen turn-end tokens.
- The first true load then exposed two missing mappings in the Qwen4 GGUF weight
  iterator. `blk.1.ple_key.weight` and `blk.1.ple_value.weight` were present in
  the AtomicChat Q4_K_M tensor table but were never yielded, causing a
  `KeyError` in `load_state_dict`. They now map to the two packed PLE projection
  operators. Both fixes have focused regression tests.
- A subsequent live load found the adjacent PLE normalization names were emitted
  as `norm_norm_key/query/conv`. The checkpoint names already include `norm_`;
  the adapter now emits the model's actual `norm_key`, `norm_query`, and
  `norm_conv` parameters, protected by the same iterator regression.

### Review findings retained for the next performance stage

- The independent review confirmed the new routed-prefill branch is required and
  is present from `21a40c8` onward; it prevents generic `materialize_layer()`
  from sweeping all 512 file-backed experts before routing.
- `qwen4_gguf` does **not** yet implement the CPU/hybrid expert executor. The
  live experiment therefore keeps explicit `--moe-backend offload`; auto/hybrid
  must not be enabled until its three-bank contract is implemented and tested.
- Selected file-backed copy is intentionally synchronous today. It is correct
  but a likely dominant cache-miss cost; bounded pinned staging/overlap remains
  a performance experiment after basic end-to-end functionality is proven.

### Accepted file-backed cache floor

- Generic FreeToken requires `moe_cache_size >= num_experts` because generic
  prefill must materialize a full layer. That consumed 1.14 GiB for 512 Qwen4
  slots and left too little CUDA scratch space for the first GDN forward.
- `qwen4_gguf` now uses the router working-set minimum (`top_k=10`) instead.
  This is safe because the file-backed prefill path routes before copying; it is
  intentionally an LRU capacity trade-off, not an assertion that ten slots give
  good cache hits. Generic formats retain the full-layer guard.

### Qwen4 text MRoPE batch contract

- Qwen4's QSA attention reads optional three-axis `rope_positions`; generic
  text batches did not declare that attribute at all, so the intended fallback
  to ordinary one-dimensional positions could not run. `Batch` now exposes it
  as `None` by default, while the attention path also remains defensive for
  compatibility with external batch-like callers.

### Runtime dependency: fused Triton router

- Installed OpenAI/Triton-Lang `triton_kernels` from the Triton `v3.6.0` commit
  (`7c56a5e`) into the FreeToken venv. The newest main revision was incompatible
  with FreeToken's pinned Triton 3.6.0 (`is_hip_gfx1250` import failure); the
  matching tag imports `triton_kernels.topk.topk` successfully.
- The source install briefly upgraded NumPy to 2.5.2; it was restored to 2.4.6,
  within FreeToken's declared `<2.5` constraint. This enables the fused router
  without silently changing the core Triton runtime.
- On the real Qwen path the optional router still rejects its model-specific
  geometry during JIT. `fused_topk` now treats optional-kernel runtime failures
  like a missing package: it logs once and uses the numerically equivalent
  PyTorch implementation rather than terminating the scheduler.

### Accepted FP16 activation boundary for Qwen4 GGUF

- Qwen4's engine honours the requested `--dtype`, but the shared GGUF embedding
  layer unconditionally returned BF16. With `--dtype float16` that made the
  very first GDN causal-convolution receive BF16 activations and FP16 state;
  Triton correctly rejected the mixed input. This was a compute-dtype bug, not
  a request to convert the GGUF expert weights.
- `GGUFEmbedding` now accepts an output dtype. The Qwen4 GGUF adapter takes it
  from the original meta-constructed embedding, which is the engine's requested
  compute dtype. Thus the checkpoint remains Q4_K_M/IQ GGUF on disk and in its
  packed expert cache, while token embeddings, activations, GDN state and
  compute use FP16 on the Turing runtime.
- The regression invokes the embedding's actual dequantization call through a
  small mocked kernel and verifies an FP16 tensor is returned. This protects
  behaviour rather than merely testing configuration plumbing.

### First end-to-end Qwen4 MRoPE path

- The first FP16 request progressed beyond GDN and failed only because the
  Qwen4 adapter imported a missing `apply_mrope_with_cos_sin_cache_inplace`
  symbol. Repository/issue search found no FreeToken implementation. We ported
  the relevant SGLang MRoPE structure into FreeToken's existing RoPE module,
  including the fresh Qwen3.8 partial-rotary bounds correction documented by
  `tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark`.
- This is a native CUDA/Triton path, not a CPU fallback: it selects the three
  temporal/height/width cosine-sine rows per token and rotates Q/K in place.
  Its padded lanes are masked to the true 32-lane rotary half-width, preventing
  the Qwen3.8 partial-RoPE out-of-bounds condition.
- A local import-contract regression guards the exact missing symbol. A live
  FP16 HTTP request now returns `200` after passing GDN, QSA and MRoPE. It is
  an important functional milestone, but not yet a usable quality/performance
  result; the observed output and speed are recorded in `TESTLOG.md`.

### Accepted Qwen4 GGUF tokenizer control-token correction

- The embedded GGUF chat template is byte-identical to Qwen's official
  `chat_template.jinja`; the template itself was not the error. The generic
  Qwen2 GGUF converter, however, did not register Qwen3.8's non-special
  `<think>`/`</think>` and XML tool delimiters as added tokens.
- Consequently `<think>` in the official generation prefix was BPE-split into
  three ordinary tokens instead of its real ID `248068`. The official prompt
  had 45 tokens while our runner used 47. The corrected loader registers these
  controls with `add_tokens` (not `add_special_tokens`), matching the official
  tokenizer's `special: false` semantics and preserving parser visibility.
- Direct comparison against the downloaded official `tokenizer.json` now has
  identical rendered text and all 45 IDs. The prior nonsense response is
  therefore invalid as a quality measurement and must be repeated after restart.

### Accepted Qwen4 GDN output-gate correction

- The official Qwen3.8 text config specifies `output_gate_type: "sigmoid"`.
  The GGUF metadata does not carry this value, and the local adapter had copied
  an older Qwen-family default of `"silu"`. This changes every Gated DeltaNet
  layer, so it is a correctness error rather than a performance preference.
- The Qwen4 GGUF architecture constant is now `sigmoid`, with a regression on
  the actual AtomicChat Q4_K_M metadata fixture. This supersedes the prior
  hard-coded SiLU assumption.
- A contemporaneous Qwen4 RMSNorm issue was investigated too. Our actual GGUF
  norm values are ones-centered, but this adapter already subtracts one before
  feeding `GemmaPlusOneRMSNorm`; the effective scale remains the raw checkpoint
  scale. That hypothesis is explicitly rejected for this runner.

### First quality-valid Qwen3.8 FreeToken smoke

- With the official sigmoid GDN gate, FP16/Q4_K_M now produces the requested
  `pong` twice from the same server process. This validates the full adapted
  architecture path sufficiently to begin performance work.
- It does not validate long-context quality or performance. The short-prompt
  repeat still takes 34.99 s and has no prompt-cache reuse under the current
  Qwen4-required naive cache; its detailed timing is retained in `TESTLOG.md`.

### Context-scaling guardrail

- The first 28-token synthetic prompt completed at 0.23 prefill tok/s, but the
  next 56-token probe stalled CPU-side for over five minutes with the GPU idle.
  The request also outlived client cancellation. This is recorded as a runtime
  limitation, not a benchmark value.
- Do not launch 1K+ context tests on this revision until the synchronous
  file-backed path has cancellation/timeout observability and host-side profiling.

### Qwen4 file-backed expert transfers are now batched per bank

- Profiling isolated the principal prefill mechanical cost after routing: a
  selected Qwen4 GGUF expert was copied to the LRU cache one row at a time,
  producing `3 * top_k` synchronous host-to-device submissions per MoE layer
  (`gate`, `up`, and `down`). This is especially costly for mmap/NVMe-backed
  source rows: kernel work is small, while submission and page-fault latency is
  repeated for each expert.
- The `qwen4_gguf` file-backed selected-miss path now gathers all chosen source
  rows once per bank with `index_select`, transfers the gathered tensor once,
  then scatters it to its remapped LRU slots with `index_copy_`. Whole-layer and
  heterogeneous compact-row layouts intentionally retain the prior
  prefix-copy fallback, so the optimisation does not widen the supported GGUF
  contract.
- A regression fails if the new selected-Qwen path reaches the former per-row
  helper, and verifies non-sequential source IDs land in the correct LRU slots
  for all three banks. This makes the optimisation behavioural rather than a
  timing-only change.

### Qwen4 routed prefill avoids flashlib's shape-unrolled LRU JIT

- During the first 56-token probe, the RTX 2070 was idle while one CPU core
  spent more than five minutes in `ptxas` compiling a 9.7 MiB (188,166-line)
  PTX file. Inspection identified `_lru_ensure_kernel` from flashlib's slot
  cache, not Qwen's compute kernels or NVMe I/O. Its sequential victim loop is
  unrolled to the routed prefill query geometry.
- File-backed Qwen4 GGUF now uses FreeToken's existing dynamic hybrid-admission
  kernel with its fetch cap set to the entire cache. Under the same invariant as
  ordinary LRU (the distinct routed expert count fits the cache), that admits
  every miss and rewrites the same slot IDs, but does not compile a loop sized
  to `tokens × top_k`. Other model formats continue to use flashlib unchanged.
- Upstream/issue search found no existing FlashML fix for this SM75
  shape-specialisation failure. FreeToken's own supported-model list also does
  not yet list Qwen3.8 Flash Next or general GGUF MoE; this remains a scoped
  fork adaptation rather than a claim of upstream support.

### Qwen4 file-backed prefill is capacity-safe

- The dynamic LRU admission used for the JIT correction is capped by its GPU
  cache size. That is naturally safe for decode (10 routes per token), but a
  long prefill can present more distinct routes than the 256-slot cache can
  admit. Leaving such routes as `-1` is not safe for the native GGUF MMVQ
  kernel, which expects only valid slot IDs.
- The Qwen file-backed prefill path now slices each MoE layer's routed input so
  `tokens_in_chunk × top_k <= cache_size`, then concatenates the independently
  computed outputs. This is a conservative correctness boundary: it avoids a
  device-to-host unique-ID sync while guaranteeing that every possible route
  fits the cache admission contract.
- The prior 2K performance probes remain useful I/O diagnostics but are marked
  provisional for quality/performance comparison until repeated with this
  capacity-safe path.

### MTP research finding

- Official Qwen3.8 configuration contains one hybrid MTP/NextN hidden layer,
  but FreeToken's GGUF adapters explicitly drop NextN/MTP and documentation
  states text-only serving with no speculative decoding. There is no `--mtp`
  runtime flag in this tree.
- A separate research artifact, `UnsignedChad/windows-freetoken-mtp`, validates
  a Qwen3.6 NVFP4 acceptance harness, but explicitly says its live server loop
  remains unwired. It is not directly mergeable: Qwen3.8 needs its own MTP
  weights plus safe rollback/commit of PLE, QSA and GDN state.

### Capacity-safe Qwen3.8 context measurements

- The first post-fix live ladder completed at 128, 296, 512, 1024 and 2047
  API prompt tokens, all with HTTP 200.  The corresponding prefill rates were
  1.22, 1.17, 4.57, 9.70 and 12.19 tok/s.  `TESTLOG.md` preserves the exact
  server configuration and profiling evidence.
- These are deliberately labelled warm expert-LRU measurements: the test
  sequence retains the GPU expert slots but has zero KV prompt-cache hits.
  This is useful for a continuing coding agent, but it is not a cold-start
  claim and must not be compared to the earlier pre-capacity-fix numbers.
- A corrected 1K trace sampled the scheduler worker rather than its HTTP
  parent: it used roughly 2.45 CPU cores and read about 575 MiB/s from NVMe,
  while the GPU averaged only about 25% utilisation.  Further prefill work
  should overlap/prefetch host expert staging; more FP16 compute or MTP cannot
  remove that bottleneck.

### 16K BF16-KV forward headroom boundary

- Qwen4 with 16,384 BF16-KV tokens and the 256-slot MoE LRU initializes on the
  RTX 2070 (0.42 GiB KV, 0.71 GiB reported free) but cannot execute a 16,383
  token prefill: GDN's causal convolution needs another 160 MiB while only
  about 101 MiB is free. The full error and clean process recovery are recorded
  in `TESTLOG.md`.
- This is a forward-workspace headroom constraint, not proof that 16K KV is
  intrinsically too large. It rules out claiming a 16K Qwen benchmark on the
  current BF16/256-slot configuration. The next configuration work is to
  preserve useful MoE reuse while making QSA's existing packed `tq4-nc` KV
  format accessible through the server CLI.

### TQ4-NC QSA KV is now a tested CLI option

- `--kv-cache-dtype tq4-nc` now reaches FreeToken's existing packed QSA KV
  implementation. A parser regression proves the public option stays accepted;
  the QSA/TQ4 focused suite passed before live serving.
- A real Qwen QSA request exposed a Triton 3.6 incompatibility hidden by the
  former CPU-only storage tests: a JIT kernel read Python global `_SCALE_EPS`.
  The kernel now uses the same literal epsilon style as the adjacent INT8/FP8
  kernel, and a CUDA regression compiles and executes the exact store path on
  SM75. The focused suite passed with the new GPU coverage.
- Live TQ4-QSA Qwen returns the deterministic thinking-disabled `pong` and
  reduces a 16K QSA KV allocation from 0.42 GiB to 0.14 GiB. It does not yet
  make 16K end-to-end viable with a useful expert LRU: 256 and 192 MoE slots
  still OOM in separate GDN/FLA temporary buffers. `TESTLOG.md` retains both
  failure points and decode-speed caveat rather than treating the memory win as
  a completed long-context benchmark.

### Qwen 16K uses the existing stateful chunked-prefill path

- Repository review confirmed that `--max-prefill-length 2048` produces one
  continued request, preserving Qwen4's naive GDN/QSA/PLE runtime state. The
  existing scheduler/QSA/Qwen regression subset passed (21 tests) before live
  use; no bespoke state transport was added.
- The first three 2048-token continuations completed at 11.26, 36.93 and 20.42
  prefill tok/s with 256 MoE slots and no OOM. The full request was deliberately
  stopped at 91–92 C because the RTX 2070 Mobile had active software thermal
  throttling and fell to 780 MHz. These partial values validate the mechanism,
  but they are not presented as a 16K benchmark until the thermal envelope is
  held stable.

### Full 16K Qwen prefill completes with normal cooling

- A subsequent unmodified-cooling repeat completed the complete 16,343-token
  API prompt with HTTP 200: TQ4-NC KV, 256 expert slots and eight stateful
  scheduler forwards require no fixed GPU clock or power cap. The useful
  2,048-token steady chunks were 42.95, 43.09, 43.47, 43.22 and 42.24 tok/s;
  the final full chunk was 36.31 tok/s. The cold first chunk and the scheduler's
  impossible final residual rate are recorded, but excluded from comparison.
- This validates the long-context execution path. It is not yet a plot-ready
  aggregate speed result because its manual control client did not preserve the
  final wall-clock JSON artifact; automated 64K/112K runs must use the checked-in
  benchmark runner so artifacts and `slices.jsonl` are written together.

### Rejected — current 64K token-major runtime is not yet viable

- A 2,048-token scheduler chunk exhausts GDN/FLA workspace early. Reducing the
  chunk to 1,024 avoids that immediate failure and sustains approximately
  8.8–10.0 prefill tok/s, but the request still OOMs around 45K when QSA creates
  a 78 MiB dense score output. The incomplete run is retained in `TESTLOG.md`;
  it is neither a decode result nor a quality result.
- The same run combines roughly 575–590 MiB/s NVMe reads and about 25% sampled
  GPU utilisation, while the drive's direct sequential ceiling is about
  1.50 GB/s. The accepted next architecture is therefore phase-aware: bounded
  QSA workspace first, then layer-major expert-deduplicated prefill and a
  separately measured decode cache.

### Proposed — evidence-gated Qwen3.8 runtime optimisation cycle

- Prior-art review found direct implementations of layer-major expert dedup,
  offset-sorted reads and asynchronous staging in Flash-MoE/S-MoE, Qwen state
  snapshots and bounded QSA in FreeToken PR #257, and reusable oracle telemetry
  in FreeToken PR #231. The design adopts those mechanisms instead of inventing
  a second model runtime.
- A public Qwen hardware-fit FreeToken draft already implements fixed-record
  file experts but concludes that its synchronous SSD path requires implausible
  2.30–4.59 GB/s at realistic miss rates. A sidecar is therefore deferred until
  dedup and asynchronous reads have measured the residual layout cost.
- The former proposal for an immediate 8–12 GiB pinned RAM L2 is narrowed. The
  default first experiment keeps Linux page cache as L2 and allocates only a
  bounded pinned staging pool; an explicit RAM cache must beat page-cache-only
  under the same 32 GiB memory wall without increasing destructive swap.
- The complete proposed architecture, acceptance thresholds and rollback flags
  are recorded in
  `docs/superpowers/specs/2026-08-28-qwen38-turing-runtime-optimization-design.md`.

### Revised — independent Sol Ultra review closes design gaps

- An independent read-only `gpt-5.6-sol` Ultra review compared the proposed
  runtime design with the current mixed-IQ GGUF/TQ4 implementation, all retained
  measurements and upstream Qwen PR #257. Its blocking claims were rechecked
  locally before the specification was changed.
- The first implementation scope is narrowed to existing-telemetry wiring and a
  page-size-4/TQ4-compatible fused QSA scorer. Page size 64, recurrent-state
  snapshots, hybrid radix and CUDA graph now have separate parity gates.
- Layer-major scheduling is separated from a future grouped IQ2_S/IQ3_S/IQ4_NL
  kernel. At reuse slab 1,024, perfect within-slab expert dedup has a 53.32
  MiB/token floor versus 61.17--62.77 MiB/token observed, so the former 50%
  physical-byte and mandatory 2x gates were removed as unsupported.
- The design now uses a single-variable E0--E17 experiment sequence, explicit
  rollback conditions, matched Ornith controls, full numerical/quality gates,
  linked staging-buffer geometry and conservative Linux page-cache handling.
- The current 16K configuration is relabelled execution-valid rather than a
  completed long-context quality baseline; the approximately 9.4 tok/s 64K
  value remains an incomplete boundary observation.

### Fixed — native padded Triton router for Qwen3.8 top-10 MoE

- Qwen3.8 routes every MoE layer to 10 experts.  The optional
  `triton_kernels.topk` path constructs `tl.arange(0, 10)`, which Triton rejects
  because the range is not a power of two; FreeToken then used the much slower
  Torch softmax/top-k fallback on every routed layer and every generated token.
- The fork now uses the already-present upstream Qwen3.8 padded/masked Triton
  router for non-power-of-two CUDA top-k.  It preserves full-row softmax,
  optional renormalisation, deterministic lowest-ID tie handling and the
  scheduler's padded-row mask. CPU invocations deliberately keep the Torch
  implementation even if `triton_kernels` is installed.
- On the matched 1K / 255-decode-token control, this improved measured decode
  throughput from 0.910 to 1.834 tok/s (+101.6%) and increased mean sampled GPU
  utilisation from 17.1% to 28.3%. The raw artifacts and all failed attempts
  are retained in `TESTLOG.md` and `benchmarks/results/`.

### Verified — 16K Qwen3.8 profile with TQ4-NC and native top-10 routing

- The automated fixed-seed runner completed 16,396 prompt tokens and 255 output
  tokens without OOM. End-to-end prompt throughput was 35.21 tok/s; warmed
  1,024-token prefill chunks sustained 39.27–40.79 tok/s and decode was 1.686
  tok/s. GPU utilisation averaged 67% and peaked at 97%.
- This validates the page-size-4/TQ4-NC QSA allocation at 16K in the exact
  disk-backed Q4_K_M runtime. Decode still has zero global-LRU hits despite a
  38.58% stationary per-layer oracle, making layer-aware cache partitioning the
  next evidence-backed optimisation rather than a larger undifferentiated LRU.

### Verified — complete 64K Qwen3.8 profile and cache-policy gate

- The bounded 16 MiB QSA score workspace completed a fixed-seed 65,548-token
  prompt plus 255 generated tokens on RTX 2070 Mobile.  End-to-end prefill was
  38.16 tok/s and decode was 1.614 tok/s; peak sampled VRAM was 7,774 MiB.
- The 64K trace proves global LRU cross-layer thrashing: all 122,400 decode
  layer-expert references missed and were evicted, while the measured
  stationary per-layer oracle is 39.01% at the same 256-slot capacity.  The
  next controlled implementation is a protected per-layer VRAM allocation,
  not a larger global cache.
- Only 12.02 MiB of process physical reads occurred in the warm run, whereas
  the expert path copied 252.24 GB host-to-device.  Linux page cache already
  served as the effective RAM tier here; a pinned-RAM L2 stays deferred until a
  matched cache-policy result proves a residual disk bottleneck.

### Recorded — 112K capacity boundary is GDN VRAM headroom

- The unmodified 122,880-token allocation loads with TQ4-NC KV (1.07 GiB) but
  leaves only 0.29 GiB free VRAM on RTX 2070 Mobile.  Its first 1,024-token
  prefill fails in the GDN Triton output kernel with CUDA OOM.
- This establishes 112K/122,880 as a capacity boundary for the current
  256-slot configuration, not a completed benchmark.  Because that global
  cache produced zero decode hits at 64K, the next capacity probe may trade
  part of it for GDN activation headroom; it will be labelled separately from
  the fixed-configuration speed matrix.

### Fixed — protected-layer admission critical regressions

- Commit `9749ea1` prevents resident routes from staging a spurious
  copy/eviction in the protected Triton admission kernel and scopes direct
  engine warmup to prefill admission. The policy remains opt-in and has not
  been used for a performance claim yet.
- CPU RED/GREEN regressions and a CUDA-gated parity regression were added. The
  latter must run on the actual RTX 2070 before the policy can enter a live A/B.

### Planned — REAP-256 Qwen model control

- A separate 61.9 GB Qwen4Exp GGUF candidate with 256 experts/layer is being
  downloaded outside `/tmp`. It will first be metadata-validated and then run
  with the stable LRU baseline. This prevents conflating REAP pruning with the
  new cache policy or with the existing Q4_K_M quantization.

### Fixed — reliable benchmark response-equivalence gate

- Commits `79a57b8` and `6dc50c5` add a prompt-private SHA-256 of the final
  streamed content and reject SSE errors, incomplete streams and missing
  terminal finish reasons before publishing an artifact. The result makes
  REAP/LRU output comparisons meaningful without storing user text.

### Verified — REAP-256 static Qwen4Exp compatibility

- The first downloading shard already proves the expected `qwen4exp` 48-layer,
  256-expert, top-10 layout. Its routed expert types (`Q8_0`, `IQ3_XXS`,
  `IQ4_NL`, `IQ4_XS`) are all supported by the existing GPU MoE-vector path.
  This is intentionally labelled static-only until both GGUF shards complete.

### Fixed — REAP-256 PLE `IQ4_NL` loader compatibility

- The GGUF PLE table's `per_layer_token_embd.weight` is `GGML_IQ4_NL`, but the
  loader had a stale `Q5_1`-only gate. It now accepts exactly the quantized
  types exposed by `DEQUANT_TYPES`, the same native `ggml_dequantize` dispatch
  used by PLE forward.
- Existing row-width, packed-byte and minimum-table-size validation remains in
  place; unsupported/unquantized types still fail explicitly.
- TDD result: RED `2 failed, 3 passed`; GREEN `5 passed`; relevant regression
  suite `27 passed, 2 skipped`. No model, benchmark, download, expert, cache,
  or runtime changes were made. Full details:
  `.superpowers/sdd/2026-08-29-qwen38-reap256-ple-iq4nl-report.md`.

### Fixed — enforce real `IQ4_NL` PLE block geometry

- Commit `f98eaef` closes the remaining Important review gap after the REAP
  `IQ4_NL` type-gate change.  Because the native dequantizer rounds flattened
  input to complete 256-value blocks, the loader now rejects IQ4_NL PLE when
  `ple_embed_dim` is not divisible by 256 and reports the unsafe value clearly.
- The acceptance regression uses Qwen's real 16-head/2,560-value geometry:
  160 values per head, table shape `(16, 160)`, and 90 packed bytes per row.
  A CUDA-gated test checks 16×160 native dequantization against a deterministic
  IQ4_NL nibble reference.
- Independent review then found that the original fixture compressed the same
  16 heads into the wrong n-gram layout. It now uses Qwen's exact
  `ngram_size=3`, `heads_per_ngram=8` configuration and covers the full host
  path: n-gram IDs → `index_select` → dequant dispatch → `(tokens, 2560)` view.
  The negative case now has a valid 32-value/18-byte IQ4_NL row and isolates the
  unsafe aggregate embedding dimension.
- Current relevant Qwen4Exp/GGUF/PLE suite: `11 passed, 1 skipped`. CUDA was
  unavailable in this environment, so no live-kernel result is claimed. No
  model, benchmark, download, cache, or runtime execution was performed.

Full report: `.superpowers/sdd/2026-08-29-qwen38-reap256-ple-iq4nl-geometry/task-ple-iq4nl-geometry-report.md`.

### Verified — REAP-256 1K stable-LRU control

- Both downloaded GGUF shards passed `hf cache verify` (`checked=2`), then the
  normal FreeToken reader enumerated all 1,224 tensors and confirmed Qwen4Exp,
  48 layers, 256 experts/layer, top-10 routing and the IQ4_NL PLE table.
- On the exact Q4_K_M 1K control geometry (BF16 activations, `tq4-nc`, page 4,
  16 MiB QSA workspace, 16,384 KV tokens, 256 global LRU slots, offload,
  naive cache, one request and seed `20260828`), REAP completed a terminal
  streaming request: 1,036 prompt and 254 output tokens in 187.14 s. Its
  end-to-end prefill/decode were 14.63 / 2.175 tok/s, versus 12.10 / 1.834
  tok/s for Q4_K_M. The final-content SHA-256 is stored only in the artifact.
- The result does **not** establish a quality comparison or a cache fix. Decode
  remained `0 / 121,920` L1 hits/misses and copied 276.94 GB host-to-device,
  versus Q4_K_M's 252.24 GB; the smaller on-disk model is therefore not proof
  of a smaller decode working set. 16K and 64K are still required.

### Verified — REAP-256 16K stable-LRU control

- With the same seed, LRU, 256 slots, `tq4-nc` KV and 18,432-token allocation
  geometry as the Q4_K_M 16K control, REAP completed 16,396 prompt plus 254
  output tokens in 567.03 s. Its end-to-end prefill/decode were 36.72 / 2.100
  tok/s, against Q4_K_M's 35.21 / 1.686; this is -8.0% wall time, +4.3%
  prefill and +24.5% decode.
- This remains a model/quantization observation, not cache-policy attribution:
  decode recorded zero L1 hits and 121,920 misses/evictions. REAP transferred
  276.94 GB H2D during decode, above the Q4_K_M control's 252.24 GB. The
  complete immutable artifact is
  `benchmarks/results/qwen38-reap256-16k-lru/context-16384.json`; 64K remains
  the last matched control.

### Verified — REAP-256 64K stable-LRU control

- The final matched control completed 65,548 prompt and 254 output tokens in
  1,838.30 s at 38.229 end-to-end prefill tok/s and 2.046 decode tok/s. Against
  Q4_K_M's 1,874.98 s / 38.162 / 1.614, this is -2.0% wall time, +0.2% prefill
  and +26.7% decode. The repeated decode advantage is now measured at 1K, 16K
  and 64K with the same seed and LRU configuration.
- It is still not cache attribution: decode is `0 / 121,920` L1 hits/misses,
  H2D is 276.94 GB versus the control's 252.24 GB, and this REAP run reported
  48.73 MiB physical reads versus 12.02 MiB. The immutable raw artifact is
  `benchmarks/results/qwen38-reap256-64k-lru/context-65536.json`.

### Added — append-only cross-model speed registry

- `benchmarks/results/model-context-speed.jsonl` is now the central durable
  registry for model × quantization × actual context × prefill/decode speed.
  It records seed, runtime profile, source revision and immutable artifact path.
- `benchmarks/qwen38_turing_profile.py` now adds every successfully terminal
  future run to that registry. The helper refuses a duplicate raw artifact;
  incomplete streams remain rejected before publication.
- Backfilled Qwen REAP-256 and new Ornith Q4_K_M 1K/16K/64K points establish the
  first plot-ready direct hardware comparison. On their model-appropriate
  FreeToken profiles, Ornith decode is 29.085/20.996/13.591 tok/s against
  Qwen's 2.175/2.100/2.046 tok/s. This is not an equal-geometry cache-policy A/B.
- The new Ornith 64K baseline uses `--max-prefill-length 640` and measured
  53.448 prefill tok/s. It is retained as a conservative baseline; a dedicated
  640-versus-1024 measurement is still required before changing production
  chunking.

### Verified — Ornith 16K prefill chunk 1024

- The requested one-variable 16K A/B is complete. With the same 16,400-token
  prompt and model/KV/MoE geometry, increasing `--max-prefill-length` from 640
  to 1024 completed without OOM and used only 40 MiB more sampled VRAM.
- End-to-end prefill/decode improved from 91.753/20.996 to 97.834/21.481 tok/s
  (+6.63%/+2.31%); wall time improved from 184.82 to 172.74 s (-6.54%). The
  immutable artifact and auto-added central-registry row are
  `benchmarks/results/ornith35-q4km-16k-p1024-r2/context-16384.json`.
- The terminal response length changed from 127 to 108 tokens even with the
  same requested seed, so this is deliberately recorded as a throughput/capacity
  result, not exact response-equivalence proof.

### Fixed — direct benchmark runner after registry integration

- The first 1024 A/B attempt exposed that direct execution of
  `python benchmarks/qwen38_turing_profile.py` did not put the repository root
  on `sys.path`, so its new `benchmarks.speed_registry` import failed before
  model load. A RED entrypoint regression now reproduces the exact invocation;
  the minimal root-path fix makes it pass. Relevant benchmark suite: 15 passed.

### Added — exhaustive prompt-private live benchmark telemetry

- `benchmarks/qwen38_turing_profile.py` now records phase-labelled one-second
  GPU utilization/VRAM/power/temperature plus process RSS, physical reads and
  page faults. It records TTFT and a timestamped SSE-delta progress trace for
  long decode without retaining prompt, visible answer, or reasoning text.
- `--ignore-eos` and `--trace-stride-events` make forced long generation a
  deliberate benchmark mode; exact final token usage remains the speed source,
  while the delta trace is only a timeline (not an assumed token count).
- `benchmarks/benchmark_ledger.py` is the append-only all-attempt ledger;
  `benchmarks/ornith_prefill_sweep.py` probes increasing chunk sizes and stops
  at the first capacity failure by default; `render_benchmark_dashboard.py`
  regenerates a single offline graph/table from all success and failure events.
- Backfill imported 21 historical raw artifacts. The first new forced live run
  added the successful p1024 16K/4K trace and its preceding path-only startup
  failure, so neither is silently lost.
- The first auto-sweep invocation also remains in the ledger as a zero-GPU
  `startup_error`: its child parser rejected a dash-prefixed server argument.
  The child command is now tested to bind every FreeToken server option as
  `--server-arg=<value>` and parser usage errors cannot be mislabeled timeout.

### Added — Ornith quantization candidate intake for RTX 2070

- Recorded a research-only, no-download candidate scan in `README.md` and
  `TESTLOG.md`. The immediate direct GGUF tests are AtomicChat
  `AD-Q4_K-IQ4_XS` and the Unsloth-Dynamic-style `UD-Q4_K_S`/`UD-IQ4_XS` tiers;
  FreeToken's existing GGUF kernels cover their named `Q4_K`/`IQ4_XS` types.
- The high-upside path is the REAP-50 source checkpoint: it reduces Ornith from
  256 to 128 routed experts while retaining top-8 routing. It is explicitly a
  future native-K-quant conversion and quality experiment, not a claim about
  the published NVFP4A16 release on Turing.
- APEX Compact is retained behind a complete GGUF-type/startup gate. TQ3 and
  generic Qwen3.5-MoE MXFP4 GGUF variants are rejected as direct candidates:
  their required tensor/runtime support is absent from the current FreeToken
  path. No benchmark, model download or runtime behavior changed.

### Research — TurboQuant/TQ3 FreeToken feasibility correction

- Scanned current FreeToken upstream, its public issue/PR surface and sampled
  forks. No existing `TQ3_4S` FreeToken implementation was found; upstream
  issue #141 remains an unassigned KV-compression proposal with no linked code.
- Distinguished three incompatible uses of the name TurboQuant: Google's
  online vector/KV-cache method, the custom GGUF `TQ3_4S` weight format in
  `turbo-tan/llama.cpp-tq3`, and vLLM's related HIGGS-style online weight path.
- Exact GGUF accounting promotes the Ornith TQ3_4S release from “unsupported,
  exclude” to “unsupported, high-priority port”: expert slots are 22.89%
  smaller than Q4_K_M and the same packed-weight budget projects about 36%
  more slots. This is not yet a benchmark or a quality claim.
- Corrected the earlier AD/UD speed hypothesis. AD-Q4_K-IQ4_XS reduces each
  expert transfer and has better publisher KLD, but its Q8 dense tensors consume
  enough extra VRAM to project fewer cache slots. UD-Q4_K_S has no slot-stride
  advantage because three Q6_K expert-down layers define the global stride.
- Internet evidence argues against replacing the existing FreeToken `tq4-nc`
  KV cache first: vLLM's merged TurboQuant study reports substantial low-bit
  throughput costs, and an SM75 llama.cpp test also measured slower prefill and
  decode. No runtime code, model download or service state changed.

### Integrated — upstream Qwen3.8 baseline before the TQ3_4S port

- Integrated upstream through `58f4b9e` while keeping official block-sparse QSA
  and the fork's GGUF token-indexed QSA as separate runtime/cache contracts.
- Fixed GGUF Qwen3.8 router-weight renormalization and migrated GDN gate values
  to the upstream string API (`sigmoid` Qwen3.8, `silu` Ornith/Qwen3.5), each
  covered by a regression test.
- Corrected an over-broad AOT registry assertion: only exact-geometry Qwen3.8
  aliases are claimed by its AOT entry; universal GGUF/JIT architectures are not.
- Replaced one bit-exact CPU PLE assertion with a measured `1e-6` tolerance after
  proving BLAS reduction order was the sole source (`max abs 8.34e-7`).
- Combined merge gate: `245 passed, 139 skipped`. No model download, TQ3 kernel,
  service restart or new performance number occurred in this integration step.
- Retained `tq4-nc` as the first TQ3-weight A/B control. A hypothetical 3-bit KV
  mode is tracked separately because it is a distinct online codec and projects
  only about +70 slots at 122,880 for the 50-byte `TURBO3_0` block, versus about
  +514 from TQ3 weights. This corrects an earlier estimate that omitted one K/V
  factor and overstated the KV saving by 2x.

### Added — TQ3_4S type metadata and CPU correctness authority

- Added GGML type 46 geometry and the exact 16-byte `block_tq3_4s` C contract.
- Added literal-tested E3M5 scale decoding, 3-bit centroid unpacking and inverse
  signed WHT in pure Torch. Type 46 remains absent from every CUDA capability
  set until the corresponding switch and SM75 parity tests land.
- Added a local GGUF reader compatibility seam for types newer than installed
  `gguf-py`, without monkeypatching its global enum or size dictionary.
- Reconciled the real Ornith sparse header: 753 tensors read, 381 TQ3 tensors,
  17,230,725,120 TQ3 packed bytes.
- Focused gates: 18 passed for TQ3/type/shards; corrected broad GGUF suite
  71 passed, 4 skipped. No runtime speed result was produced.

### Added — exact TQ3_4S CUDA dequantization on SM75

- Added the exact type-46 materialized CUDA decoder: E3M5 scales, authoritative
  centroids, 3-bit unpack, inverse signed WHT and one-warp-per-block launch.
- Kept the donor's approximate DP4A vecdot out of the correctness authority and
  kept TQ3_4S absent from MMVQ/MMQ/MoE capability sets.
- Real RTX 2070 FP32/FP16/BF16 parity matched the pure-Torch oracle bit-for-bit on
  fixed adversarial/random blocks (`max_abs=0`, `mean_abs=0`).
- TDD record: unsupported-type RED after native build; final CUDA GREEN `3
  passed`; combined GGUF/SM75 gate `77 passed, 1 skipped`.
- Fixed a previously skipped IQ4_NL PLE CUDA fixture that used `uint8` as a LUT
  index. No real-model speed or quality result is claimed yet.
- Luna review found no exact-kernel blocker; explicit output casting and stricter
  exact-SM75/FP32/zero-scale/literal gates were added. Exact `getrows` is tracked
  as a later performance path.

### Added — transformed TQ3_4S dense MMVQ on SM75

- Added source-type-aware signed normalized WHT in Q8_1 activation quantization
  and a packed PRMT+DP4A type-46 MMVQ path for batch-one/small-batch decode.
- Preserved the exact materialized decoder as the correctness authority. A
  missing-WHT negative control is over 100x worse than the fitted kernel, so a
  metadata-only or untransformed dispatch cannot silently pass.
- Audited the donor's symmetric DP4A levels before accepting them: they produced
  about 6.7% relative L2 error because TQ3_4S uses an asymmetric Lloyd-Max
  codebook. Refit a common scale and eight int8 levels to those exact centroids,
  reducing weighted centroid RMSE from 0.05937 to 0.00231 with the same PRMT and
  DP4A instruction count.
- Added a reproducible SM75 benchmark that retains every CUDA-event sample plus
  source/GPU/seed/thermal provenance. The final source-hashed run alternates
  packed/fallback order and enforces cosine/relative-L2 gates. Across the two
  real Ornith expert matrix shapes and FP16/BF16 it measured
  0.02504-0.02587 ms p50 versus 0.13731-0.13915 ms for exact
  materialize-plus-MM, a 5.31-5.55x kernel speedup. Earlier sequential-order
  5.70-6.15x runs remain preserved as intermediate artifacts.
- Real-shape relative L2 against the exact FP32 CPU authority was 0.55-0.61%.
  This is explicitly not a full-model tok/s, cache-hit, quality, fused-MoE or
  prefill result; TQ3_4S remains absent from those capability sets for now.
- Added an offline exhaustive fitter that derives Gaussian Lloyd-Max bin weights,
  enumerates 420 reachable int8 tables and reproduces the checked-in levels,
  scale, packed hex words and 0.002305 weighted RMSE.
- Added TQ3-specific API guards for 32-column block geometry, contiguous input,
  packed-uint8 storage and 16-byte weight alignment. Extended MMVQ parity to
  FP32 and zero-scale blocks after independent Luna review. Final expanded
  Task-4 gate: 59 passed with one pre-existing read-only mmap warning.

### Added — packed TQ3_4S selected-expert MoE on SM75

- Connected type 46 to the existing expert-slot MMVQ kernel while preserving
  packed bytes end-to-end. MoE activation Q8_1 now applies the same source-keyed
  signed WHT as dense MMVQ; no BF16 expert materialization was introduced.
- Added input/type/slot-size/stride/alignment guards around the TQ3 MoE API and
  retained the existing byte-stride cache contract for layers whose native
  expert matrix is smaller than the global maximum slot.
- Added an in-device cache-slot bounds guard after independent Luna review found
  and reproduced an invalid global read. Invalid TQ3 route IDs now leave their
  pre-zeroed output rows untouched, avoiding both OOB and a per-token CPU/GPU
  synchronization. Dense packed matrix bytes must occupy the prefix of a slot;
  expert-level tail padding is supported, row padding is explicitly unsupported.
- Added FP16/BF16 selected-expert parity, hostile expert-tail sentinels and a
  full top-k SwiGLU routing-accumulation comparison against exact materialized
  FP32 weights. Initial focused gate: 3 passed; mixed existing MoE/cache/grid-z
  gate: 52 passed with one pre-existing mmap warning.
- Added a source-hashed resident-slot benchmark at the real Ornith H=2048,
  I=512, top-8 geometry. Three post-guard repeats measured p50 ranges of
  0.182496-0.196032 ms FP16 and 0.187856-0.207056 ms BF16 while touching 12 MiB
  packed; medians are 0.194096/0.204256 ms. V4 records and verifies every public
  wrapper, activation and CUDA source hash. Relative L2 after both expert
  projections and routing was 1.006%/1.151%. This is kernel-only and excludes
  cache transfer, the rest of the model and model tok/s.
- Final mixed Task-5 regression gate: 55 passed with one pre-existing read-only
  NumPy mmap warning. The invalid-ID regression separately passed CUDA memcheck
  with zero reported errors.
- Final independent Luna review found no P1/P2 blocker, reran 4 MoE and 2
  provenance tests successfully, and matched all 10 v4 source hashes.

### Added — TQ3_4S correctness-first prefill sweep

- Added CPU dispatch, SM75 exact large-batch dense and 32-token top-8 MoE gates;
  all pass without adding an unverified TQ3 MMQ case.
- Added a source-hashed prefill layer benchmark over 1/6/7/16/64/128/256/512/
  1024 tokens with every CUDA-event sample and peak VRAM allocation retained.
- Dense exact materialization crosses from 0.043632 ms packed MMVQ at six tokens
  to 0.295168 ms at seven, then amortizes to 0.509856 ms at 1,024 with 6 MiB
  peak allocated delta. Resident top-8 MoE reaches 66.854015 ms and 88.016 MiB
  peak delta at 1,024. No OOM occurred.
- The result identifies the selected-expert MMVQ prefill path and real cache
  traffic as the likely bottlenecks; all measurements remain explicitly
  per-layer, resident-only, and not model tok/s.
- Expanded loader/kernel/prefill/provenance gate: 61 passed with one
  pre-existing read-only mmap warning.

### Verified — real Ornith TQ3_4S checkpoint intake

- Pinned `YTan2000/Ornith-1.5-35B-A3B-TQ3_4S` at revision `d63085f` and
  downloaded only its text GGUF into `/home/random/dev/qwen/models/`.
- The Hugging Face CLI stalled at 87,883,318 bytes; an eight-connection aria2
  retry completed the 18,051,687,776-byte object, whose SHA-256 exactly matches
  the published LFS hash.
- Header-only inspection found 753 tensors, including 381 TQ3_4S tensors with
  17,230,725,120 packed bytes. FreeToken resolves 41 advertised blocks to 40
  served decoder layers plus one omitted NextN/MTP block, with 256 experts and
  top-8 routing.
- Added immutable checkpoint/download/config evidence at
  `benchmarks/results/ornith35-tq3-sm75-smoke-task6-v1/checkpoint-audit.json`.

### Fixed — packed TQ4 cannot enter generic Triton MHA

- A real Ornith TQ3_4S startup exposed that `tq4-nc` half-width storage was
  accepted for generic MHA even though only QSA has a packed-nibble reader. It
  failed during CUDA-graph warmup after model loading, not in a TQ3 weight
  kernel and not from OOM.
- Added a red-first validation regression and now reject generic-MHA TQ4 before
  expensive loading; the dedicated QSA + TQ4 combination remains valid. The KV
  gate passes 9 tests with one CUDA-only skip.
- Preserved the failed startup geometry, resource peaks and traceback summary in
  `runtime-smoke-v1-tq4-nc.json`. Historical TQ4 Ornith artifacts came from a
  dirty worktree; no reachable generic packed-MHA implementation was found.

### Verified — Ornith TQ3_4S serves on RTX 2070

- Repeated the 16K launch with the supported INT8 KV path. Auto-sizing selected
  2,633 1,572,864-byte expert slots, the server completed CUDA/prefill warmup,
  and two deterministic requests returned exactly `Turing works` with HTTP 200.
- A cold real 1,012-token repository task measured 8.983 s TTFT, 112.660
  prefill tok/s and 37.612 decode tok/s. A 383-token warm generation sustained
  38.883 decode tok/s and reported 960 cached prompt tokens; its total-prompt
  quotient is deliberately excluded from cold-prefill comparisons.
- The saved Q4_K_M 1K control measured 28.443 decode tok/s, making the initial
  TQ3 decode signal 32-37% faster. Cold prefill moved the opposite way: 112.660
  versus 183.534 tok/s (-38.6%) with 62.9% longer TTFT. Context/KV/slot/source
  controls differ, so neither direction is final until Task 7's matched
  fixed-slot and auto-slot A/B.
- Expanded post-server regression: 77 passed with one pre-existing GGUF mmap
  warning. All successful and failed raw evidence is under
  `benchmarks/results/ornith35-tq3-sm75-smoke-task6-v1/`.
- Corrected the initial weight-control policy from generic-MHA TQ4 to INT8.
  Three-bit KV remains a separate codec project; the preferred first design is
  asymmetric INT8 K plus TQ3_0 V, not reuse of TQ3_4S weight blocks.
- Corrected benchmark publication semantics after Luna review: warm cache hits
  now expose cached/new token counts and a null cold-prefill rate, while the
  naive total-prompt quotient remains separately available. Added exact
  dirty-source, software, model-revision and checkpoint provenance for the real
  Task-6 run, plus `_adjust_config` coverage for the generic-MHA TQ4 rejection.
  Provenance now hashes `git diff HEAD`, so staged experiment code cannot
  silently produce the SHA-256 of an empty diff.
- Fixed a latent `qsa.py` versus `qsa/` import collision by relocating legacy
  gathered-QSA kernels into the package and re-exporting both legacy and newer
  paged APIs. The expanded RTX-2070 KV/QSA matrix passes 327 tests with 3 skips.
- Closed the second Luna provenance review: `/v1/stats` now publishes KV dtype;
  the benchmark requires an intake-provided model SHA-256 and captures model
  revision/file stats plus GPU/driver identity; Git identity includes staged,
  unstaged and untracked sources. Absent cache telemetry remains unknown instead
  of being mislabeled as a cold prefill, and the active plan now uses INT8 for
  generic-MHA controls.
- Final post-review combined Task-6 gate: 135 tests passed; the separate broader
  RTX-2070 KV/QSA matrix remains 327 passed with 3 skips.

### Verified — matched Ornith TQ3_4S weight/cache A/B on SM75

- Repeated Q4_K_M fixed-1429, TQ3_4S fixed-1429 and TQ3_4S auto-2633 under the
  same 16K INT8-KV, prefill-1024, greedy single-request controls. Two independent
  series reproduced the same ordering and bit-identical output per weight format.
- Canonical decode was 24.687 / 27.551 / 33.425 tok/s. At fixed slots TQ3_4S was
  11.60% faster; using its smaller residency footprint for 1,204 more expert
  slots made it 35.40% faster than Q4_K_M and 21.32% faster than fixed TQ3.
- Fixed→auto TQ3 cut MoE expert-cache misses and decode copy bytes by 35.41%
  without changing the generated bytes. Against Q4, auto TQ3 moved 48.85% fewer
  expert bytes per output token. This separates the weight-width benefit from
  the larger-residency benefit.
- Added aggregate CPU/iowait and physical-NVMe interval telemetry to the context
  runner. The red import test became green; partitions are excluded to avoid
  namespace double-counting. Physical reads are documented as page-cache/order
  dependent rather than used as a codec-only causal result.
- Excluded prior benchmark results from repository prompts/provenance, inferred
  a zero prompt-cache hit only from fresh server counters, and used static cache
  geometry/declared KV mode before first-request stats are populated.
- Added a dependency-free SVG renderer and published every valid Task-7 point
  into the 21-row append-only context-speed ledger. Both focused A/B and complete
  ledger SVG/PNG plots are checked in with their raw JSON authority.
- Do not claim quality parity: the lexical rubric scored all runs 4/5, but the
  TQ3 answer reversed accepted/rejected experiment facts and was truncated; Q4
  was more accurate on this single prompt. Broader quality evaluation remains.

### Fixed — objective 2D model/feature throughput graph

- Replaced the misleading all-ledger polyline and interim dashboard with one 2D
  figure: linear decode tok/s on X and actual context tokens on a log2 Y axis
  with one equally spaced tick per doubling.
- Added `benchmarks/comparison_cohorts.json` as an auditable assignment layer.
  All 21 immutable ledger rows are plotted once; the unmatched
  108-output-token run remains visible as a cross with an exclusion reason.
- Made graph generation fail on unclassified or multiply assigned rows, so a
  future benchmark cannot silently become a visually invalid comparison.
- Connected only the unchanged Ornith Q4_K_M and Qwen REAP configurations across
  1K/16K/64K. TQ3/cache/prefill-block repeats remain isolated points, preventing
  a one-context modification from becoming a fabricated context curve.
- Used blue shades and distinct marker shapes for Ornith modifications and a
  separate green series for Qwen. Kept raw repeat points rather than replacing
  them with a selected or averaged result.
- Closed independent-review P2 gaps: baseline series/category metadata is
  checked against the ledger model/quant/context identity; duplicate artifact
  rows are rejected before dictionary construction; SVG tests now compare all
  21 exact artifact coordinates and both line-membership sets with source data.
- Mitigated real 16K point overlap without coordinate jitter by cycling the
  p1024-p4096 sweep through four marker shapes and eight related blue shades;
  exact values remain printed below the plot.
- Kept every source measurement unchanged in the append-only JSONL registry;
  regenerated only the derived `model-context-speed.svg/png` presentation.

### Fixed — live context-speed trajectories instead of terminal-only points

- Root-caused the 16K pile-up to the presentation layer: it read one terminal
  average from each of 21 ledger rows and ignored the raw per-window telemetry
  already saved beside or inside every artifact.
- Added native extraction from FreeToken `Decode batch` stdout records and from
  `runtime_samples[].server_stats`, yielding 900 stable live decode points
  across all 21 runs (830 stdout plus 70 runtime-stat samples).
- Added tracked `model-context-speed-live.jsonl` with source SHA-256 and
  line/index for every point. This preserves reproducibility in clean clones
  without force-adding the locally retained ANSI `*.stdout.log` files ignored
  by Git; CLI rendering automatically consumes the sibling portable ledger.
- Excluded only non-decode transition measurements: the first stdout interval
  spans prefill because the reporter's decode timer starts at construction;
  runtime-stat windows begin after at least 16 completion tokens. Raw evidence
  remains unchanged and every retained SVG mark records its artifact and exact
  source line/index.
- Replaced isolated terminal modification marks with per-run live traces. The
  eight p1024-p4096 runs now expose all 808 measured windows from 16,481 through
  20,481 current KV tokens, with explicit shade/shape legend and no coordinate
  jitter or smoothing.
- Removed all cross-invocation summary trends and terminal mean markers from
  the live figure. It now contains only 900 native interval samples and 21
  within-invocation traces; terminal means remain in the separate JSONL ledger.
- Added exact raw-source/coordinate, sample-count, trace-membership, transition
  filtering and legend regressions; missing artifacts or runs without live
  evidence now fail graph generation.
- Rebased stale absolute artifact paths onto the current checkout whenever the
  declared artifact exists there, and convert runtime `used_pages` to tokens by
  multiplying by `page_size` (the recorded production samples use page size 1).

### Deployed — one persistent Ornith TQ3_4S 64K model service

- Audited user/system systemd units, processes and ports. Archived and removed
  14 obsolete user-level model runtimes plus three custom system Ollama/Gemma
  units; the recoverable archive SHA-256 is recorded in `TESTLOG.md` and the
  deployment artifact. Retained package units `llama-cpp.service` and
  `ollama.service` are masked.
- Added the source-controlled `deploy/systemd/freetoken-ornith.service` and its
  systemd contract tests. It is the sole enabled model-runtime unit and serves
  the pinned Ornith 1.5 35B TQ3_4S checkpoint at `0.0.0.0:1919`.
- Fixed total context, token storage and KV reserve to exactly 65,536 tokens;
  retained INT8 KV, radix prompt cache, automatic LRU expert sizing, one running
  request and the measured p2560 prefill candidate. Startup resolved 65,536 KV
  pages, 2,311 expert slots and 1.07 GiB free VRAM after graph capture.
- Recorded a fresh end-to-end 4,084-input/127-output repository smoke:
  24.873 s TTFT, 31.898 decode tok/s, 7,282 MiB peak VRAM, 100% maximum GPU and
  no OOM. This validates the combined service path, not full-context 64K speed.
- Recorded the output-budget boundary for the default thinking model: an
  eight-token cap produced only parsed reasoning and empty visible content,
  while a 96-token control returned the exact `23*9 = 207` result in 2.340 s.
- Preserved OmniRoute, DeepSeek Harness and Open WebUI because they are gateway,
  harness and UI layers rather than competing model runtimes.

## 2026-08-31 — Synced Turing upstream and verified sequential two-model runtime

### Upstream audit

- Merged upstream `main` at `3a20a79` and [FreeToken PR #24](https://github.com/FlashML-org/FreeToken/pull/24)
  at `35668da` into `feat/qwen4exp-gguf-turing`; merge commit: `429cf02`.
- PR #24 supplies the SM75 build gate, Turing-safe Triton attention tile
  choices and the Turing kernel-cache architecture needed by RTX 2070.
- Compared [PR #185](https://github.com/FlashML-org/FreeToken/pull/185): its
  long-prefill `gridDim.z` protection is already covered by local
  token-aligned GEMV chunking (`9952a39`), so it was not merged twice.
- Kept [PR #300](https://github.com/FlashML-org/FreeToken/pull/300) (KV
  ladder), [PR #309](https://github.com/FlashML-org/FreeToken/pull/309)
  (quantized KV) and [PR #311](https://github.com/FlashML-org/FreeToken/pull/311)
  (disk-backed Qwen PLE) as separate references. They target Qwen3.8 or newer
  KV paths and are not safe defaults for the current Ornith/Gemma profile.

### Implemented — full sequential GPU ownership

- Replaced the old Ornith MoE-only parking transition with an actual systemd
  stop/start lifecycle. Gemma GPU is started only after `freetoken-ornith.service`
  is inactive; returning to Ornith starts the service and verifies readiness plus
  the fixed `KV=65536`, `Mamba=8`, `SWA=0`, `MoE=2311` geometry.
- Removed the arbiter's hard `Requires=freetoken-ornith.service`; it now keeps
  the daemon as a requirement and Ornith as a wanted dependency. This permits
  the arbiter to stop Ornith without taking down the public `0.0.0.0:1919`
  endpoint.
- Reconciliation now fails closed on an active Ornith service with no ready
  endpoint, a stale ready endpoint from an inactive service, or any simultaneous
  Gemma-GPU/Ornith ownership. A parked-but-resident Ornith process is no longer
  accepted as safe alongside Gemma GPU.
- Fixed the CPU fallback unit to invoke the llama.cpp wrapper and set
  `LD_LIBRARY_PATH=/opt/llama-cpp/lib`; the previous direct `.real` invocation
  failed with `libllama-server-impl.so: cannot open shared object file`.

### Live results

- The previous park-only attempt remained loading until the 300-second timeout:
  with Ornith still resident, only about 3.7 GiB RAM was available and Gemma
  never reached readiness. No generation occurred.
- With full sequential stop, Gemma Q4_0 on FreeToken/RTX 2070 reached readiness
  at 12:29:40 and returned `Столица России — Москва.` through the public arbiter:
  52 prompt tokens, 7 completion tokens, HTTP 200, 85.555 s wall time including
  cold load/warmup. FreeToken logged 4.81 prefill tok/s for the short request;
  its dense model ignored MoE flags, used BF16 KV and had 3.81 GiB free VRAM
  after CUDA-graph capture.
- The reverse transition stopped the Gemma daemon, started Ornith, rebuilt
  active cache capacity and verified `65536` INT8 KV pages, `2311` MoE slots and
  `8` Mamba slots. The same short answer returned HTTP 200 in 82.659 s including
  reload.
- A direct CPU fallback check returned HTTP 200 in 4.827 s warm, with 13.126
  prompt tok/s and 9.051 decode tok/s; the CPU service was stopped afterwards.
- Final state is clean: arbiter, daemon and Ornith active; Gemma GPU/CPU stopped;
  arbiter counters `requests=2`, `completed=2`, `errors=0`.

The complete raw record, timestamps, topology and final-state evidence are in
`benchmarks/results/two-model-arbiter-2026-08-31/sequential-offload.json`.

### Post-reboot reconciliation observation

After a later host boot, HuggingVoice selected Gemma. Five early warmup attempts
hit the old boot-race behavior (`503 state_ambiguous`) while Ornith was still
loading; the eventual warmup reached HTTP 200 once Gemma became ready. The
arbiter was then restarted after the no-auto-start change: Gemma remained ready,
Ornith was not started, and a warm public request returned HTTP 200 in 0.207 s.
This is an observation, not a controlled benchmark; the raw record is
`benchmarks/results/two-model-arbiter-2026-08-31/post-reboot-reconcile-observation.json`.

## 2026-08-31 — Fixed Gemma speaker-memory calls and raised context to 8K

- Added a narrowly scoped Gemma arbiter policy for the exact nine official
  HuggingVoice `speaker_memory_*` tools and recognizable voice-policy markers.
  It replaces only that one system message with the checkpoint-specific
  minimal instruction; extra system messages, subsets, arbitrary same-prefix
  tools, ordinary chat, mixed tools and Ornith are unchanged.
- Added `<|tool_response>` to Gemma 4 GGUF EOG tokens, matching FreeToken
  upstream issue #201 and preventing the marker from leaking into content after
  a parsed native call.
- Added a non-mocked public `:1919` acceptance program with the real nine tool
  schemas. Textual claims of memory without structured `tool_calls` fail.
- Raised Gemma GPU KV/context and llama.cpp CPU-fallback context from 4,096 to
  8,192 tokens. The BF16 KV allocation is 0.18 GiB and leaves 3.75 GiB free
  VRAM after graphs on RTX 2070; response length remains request-controlled.
- Improved the real gate from 0/3 to 5/5. At 8K, the final reviewed warm repeat
  measured 2.675 s warm-only tool-call p50 and 0.078 s short-Russian streaming
  TTFT. A broader compact policy only reached 3/6, so reliability of the other
  eight tool choices is not claimed.
- Made the acceptance gate reproducible across cold 82-89 second switches with
  a 180-second default timeout and controlled JSON failures for transport,
  non-JSON, malformed arguments and streaming errors. A real post-restart
  connection-refused race is preserved as a negative artifact.
- Added unit coverage for prompt scoping, Gemma EOG and both 8K launch paths.
  The changed-scope suite passes `114 passed, 1 skipped`.
- Preserved all raw positive and negative runs under
  `benchmarks/results/gemma-tool-acceptance-2026-08-31/`.
