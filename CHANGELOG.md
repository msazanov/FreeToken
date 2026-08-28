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
