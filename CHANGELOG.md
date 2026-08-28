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
