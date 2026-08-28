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
