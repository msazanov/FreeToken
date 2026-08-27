# RTX 2070 Fork Changelog

This changelog is append-only for experiments as well as code. Keep successful,
failed and inconclusive hypotheses; do not rewrite history.

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
