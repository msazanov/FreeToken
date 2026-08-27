# Ornith Agent Runtime Optimization Design

## Goal

Maximize useful coding-agent throughput and answer quality on RTX 2070 Mobile 8
GiB, i7-8750H, 32 GiB DDR4 and NVMe using DeepSeek Harness, Ornith 1.5 35B A3B
and FreeToken. Measurements, not a target context number, decide deployment
settings.

## Constraints

- FreeToken remains the only inference runtime; the active model stays Ornith
  1.5 35B A3B with TQ4-NC KV.
- DeepSeek Harness remains the agent loop. Use its existing pruner and
  compaction plugins rather than changing the agent loop.
- Every performance point records hardware, commit, dirty state, server cache
  geometry, prompt shape, cache hit fields, TTFT, prefill, decode, quality and
  raw artifact path.
- Context loss is unacceptable without a measured quality comparison on a real
  repository task. Summaries retain file paths, commands, decisions, errors,
  unresolved work and source references.
- Never restart or resize the live user FreeToken service solely for an
  experiment without an explicit maintenance step and a recorded before/after
  configuration.

## Architecture

The benchmark layer drives the existing OpenAI-compatible FreeToken API with a
fixed harness-shaped prefix, an exact repeated request, and an append-only turn.
It records cache reuse separately from cold prefill, so a large cache hit cannot
be mistaken for a faster attention kernel. A repository-compression quality task
is reused at each target context.

Harness configuration is an out-of-tree Cordis patch. It first loads the
model-free tool-result pruner, then configures the existing basic compaction
provider for the local Ornith route. The selected retention threshold is derived
from the cache/quality matrix. This keeps the system/tools prefix stable and
uses only the changing history as the compaction replacement range.

FreeToken tuning is gated by measured stage attribution. First collect cache,
MoE and host/GPU evidence. Only if a stage accounts for material wall time do
we change cache geometry, expert residency, or Turing kernels. Experimental KV
compression below TQ4-NC is excluded until a quality and end-to-end speed probe
proves a gain.

## Evaluation matrix

1. Cache: cold, exact warm replay, and append-only delta on the same
   harness-shaped prompt at 1K, 16K, 64K and 112K.
2. Context packing: original history, pruned tool results, and compacted
   checkpoint at a matched task. Compare 16K, 32K, 48K and 64K retained
   contexts.
3. Runtime allocation: only after selecting the working retained context,
   compare the current 122880-page / 1429-slot geometry with one or more
   context-for-MoE alternatives at the same effective prompt length.
4. Kernel work: only after stage telemetry identifies attention or MoE as the
   bottleneck; all changes use matched cold and warm runs plus the quality task.

## Success criteria

- A saved cache matrix demonstrates the exact cache-hit behaviour of the
  harness-shaped prefix and append-only turns.
- The deployed Harness policy has a real composition/configuration artifact,
  never an undocumented prompt rewrite.
- The chosen context budget has equal or better repository-task quality than
  the 64K reference and materially lower TTFT than the 112K cold baseline.
- Each adopted runtime change improves a matched metric; rejected hypotheses
  remain in `TESTLOG.md` and `CHANGELOG.md` with raw evidence.

## Risks and controls

Summarization can lose source facts; preserve path/command/error anchors and
evaluate them. Cache hits can be invalidated by dynamic system data or rewritten
history; use token-identical prefix checks. Extra MoE slots trade away KV pages;
test only after semantic packing makes those pages unnecessary. Quantizing KV
further can exchange capacity for slower dequantization or quality loss; it is a
separate gated experiment, not a default.
