# Qwen3.8 E1–E4 Telemetry and QSA Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current Qwen3.8 Q4_K_M runtime observable per request and remove its 64K QSA-score allocation failure without changing model outputs or the TQ4 physical-attention contract.

**Architecture:** E1 extends the existing single-request MoE counters through the scheduler’s terminal-reply wire into `/v1/stats`; it does not introduce a second monitoring server. E2 replaces the per-head `torch.mm` score temporary with a CUDA-only Triton scorer that writes one head-reduced FP32 score per `(query, compressed block)`; the CPU reference remains the oracle. E3 exposes one bounded workspace setting and uses the same selection semantics at 8, 16 and 32 MiB. E4 adds a bounded, prompt-free route/copy summary to the same terminal snapshot and a repeatable Qwen benchmark runner.

**Tech Stack:** Python 3.12, PyTorch, Triton 3.6, pytest, FreeToken ZMQ message wire, NVIDIA Turing SM75.

**Spec:** `docs/superpowers/specs/2026-08-28-qwen38-turing-runtime-optimization-design.md`

## Global Constraints

- Target only the local Qwen3.8 Flash Next Q4_K_M GGUF path on RTX 2070 Mobile / 8 GiB VRAM, i7-8750H / 32 GiB RAM / NVMe.
- Retain FreeToken, the existing mixed-IQ Q4_K_M expert path, page size 4 and `tq4-nc`; do not use llama.cpp, NVFP4 or MTP.
- Do not merge upstream PR #257 or port its page-size-64/unpacked-KV kernels; port only semantically compatible code with differential tests.
- Telemetry mode is benchmark-only: require `--max-running-requests 1`, snapshot only at a terminal reply and do no device-to-host read per decode step.
- Qwen disables CUDA graphs today; the telemetry payload must explicitly report whether routing-frequency collection was enabled rather than silently producing incomplete data.
- Preserve prompt privacy: E4 records aggregate layer/copy counters only, never prompt text, hidden states, or unbounded per-token IDs.
- Do not claim a speed increase until an identical-seed, complete runtime measurement is saved in `benchmarks/results/` and described in `TESTLOG.md` and `CHANGELOG.md`.
- Keep the active Ornith service untouched until the Qwen benchmark runner is explicitly launched on another port or the user-approved research stop is required.

---

### Task 1: E1 counter snapshot and single-request guard

**Files:**
- Modify: `python/freetoken/server/args.py`
- Modify: `python/freetoken/engine/engine.py`
- Modify: `python/freetoken/moe/offload_cache.py`
- Create: `tests/moe/test_telemetry.py`
- Modify: `tests/server/test_parser_auto_selection.py`

**Interfaces:**
- Produces `EngineConfig.moe_collect_stats: bool` set by `--moe-collect-stats`.
- Produces `OffloadMoeCache.telemetry_snapshot() -> dict[str, object]` with `miss`, `per_layer`, `routing` and `routing_frequency_enabled` keys.
- Consumes existing `decode_miss_stats()`, `decode_miss_stats_per_layer()`, `decode_routing_stats()` and `decode_freq`.

- [ ] **Step 1: Write failing parser and cache tests.** Add a parser assertion that `parse_args(["--model", "x", "--moe-collect-stats"])` returns `args.moe_collect_stats is True`. Instantiate a CPU `OffloadMoeCache`, fill `decode_freq`, call `reset_stats()`, and assert it becomes all zero. Assert the snapshot has a `routing.method == "stationary_per_layer_top_c"` label and does not call the score a dynamic oracle.

- [ ] **Step 2: Run the focused tests to verify the missing behaviour.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_telemetry.py tests/server/test_parser_auto_selection.py -q`

Expected: FAIL because the CLI flag and `telemetry_snapshot` do not exist and because `reset_stats()` does not zero `decode_freq`.

- [ ] **Step 3: Implement the smallest data-plane change.** Add the CLI flag next to the existing MoE options. In `_adjust_config`, reject telemetry with `max_running_req != 1`; enable `cache.collect_decode_freq` only when telemetry is requested and `config.cuda_graph_bs` is empty. Make `reset_stats()` zero `decode_freq`. Implement `telemetry_snapshot()` as a one-time serializable read; label the stationary top-C estimate correctly and include cache geometry and route-frequency status.

- [ ] **Step 4: Run focused tests and the non-GPU regression subset.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_telemetry.py tests/server/test_parser_auto_selection.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the isolated counter change.**

Run: `git add python/freetoken/server/args.py python/freetoken/engine/engine.py python/freetoken/moe/offload_cache.py tests/moe/test_telemetry.py tests/server/test_parser_auto_selection.py && git commit -m "feat: expose bounded MoE telemetry"`

### Task 2: E1 scheduler-to-stats wire

**Files:**
- Modify: `python/freetoken/message/tokenizer.py`
- Modify: `python/freetoken/message/frontend.py`
- Modify: `python/freetoken/scheduler/scheduler.py`
- Modify: `python/freetoken/tokenizer/server.py`
- Modify: `python/freetoken/server/stats.py`
- Modify: `tests/server/test_message_wire.py`
- Create: `tests/server/test_stats.py`

**Interfaces:**
- Consumes `OffloadMoeCache.reset_stats()` before the first forward of a telemetry request and `telemetry_snapshot()` after `copy_done.synchronize()` on its finished final reply.
- Produces optional `moe_stats: dict[str, object] | None` on both `DetokenizeMsg` and `UserReply`.
- Produces top-level `/v1/stats["moe"]`, `null` until the first terminal telemetry request.

- [ ] **Step 1: Write failing wire and frontend tests.** Extend the existing round trip with `moe_stats={"schema_version": 1, "miss": {"miss_rate": 0.5}}`. Add a `StatsTracker` test that `observe()` retains this terminal payload and `build_stats()` returns it under `moe` without changing existing fields.

- [ ] **Step 2: Run the focused tests to verify they fail.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_message_wire.py tests/server/test_stats.py -q`

Expected: FAIL because neither message type carries the payload and `build_stats()` has no `moe` field.

- [ ] **Step 3: Implement the wire at the existing lifecycle boundaries.** On admission of the sole telemetry request call `cache.reset_stats()` before it enters `PrefillManager`; after `copy_done.synchronize()` and only on the final `DetokenizeMsg`, assign `cache.telemetry_snapshot()`. Propagate it to `UserReply`; have `StatsTracker.observe()` save only a non-null terminal payload and expose it as `moe`.

- [ ] **Step 4: Run the focused tests.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_message_wire.py tests/server/test_stats.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the wire change.**

Run: `git add python/freetoken/message/tokenizer.py python/freetoken/message/frontend.py python/freetoken/scheduler/scheduler.py python/freetoken/tokenizer/server.py python/freetoken/server/stats.py tests/server/test_message_wire.py tests/server/test_stats.py && git commit -m "feat: publish MoE telemetry in runtime stats"`

### Task 3: E2 exact fused QSA head-reduced scorer

**Files:**
- Modify: `python/freetoken/kernel/triton/qsa.py`
- Modify: `python/freetoken/attention/qsa.py`
- Modify: `tests/attention/test_qsa.py`

**Interfaces:**
- Produces `qsa_head_reduced_scores(index_q, compressed_keys, *, row_start: int, row_stop: int) -> torch.Tensor` for CUDA tensors, shape `[row_stop-row_start, blocks]`, FP32, value `sum(head, relu(dot(q, key))) * dim**-0.5`.
- Keeps `select_qsa_logical_rows(..., score_workspace_bytes: int | None = None)` as the public selection function and preserves output IDs/counts.
- CPU tensors use the present PyTorch reference; CUDA tensors use the new scorer and never allocate the old `[rows * heads, blocks]` dot matrix.

- [ ] **Step 1: Write the failing CPU selection API test.** Add a deterministic input without tied scores and call `select_qsa_logical_rows` at two workspace sizes. Assert both `selected` and `counts` are exactly equal. Add a CUDA-gated test at head dimension 256 comparing the fused score tensor and final selected IDs to the existing explicit reference calculation.

- [ ] **Step 2: Run the CPU test to verify the new argument is absent.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/attention/test_qsa.py -q`

Expected: FAIL at the new `score_workspace_bytes` argument; pre-existing QSA failures are separately recorded baseline failures and must not be hidden.

- [ ] **Step 3: Implement the fused scorer without changing physical attention.** Add one Triton program per `(row, compressed-block tile)`, loop over index heads and head dimension, accumulate reduced FP32 scores, and write only the reduced score. Use a fixed tile chosen for SM75; mask incomplete block tails. Have CUDA selection call it per bounded row chunk before the existing visibility mask/top-k/compaction. Do not alter `qsa_tq4_sparse_gqa`, `QSAKVCache`, page size or logical-to-physical mapping.

- [ ] **Step 4: Run selection tests.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/attention/test_qsa.py -q`

Expected: new workspace-independence test passes; known baseline import failures remain listed separately until branch reconciliation.

- [ ] **Step 5: Commit the scorer.**

Run: `git add python/freetoken/kernel/triton/qsa.py python/freetoken/attention/qsa.py tests/attention/test_qsa.py && git commit -m "fix: bound Qwen QSA score workspace"`

### Task 4: E3 workspace configuration and capacity sweep

**Files:**
- Modify: `python/freetoken/engine/config.py`
- Modify: `python/freetoken/server/args.py`
- Modify: `python/freetoken/core.py`
- Modify: `python/freetoken/engine/engine.py`
- Modify: `python/freetoken/attention/qsa.py`
- Modify: `tests/server/test_parser_auto_selection.py`
- Modify: `tests/attention/test_qsa.py`

**Interfaces:**
- Produces `EngineConfig.qsa_score_workspace_mib: int = 16`; CLI `--qsa-score-workspace-mib` accepts positive integers only.
- Produces `Context.qsa_score_workspace_bytes: int`, initialized once from config before the attention backend is created.
- `QSAAttnBackend._select_physical_rows()` passes this byte budget to `select_qsa_logical_rows()`.

- [ ] **Step 1: Write failing config tests.** Assert the parser accepts `--qsa-score-workspace-mib 8` and rejects `0`; construct a `Context` with the configured byte value and assert a selection caller receives `8 << 20`.

- [ ] **Step 2: Run tests to verify failure.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_parser_auto_selection.py tests/attention/test_qsa.py -q`

Expected: FAIL because no CLI/config/context field exists.

- [ ] **Step 3: Implement only the explicit propagation.** Add the config field and parser validator. Store `mib << 20` on `Context` immediately after context creation; reject non-positive values in `_adjust_config`; use it in QSA. Do not use module globals and do not dynamically resize a captured buffer.

- [ ] **Step 4: Run tests.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_parser_auto_selection.py tests/attention/test_qsa.py -q`

Expected: new tests PASS.

- [ ] **Step 5: Commit the configurability change.**

Run: `git add python/freetoken/engine/config.py python/freetoken/server/args.py python/freetoken/core.py python/freetoken/engine/engine.py python/freetoken/attention/qsa.py tests/server/test_parser_auto_selection.py tests/attention/test_qsa.py && git commit -m "feat: configure QSA score workspace"`

### Task 5: E4 bounded aggregate route/copy trace and reproducible runner

**Files:**
- Modify: `python/freetoken/moe/offload_cache.py`
- Modify: `python/freetoken/scheduler/scheduler.py`
- Modify: `python/freetoken/server/stats.py`
- Create: `benchmarks/qwen38_turing_profile.py`
- Create: `tests/benchmarks/test_qwen38_turing_profile.py`

**Interfaces:**
- Produces a bounded telemetry `trace` object with separate prefill/decode layer aggregate counters: `route_references`, `route_unique`, `l1_hits`, `l1_misses`, `copy_records`, `copy_bytes` and `evictions`.
- Adds process counters to each run record: elapsed time, prompt/decode tok/s, `/proc/<pid>/io.read_bytes`, major/minor faults and sampled GPU utilisation/memory.
- Produces one JSON result per fixed-seed context point under `benchmarks/results/YYYY-MM-DD-qwen38-*/`; the runner never writes model files or result data to `/tmp`.

- [ ] **Step 1: Write failing pure-unit tests for trace aggregation and result parsing.** Feed a small fake layer sequence into the trace collector and assert no raw IDs are retained, only counters. Feed a synthetic `/proc` record to the runner parser and assert bytes/fault deltas and `null` handling are correct.

- [ ] **Step 2: Run tests to verify failure.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q`

Expected: FAIL because the collector and runner do not exist.

- [ ] **Step 3: Implement aggregate-only collection and runner.** Increment per-layer route counts immediately before expert IDs are rewritten, measure unique IDs per `ensure_experts` call, and derive hit/miss/copy/eviction deltas from existing cache state. Reset and snapshot with E1. The runner starts Qwen on an explicit unused local port, uses temperature 0 and a fixed seed, writes stdout/stderr plus JSON under its result directory, then invokes `/v1/stats` after terminal completion.

- [ ] **Step 4: Run unit tests.**

Run: `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the runner and trace.**

Run: `git add python/freetoken/moe/offload_cache.py python/freetoken/scheduler/scheduler.py python/freetoken/server/stats.py benchmarks/qwen38_turing_profile.py tests/benchmarks/test_qwen38_turing_profile.py && git commit -m "feat: profile Qwen expert routing and transfers"`

### Task 6: Live capacity and expert-use measurements

**Files:**
- Create: `benchmarks/results/YYYY-MM-DD-qwen38-e1-e4/*`
- Modify: `TESTLOG.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes Tasks 1–5 and the same Q4_K_M model, TQ4-NC KV, page size 4, one running request, seed and prompt fixture for every point.
- Produces raw service logs, JSON telemetry, environment metadata and a one-row summary for each complete or failed point.

- [ ] **Step 1: Capture pre-change control only when the current server can be reproduced.** Record model SHA/config, GPU driver, FreeToken commit, prompt SHA and exact command. Never compare an incomplete 45K control as a completed 64K throughput result.

- [ ] **Step 2: Run the QSA workspace sweep.** At 64K with 1,024-token chunks test 8, 16 and 32 MiB in ascending order; stop increasing after a capacity failure. Record completion, TTFT, steady prefill tok/s, peak allocated/reserved VRAM and QSA pass count.

- [ ] **Step 3: Run the fixed-seed expert/profile matrix.** Run complete 1K, 16K, 64K and 112K context points. Generate 256 decode tokens after each prompt. Record prompt/decode speed, MoE miss/routing/aggregate trace, read bytes, faults, GPU/CPU use and error output. Do not run 112K if 64K fails.

- [ ] **Step 4: Interpret against the gates.** Compare static per-layer top-C to realized L1 hit rate; only propose a cache-policy experiment if the gap is above five percentage points. Compare physical expert bytes/token against the 53.32 MiB/token full-coverage floor and use the trace to decide whether cache capacity, deduplication or NVMe throughput is dominant.

- [ ] **Step 5: Record every result, then update upstream and publish branch commits.** Append successful and failed experiments to `TESTLOG.md`; append user-relevant implementation decisions to `CHANGELOG.md`. Run `git fetch upstream --prune`, rebase/merge only if a safe relevant upstream change exists, run the verification suite, then push the feature branch to `origin`.

## Plan Self-Review

| Spec requirement | Implementing task |
| --- | --- |
| Existing telemetry first; no new monitoring server | Tasks 1–2 |
| QSA local page-4/TQ4 capacity fix | Tasks 3–4 |
| 8/16/32 MiB ascending sweep; no initial 48 MiB | Task 6 |
| Bounded expert routing/copy evidence before cache policy | Task 5–6 |
| Complete 1K/16K/64K/112K and 256-token decode measurements | Task 6 |
| Preserve raw evidence and record failures | Task 6 |
| No upstream wholesale merge, no llama.cpp/NVFP4/MTP | Global constraints |

No placeholders, cross-task interface name conflicts or unowned spec requirements remain after this review. The existing non-QSA branch test failures are baseline evidence and are deliberately kept out of this plan: the focused new tests must pass, and the final report will enumerate the unrelated failing tests rather than implying a clean full suite.
