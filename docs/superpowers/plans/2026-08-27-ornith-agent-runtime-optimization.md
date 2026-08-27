# Ornith Agent Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish measurable cache, context-packing and runtime-allocation decisions for DeepSeek Harness + Ornith + FreeToken on RTX 2070.

**Architecture:** A FreeToken benchmark records cold, warm and append-only prompt requests with the same real repository task. A Harness Cordis patch uses existing tool-result pruning and compaction only after the baseline establishes an evidence-based target. Runtime cache and kernel changes are conditional on the resulting stage evidence.

**Tech Stack:** Python stdlib SSE client, FreeToken OpenAI API and stats endpoints, JSONL benchmark artifacts, DeepSeek Harness Cordis configuration, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-ornith-agent-runtime-optimization-design.md`

## Global Constraints

- Do not alter the live FreeToken service without an explicit recorded maintenance action.
- Keep raw benchmark data below `benchmarks/results/`, never tmpfs.
- `temperature=0` means greedy deterministic sampling; do not claim API seed control until FreeToken forwards it.
- Update `TESTLOG.md` and `CHANGELOG.md` for every completed or rejected experiment.

---

### Task 1: Harness-shaped cache benchmark

**Files:**
- Create: `benchmarks/ornith_harness_cache_bench.py`
- Create: `tests/benchmarks/test_ornith_harness_cache_bench.py`
- Modify: `TESTLOG.md`

**Interfaces:**
- Consumes: FreeToken `/v1/chat/completions`, `/v1/stats`, `/v1/cache/status`.
- Produces: one JSON artifact per cold/exact-warm/append case, and a JSONL slice index.

- [ ] Write tests for canonical static-prefix construction, case-marker uniqueness, SSE usage parsing and cache-rate calculation.
- [ ] Run the focused pytest file and observe each missing implementation failure.
- [ ] Implement the minimal runner with fixed prefix, real repository dossier, one exact replay and one suffix-only user delta.
- [ ] Run focused tests, then execute 1K/16K/64K/112K cache cases sequentially against `:1919`.
- [ ] Append metrics and interpretation to `TESTLOG.md` and `CHANGELOG.md`; commit the runner and raw artifacts.

### Task 2: Inspect and compose Harness retention policy

**Files:**
- Read: the active DeepSeek Harness profile and effective `dsh --profile web --dump-config` output.
- Create: one out-of-tree local Cordis patch only after identifying the active profile.
- Modify: Harness config only when its target file is known and not occupied by unrelated user changes.

**Interfaces:**
- Consumes: existing `dsh-compaction-tool-result-pruner` and `dsh-compaction-basic` plugins.
- Produces: a local-Ornith model policy with explicit threshold and retained-tail tokens.

- [ ] Record the active profile, current compaction/pruner rows and dynamic prompt contributors.
- [ ] Select retention candidates from Task 1 and create a patch that mounts the pruner before basic compaction.
- [ ] Add a harness-shaped repository quality test at each candidate retained budget.
- [ ] Verify the effective configuration and one actual compacted session; record cache and quality outcome before adopting it.

### Task 3: Stage attribution and MoE allocation probe

**Files:**
- Create: `benchmarks/ornith_runtime_stage_profile.py`
- Create: `tests/benchmarks/test_ornith_runtime_stage_profile.py`
- Modify: `TESTLOG.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: request stats, cache geometry, GPU/host samples and FreeToken MoE counters exposed by the active runtime.
- Produces: per-context stage evidence and a ranked bottleneck report; it does not alter service geometry.

- [ ] Write tests for profile-record normalization and bottleneck ranking from synthetic samples.
- [ ] Run tests red, implement the read-only collector, then run tests green.
- [ ] Capture matched 16K/64K long-context runs and compare attention-growth, cache and MoE signals.
- [ ] If MoE misses dominate decode, create one separate geometry experiment; otherwise retain current geometry and record the rejection.

### Task 4: Conditional runtime optimization

**Files:**
- Modify only the FreeToken subsystem identified by Task 3.
- Modify: matching focused tests, `TESTLOG.md`, `CHANGELOG.md`.

**Interfaces:**
- Consumes: Task 3 bottleneck evidence and the Task 1/2 fixed workload.
- Produces: a matched old/new measurement point linked to a commit and parameter label.

- [ ] Write a failing focused test for the one chosen runtime behaviour.
- [ ] Implement one isolated change.
- [ ] Run focused tests and matched cold/warm performance plus repository-quality benchmarks.
- [ ] Keep the change only if it improves the target metric without a quality regression; otherwise revert it and preserve the failed result.
