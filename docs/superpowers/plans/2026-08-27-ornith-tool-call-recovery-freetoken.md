# Ornith Tool-Call Recovery: FreeToken Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End Ornith generations as successful tool calls when a complete Qwen3-Coder invoke omits its outer wrapper, without sacrificing parallel calls or scheduler cleanup.

**Architecture:** The detector reports structure; the protocol-neutral generation loop counts tokens and decides when to stop; a dedicated successful-stop wire releases scheduler resources without producing an abort error. `StatsTracker` records bounded active and settled recovery diagnostics.

**Tech Stack:** Python 3.11+, asyncio, pytest, FastAPI, FreeToken tokenizer/scheduler message buses.

**Spec:** `docs/superpowers/specs/2026-08-27-ornith-tool-call-recovery-design.md`

## Global Constraints

- Recover only calls closed by `</function>` whose final argument ledger is valid JSON.
- Bare-call grace is 32 completion tokens by default; parser silence guard is 128 tokens.
- All three OpenAI-compatible protocol adapters consume the shared semantic event stream.
- Existing client-abort behavior and accounting must remain unchanged.
- Raw model output remains loopback-only and bounded.

---

### Task 1: Detector Recovery State

**Files:**
- Modify: `python/freetoken/server/function_call_parser.py`
- Test: `tests/server/test_streaming_model_matrix.py`

**Interfaces:**
- Produces: immutable `ToolParserControlState` with `complete_calls`, `awaiting_bare_continuation`, `withholding`, and optional `terminal_reason`.
- Produces: `FunctionCallParser.control_state()` and `FunctionCallParser.consume_terminal_reason()`.
- Consumes: existing `InvokeParamStreamMixin` call ledger and stream buffer.

- [ ] **Step 1: Write failing parser tests**

Add literal fixtures for wrapped termination, two bare parallel invokes, split markers, invalid `<tool\n` continuation, and an incomplete invoke. Assert ordered call fragments plus control state; assert the incomplete invoke never reports a terminal decision.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/server/test_streaming_model_matrix.py -k 'qwen3_coder and recovery_control'`

Expected: failures because `control_state()` and terminal decisions do not exist.

- [ ] **Step 3: Implement the detector state**

Track whether the current block was entered through a real outer opener or bare invoke. After a valid bare `</function>`, enter `awaiting_bare_continuation`; accept only whitespace and prefixes of `</tool_call>`, `<function=`, or `<tool_call>`. Report `invalid_continuation` without exposing debris. A real outer close reports `explicit_close`. Keep the default control state inert for every other detector.

- [ ] **Step 4: Verify GREEN and parser regression suite**

Run: `pytest -q tests/server/test_streaming_model_matrix.py tests/server/test_function_call_parser.py`

Expected: all selected parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/server/function_call_parser.py tests/server/test_streaming_model_matrix.py
git commit -m "fix: detect recoverable Ornith tool boundaries"
```

### Task 2: Successful Scheduler Stop Protocol

**Files:**
- Modify: `python/freetoken/message/backend.py`
- Modify: `python/freetoken/message/tokenizer.py`
- Modify: `python/freetoken/message/__init__.py`
- Modify: `python/freetoken/tokenizer/server.py`
- Modify: `python/freetoken/scheduler/scheduler.py`
- Modify: `python/freetoken/server/api_server.py`
- Test: `tests/server/test_message_wire.py`
- Test: `tests/scheduler/test_abort_inflight_prefill.py`
- Test: `tests/server/test_prepare_stop_accounting.py`

**Interfaces:**
- Produces: `StopGenerationMsg(uid)`, `StopGenerationBackendMsg(uid)`, and `GenerationStoppedMsg(uid)`.
- Produces: `FrontendManager.stop_generation(uid)`, which sends the stop control without deleting acknowledgement state or marking client abort.
- Consumes: the scheduler's existing race-safe request removal/free path.

- [ ] **Step 1: Write failing message and scheduler tests**

Round-trip each new dataclass through the existing serializers. Add scheduler fixtures proving a stop removes queued/decode/in-flight requests, emits `GenerationStoppedMsg` only after prior sampled output, and is idempotent. Add frontend accounting coverage proving the terminal reply completes rather than aborts the request.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/server/test_message_wire.py tests/scheduler/test_abort_inflight_prefill.py tests/server/test_prepare_stop_accounting.py`

Expected: collection/import failures for the missing stop messages or behavioral assertion failures.

- [ ] **Step 3: Implement the stop wire**

Add tokenizer/backend messages, translate them in `tokenizer/server.py`, and factor scheduler request removal so abort and successful stop share the same resource-safety path while retaining separate tombstones and terminal reply sets. Convert `GenerationStoppedMsg` to `UserReply(finished=True, finish_reason="stop")`; `FrontendManager.stop_generation()` sends immediately and leaves acknowledgement maps intact.

- [ ] **Step 4: Verify GREEN and scheduler regressions**

Run: `pytest -q tests/server/test_message_wire.py tests/scheduler/test_abort_inflight_prefill.py tests/server/test_prepare_stop_accounting.py tests/scheduler/test_scheduler_chunked_prefill.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/message python/freetoken/tokenizer/server.py python/freetoken/scheduler/scheduler.py python/freetoken/server/api_server.py tests/server/test_message_wire.py tests/scheduler/test_abort_inflight_prefill.py tests/server/test_prepare_stop_accounting.py
git commit -m "feat: add successful generation stop control"
```

### Task 3: Token-Bounded Early Termination

**Files:**
- Modify: `python/freetoken/server/generation.py`
- Modify: `python/freetoken/server/args.py`
- Test: `tests/server/test_streaming_model_matrix.py`
- Test: `tests/server/test_generation_accounting.py`
- Test: `tests/server/test_parser_auto_selection.py`

**Interfaces:**
- Consumes: `FunctionCallParser.control_state()`, `consume_terminal_reason()`, and `FrontendManager.stop_generation(uid)`.
- Produces: server config `tool_call_bare_grace_tokens: int = 32` and `tool_call_silent_guard_tokens: int = 128`.
- Produces: exactly one `ToolCallsDelta` followed by `GenDone("tool_calls", ...)` after successful stop acknowledgement.

- [ ] **Step 1: Write failing generation tests**

Use real parser output and a fake acknowledgement stream with literal token deltas. Prove continuation at token 31 remains eligible, token 32 stops, invalid continuation stops immediately, parallel invoke resets grace, incomplete invoke reaches `length`, racing post-decision output is not emitted, and prompt/completion accounting includes the terminal acknowledgement.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/server/test_generation_accounting.py tests/server/test_streaming_model_matrix.py -k 'recovery or early_stop or grace or silent_guard'`

Expected: behavioral failures because generation does not stop from parser control state.

- [ ] **Step 3: Implement minimal generation control**

Count `ack.completion_tokens_delta` from the transition into bare grace. After each parsed chunk, evaluate explicit/invalid decisions, 32-token grace, then the 128-token withholding guard. Close the open call, dispatch one successful stop, suppress later sampled text, wait for `finished`, and override the semantic finish reason to `tool_calls`. Add non-negative integer CLI arguments; zero disables the corresponding threshold.

- [ ] **Step 4: Verify GREEN and all wire adapters**

Run: `pytest -q tests/server/test_generation_accounting.py tests/server/test_streaming_model_matrix.py tests/server/test_openai_api.py tests/server/test_anthropic_api.py tests/server/test_responses_api.py tests/server/test_parser_auto_selection.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/server/generation.py python/freetoken/server/args.py tests/server/test_generation_accounting.py tests/server/test_streaming_model_matrix.py tests/server/test_parser_auto_selection.py
git commit -m "feat: stop decode after recoverable tool calls"
```

### Task 4: Recovery Telemetry and Settled Debug Tail

**Files:**
- Modify: `python/freetoken/server/stats.py`
- Modify: `python/freetoken/server/generation.py`
- Test: `tests/server/test_stats.py`
- Test: `tests/server/test_control_api.py`

**Interfaces:**
- Produces: `StatsTracker.observe_parser_state(uid, ...)` and `StatsTracker.observe_tool_recovery(uid, reason, tool_count, grace_tokens)`.
- Produces: public recovery counters/state in `build_stats()` and active-or-settled data from `stream_debug_snapshot()`.

- [ ] **Step 1: Write failing telemetry tests**

Assert active parser state and silent-token counts, cumulative recovery counters, stop-dispatch failures, a settled raw tail retained for 600 seconds, expiry at 601 seconds, and immediate clearing when the next request starts. Assert public stats never contain raw text.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/server/test_stats.py tests/server/test_control_api.py -k 'recovery or settled or parser'`

Expected: failures for missing fields and current immediate tail deletion.

- [ ] **Step 3: Implement bounded telemetry**

Add recovery/parser fields to `_ActiveInference`, process counters to `StatsTracker`, and a copied settled snapshot with monotonic expiry. Feed parser state from the generation loop. Keep raw bytes exclusive to the existing loopback route.

- [ ] **Step 4: Verify GREEN and full server tests**

Run: `pytest -q tests/server`

Expected: all server tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/server/stats.py python/freetoken/server/generation.py tests/server/test_stats.py tests/server/test_control_api.py
git commit -m "feat: expose Ornith tool recovery telemetry"
```

### Task 5: FreeToken Integration Verification

**Files:**
- Verify only; no planned source changes.

**Interfaces:**
- Consumes the complete FreeToken implementation from Tasks 1–4.
- Produces evidence for the Harness integration plan.

- [ ] **Step 1: Run focused CPU-safe suite**

Run: `pytest -q tests/server tests/scheduler/test_abort_inflight_prefill.py tests/scheduler/test_scheduler_chunked_prefill.py`

Expected: all tests pass.

- [ ] **Step 2: Run formatting/static checks used by the repository**

Run: `python -m compileall -q python/freetoken && git diff --check`

Expected: exit code 0.

- [ ] **Step 3: Review the complete diff against the spec**

Check every finish-reason, partial-call, parallel-call, cancellation, telemetry, and privacy requirement. Fix only through a new failing test.

- [ ] **Step 4: Commit any test-driven review fixes**

If review required code changes, stage only the FreeToken source/test files changed by that new
failing-test cycle and commit them with `git commit -m "fix: harden Ornith tool recovery"`. If
review found no defect, do not create an empty commit.
