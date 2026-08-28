# Task 2 Report: E1 scheduler-to-stats wire

## Status

Implemented the Task 2 scheduler-to-stats MoE telemetry wire on the current
branch. No model server, GPU runtime, OpenCode, or OmniRoute was used.

## Changes

- Added optional `moe_stats: dict[str, object] | None` to `DetokenizeMsg` and
  `UserReply`.
- Reset the engine's MoE telemetry immediately before
  `prefill_manager.add_one_req(msg)` after prompt-length validation, guarded by
  `config.moe_collect_stats`.
- After `copy_done.synchronize()`, snapshot telemetry only when a finished
  `DetokenizeMsg` exists, and attach the snapshot only to finished replies.
- Propagated `moe_stats` from `DetokenizeMsg` through tokenizer conversion to
  `UserReply`.
- Made `StatsTracker` retain only non-null terminal telemetry and expose it as
  top-level `/v1/stats["moe"]`, initially `null`.
- Extended the message round-trip tests and added `tests/server/test_stats.py`.

## TDD evidence

The required red test command initially failed for the intended missing
behavior: both message constructors rejected `moe_stats`, and `build_stats()`
had no `moe` key. The worktree-targeted green command then passed all 9
focused tests.

## Verification

Passing focused command:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_message_wire.py tests/server/test_stats.py -q
9 passed
```

Passing related Task 1 regression:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_telemetry.py -q
3 passed in 3.55s
```

`git diff --check` and targeted `compileall` both passed.

## Concerns

- The brief's exact virtualenv command, when run without `PYTHONPATH`, imports
  the installed package from `/home/random/freetoken-turing` instead of this
  worktree and therefore reports the expected old-code failures. The passing
  command explicitly places this worktree's `python/` directory first.
- A broader combined server regression run printed 15 dots and then stopped
  producing output; no Python process remained visible, so it was terminated
  with status 130 and is not claimed as passing. The required focused suite and
  the separate telemetry regression are the verified gates.
- The pre-existing untracked plan file was preserved and not included in the
  Task 2 changes.

## Fix Round 1

Addressed all review findings:

- Moved `reset_stats()` out of `UserMsg` queue handling. The scheduler now
  resets only at the `_forward()` seam for a prefill `Batch` carrying
  `prompt_admissions`, immediately before `engine.forward_batch()`. A queued
  request therefore cannot clear the counters of a running request.
- Both message encoders omit `moe_stats` when it is `None`, allowing an old
  peer to decode telemetry-disabled new messages. New message decoders retain
  the default `None` when reading old payloads without the optional key.
- Added compatibility tests for old payloads and absent optional fields, plus
  lifecycle tests proving queue isolation and `copy_done.synchronize()` before
  the terminal snapshot.

Fix-round TDD evidence:

- Red: the focused suite failed on absent-field serialization and queue-time
  reset; the first lifecycle fixture failure was corrected as a test-fixture
  issue before rerunning.
- Green: the focused command below passed 14 tests.

Final Fix Round 1 verification:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_message_wire.py tests/server/test_stats.py -q
14 passed in 3.60s
```

The pre-existing untracked plan remains preserved. No model server, GPU,
OpenCode, or OmniRoute was used.

## Fix Round 2

Addressed the remaining batch-wrapper compatibility finding:

- The message encoder now recursively removes `moe_stats` only when it is
  `None` on serialized `UserReply` or `DetokenizeMsg` objects, including those
  nested inside `BatchFrontendMsg` and `BatchTokenizerMsg`. Arbitrary client
  dictionaries are not filtered by this compatibility pass.
- Added regression tests for both batch wrapper types. Each test verifies that
  absent nested telemetry produces no nested `moe_stats` key and that an old
  peer-style payload without the optional field remains decodable by the new
  peer.

Fix-round TDD evidence:

- Red: the new batch tests failed because nested message dictionaries still
  contained `moe_stats: None`.
- Green: the final focused command below passed 16 tests.

Final Fix Round 2 verification:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_message_wire.py tests/server/test_stats.py -q
16 passed
```

## Fix Round 3

Addressed the compatibility regression introduced by Fix Round 2:

- The omission logic now traverses original Python message instances in
  parallel with their serialized values. It removes `moe_stats` only from
  actual `UserReply`/`DetokenizeMsg` instances and their `Batch*` children.
- Free-form dictionaries are not traversed, so nested client data containing
  `{"__type__": "DetokenizeMsg", "moe_stats": None}` remains unchanged in
  `chat_template_kwargs` and tool schemas.
- Retained both batch compatibility tests and extended the hostile-dictionary
  regression to cover this exact nested payload.

Fix-round TDD evidence:

- Red: the hostile `TokenizeMsg` regression failed because the serialized
  dictionary scrub deleted the nested client field.
- Green: the final focused command passed 16 tests.

Final Fix Round 3 verification:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/server/test_message_wire.py tests/server/test_stats.py -q
16 passed
```
