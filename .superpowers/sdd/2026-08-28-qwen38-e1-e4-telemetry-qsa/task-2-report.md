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
