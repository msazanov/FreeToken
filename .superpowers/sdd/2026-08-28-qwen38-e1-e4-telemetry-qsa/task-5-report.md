# Task 5 report — E4 aggregate trace and Qwen profiler

Status: implemented and committed in the Task 5 source/test scope.

## Delivered

- `moe.trace` now has fixed `prefill` and `decode` layer arrays. Each layer retains only `route_references`, `route_unique`, `l1_hits`, `l1_misses`, `copy_records`, `copy_bytes`, and `evictions`; no prompt content or expert IDs are serialized.
- The scheduler selects the trace phase at the existing E1 pre-forward seam. The existing terminal reply and `/v1/stats` wire carry the extended snapshot without a second monitoring path.
- `benchmarks/qwen38_turing_profile.py` creates one result JSON and paired child stdout/stderr logs per fixed-seed context point below `benchmarks/results/YYYY-MM-DD-qwen38-*/`. It uses a selected explicit localhost port, forces temperature zero, requests the fixed seed, samples process/GPU counters, fetches `/v1/stats` only after the terminal completion, and terminates only its own Qwen child PID.

## Verification

- `/home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q` — `2 passed`
- `PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_telemetry.py tests/server/test_stats.py tests/server/test_message_wire.py -q` — `19 passed`
- `py_compile` for the new runner and edited telemetry/scheduler modules — passed.
- `git diff --check` — passed.

## Concerns

- No live Qwen server or GPU benchmark was run, as required for this task handoff.
- The shared `/home/random/freetoken-turing/.venv` is editable against the sibling `freetoken-turing` checkout. The focused Task 5 test pins this workspace's `python/` directory so the required command tests the Task 5 code. Broader checks must likewise set `PYTHONPATH` until that virtual-environment binding is corrected.

## Fix round 1 — reviewer follow-up

- Standard host-bank prefill now records aggregate route references, unique routes, L1 hits/misses immediately before the full-layer path. `materialize_layer()` counts only rows displaced from another layer as evictions. The production cache-method seam has a CPU regression test; no IDs enter the serialized trace.
- OpenAI chat and completion requests now carry an optional non-negative `seed` through `SamplingParams`. Non-greedy seeded requests keep a per-request `torch.Generator` and use the explicit torch sampling path, so the value controls a persistent RNG stream instead of being silently discarded. The profiler remains temperature-zero/greedy as required; it sends the seed but correctly does not claim that greedy argmax consumed randomness.
- The Qwen child environment now removes `FREETOKEN_API_LOG_DIR` before import/startup, preventing request-body logs from escaping the result directory. A pure runner test covers that sanitization.

### Fix-round verification

- `PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_telemetry.py tests/benchmarks/test_qwen38_turing_profile.py -q` — `7 passed`.
- Focused OpenAI seed plus Responses/Anthropic adapter checks — `4 passed` (one upstream FastAPI/httpx deprecation warning).
- CPU-only seeded torch-sampling probe returned identical samples for two generators seeded with `73`.
