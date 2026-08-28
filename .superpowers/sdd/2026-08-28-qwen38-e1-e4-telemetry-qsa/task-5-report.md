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

## Fix round 2 — seeded nucleus boundary and production seams

- Seeded torch sampling now keeps the first token whose inclusion crosses the `top_p` boundary, matching the kernel's nucleus semantics. The direct regression uses probabilities `[0.6, 0.4]` with `top_p=0.8` and proves both remain eligible.
- Added a CPU-only test through `Sampler.prepare()` and `Sampler.sample()` that verifies two independently created seeded requests produce the same sampled sequence and retain their per-request generator.
- Added a CPU-only `OffloadMoELayer._prefill_routed()` standard-host-bank seam test. It proves that actual layer method records aggregate prefill route references, unique routes, hits, and misses before materialization; the snapshot retains no raw route IDs or prompt text.

### Fix-round verification

- Red phase: `PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/engine/test_sample.py::test_seeded_top_p_retains_first_crossover_token -q` — failed as expected before the mask fix: observed `[1, 0]` rather than `[0.6, 0.4]`.
- `PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/engine/test_sample.py tests/moe/test_telemetry.py tests/server/test_openai_api.py::test_openai_request_seed_reaches_sampling_parameters tests/benchmarks/test_qwen38_turing_profile.py -q` — `11 passed` (one existing FastAPI/httpx deprecation warning).
- `py_compile` for the edited sampler and added CPU tests, plus `git diff --check` — passed.
- No live server, model load, or GPU benchmark was run.

## Fix round 3 — seeded nucleus cutoff ties

- Seeded `top_p` fallback now derives a probability threshold from the first sorted cumulative-mass crossover, then retains every original token whose probability is at least that threshold. This matches the Triton sampler's threshold (`x >= thr`) behavior and preserves ties at the cutoff.
- Added a direct regression for probabilities `[0.4, 0.3, 0.3]` at `top_p=0.5`; all three probabilities remain eligible after normalization. Existing CPU sampler determinism and standard-prefill trace tests are retained.

### Fix-round verification

- Red phase: `PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/engine/test_sample.py::test_seeded_top_p_retains_all_tokens_tied_at_cutoff -q` — failed as expected before the threshold fix: observed `[0.5714, 0.4286, 0]`.
- `PYTHONPATH=/home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/engine/test_sample.py -q` — `3 passed`.
- Focused Task 5 regression suite (sampler, prefill trace, runner sanitization, seed adapter, terminal stats/wire) — `28 passed` (one existing FastAPI/httpx deprecation warning); `py_compile` and `git diff --check` also passed.
- No live server, model load, or GPU benchmark was run.
