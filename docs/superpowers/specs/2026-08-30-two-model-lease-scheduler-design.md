# Two-Model Lease Scheduler Design

**Date:** 2026-08-30
**Status:** Proposed for user review
**Tracked by:** `ornith-q4km-turing-59w`

## Purpose

Expose Ornith 1.5 35B and Gemma 4 E2B through one stable OpenAI-compatible
endpoint on the RTX 2070 host while allowing only one model to execute
inference at a time. The scheduler must preserve the active model's locality,
queue work deterministically, avoid mid-request migration, retain Ornith's 64K
prefix/KV cache while parked, and use CPU Gemma only as a fallback after Gemma
owns the execution lease.

## User-visible contract

- `GET /v1/models` always advertises both stable IDs:
  - `ornith-35b`
  - `gemma-4-e2b`
- OpenAI-compatible chat requests select a model with the request's `model`
  field. Unknown IDs fail with a deterministic 4xx response and do not start or
  stop any backend.
- At most one model executes inference at any instant. An idle process or
  mmap-resident weight file is not considered active execution.
- Requests for each model form an independent FIFO queue.
- The current lease owner has affinity priority. It completes its active
  request and drains its already admitted queue before the other model may take
  the lease. Requests are never migrated between runtimes mid-token.
- When no model owns the lease and both queues are non-empty, Ornith wins the
  tie. This tie-break does not preempt an active Gemma lease.
- New requests for the current lease owner remain ahead of the waiting model.
  Continuous traffic can therefore delay the other model indefinitely; this is
  intentional because the requested policy makes the active model strictly
  higher priority than a model switch.
- A queued client may cancel its request. Cancellation removes work that has
  not started and propagates upstream for work already running.

## Runtime topology

```text
LAN / local clients
        |
        v
single-model arbiter :1919
        |
        +-- Ornith FreeToken :19191 (private, normally resident)
        |
        +-- Gemma FreeToken  :19192 (private, GPU primary, on demand)
        |
        +-- Gemma llama.cpp  :19193 (private, CPU fallback, RAM resident)
        |
        +-- FreeToken daemon/control :1900 (private lifecycle API)
```

The arbiter is a torch-free process. It owns public admission, model identity,
queues, streaming proxying, readiness commits, metrics, and the execution
lease. It does not import model or CUDA code.

Ornith remains a separately supervised FreeToken process so that its loaded
weights and radix prefix cache survive Gemma activity. The existing FreeToken
daemon supervises the on-demand Gemma GPU child. The CPU llama.cpp process may
remain mmap-resident, but it receives inference only while Gemma owns the lease
and its GPU backend is unavailable.

Internal inference and administration ports bind to loopback. Only the arbiter
binds to `0.0.0.0:1919`. Administrative cache-rebuild and process-control
endpoints are not exposed to the LAN.

## Why this topology

FreeToken currently constructs one model, tokenizer, scheduler, CUDA graph set,
and model-specific cache geometry per engine. A single engine cannot contain
both models, and FreeToken has no dense CPU execution backend for Gemma E2B.
GPU and CPU runtimes also have incompatible KV layouts, graph state, and random
number generator state, so a partially generated request cannot move from one
runtime to another.

The existing FreeToken daemon already provides serialized child lifecycle,
drain/abort accounting, health proxying, and process re-adoption. The new
arbiter composes those primitives instead of duplicating model-process
management. `llama-swap` was considered because it provides OpenAI proxying,
TTL unloading, readiness checks, and a swap matrix. Its generic model-based
replacement policy does not express this design's combination of a permanently
parked Ornith process, MoE-only cache resizing, strict active-model affinity,
and Gemma GPU-to-CPU fallback. It remains a reference implementation rather
than an additional production dependency.

## Model and memory profiles

### Ornith active

- Checkpoint: current Ornith 1.5 35B TQ3_4S GGUF.
- Context and KV budget: exactly 65,536 tokens.
- KV dtype: INT8.
- MoE expert cache: the measured active profile, currently 2,311 slots.
- Expected measured footprint: about 6.9 GiB, with the current smoke peaking at
  7,282 MiB.

### Ornith parked

- Keep the process and 65,536-token KV allocation alive.
- Resize only the MoE expert cache to the validated minimum, initially 256
  slots.
- Do not resize KV, Mamba, or SWA pools. The scheduler's MoE-only rebuild path
  preserves the radix prefix cache; resizing the other pools clears it.
- Expected footprint is approximately 3.5 GiB and must be measured rather than
  treated as an acceptance value.

### Gemma GPU active

- Checkpoint: `google/gemma-4-E2B-it-qat-q4_0-gguf`, local Q4_0 GGUF.
- Runtime: FreeToken, after a targeted port of the Gemma E-series behavior from
  upstream PR #59.
- Context gate: 4,096 tokens first, then at most 8,192 for the voice workload.
- Reasoning disabled and one running request.
- The first implementation uses graph-safe GPU PLE. If parked Ornith plus
  Gemma fails the co-residency gate, add a graph-safe host-resident Q6_K PLE
  path by adapting the existing Qwen host-PLE machinery. A llama.cpp GPU route
  is rollback-only and does not prove the FreeToken Gemma objective.

### Gemma CPU fallback

- Runtime: the existing llama.cpp build with zero GPU layers.
- It is eligible only after Gemma owns the lease and Gemma GPU failed startup,
  readiness, or the VRAM admission gate.
- It never serves concurrently with active Ornith.
- Start with four CPU threads and low CPU/IO scheduling weight. Concurrent
  resource contention is still measured because parked Ornith retains host
  expert banks and RAM bandwidth matters on this machine.

## Gemma 4 E2B compatibility work

The current FreeToken Gemma parser assumes the large MoE Gemma 4 architecture
and rejects dense E2B metadata. Upstream PR #59 adds the required E-series
features: per-layer embeddings, shared KV layers, double-wide MLP layers,
scalar KV-head handling, dense GGUF parsing, and q-only shared-attention
weights.

The PR is open, conflicts with current main, and contains unrelated Llama GGUF,
FP8 KV, and launcher commits. The implementation ports only the E-series
behavior from commits `6486907` and `abd3b14`, reconciled with current upstream
and this fork. It does not cherry-pick the entire PR.

Required tests cover dense-vs-MoE configuration detection, exact E2B metadata,
per-layer FFN widths, PLE tensor mapping and indexing, shared-KV layer reuse,
scalar KV heads, graph capability selection, and a live SM75 generation gate.

## Lease state machine

The arbiter persists no prompts or queue bodies. Its in-memory state is one of:

- `IDLE`: no owner and no switch in progress.
- `ORNITH_PREPARING`: Ornith cache expansion and readiness verification.
- `ORNITH_ACTIVE`: only Ornith requests may be admitted upstream.
- `ORNITH_DRAINING`: no new Ornith request starts while terminal replies finish.
- `GEMMA_PREPARING`: Ornith is parked and the Gemma GPU or CPU backend is being
  selected and checked.
- `GEMMA_ACTIVE_GPU`: only Gemma requests execute on FreeToken GPU.
- `GEMMA_ACTIVE_CPU`: only Gemma requests execute on llama.cpp CPU.
- `GEMMA_DRAINING`: no new Gemma request starts while terminal replies finish.
- `FAILED`: no backend is safe to receive work; queued requests fail explicitly.

### IDLE to Ornith

1. Atomically grant the lease to Ornith.
2. Stop an unowned Gemma GPU child if one survived a previous crash recovery.
3. Request a MoE-only expansion to the active expert-slot count with
   `mode=if_idle`.
4. Verify cache geometry, `/health`, and `/v1/models`.
5. Commit `ORNITH_ACTIVE` and forward the head of the Ornith FIFO.

### IDLE or drained Ornith to Gemma

1. Atomically grant the lease to Gemma and stop Ornith admission.
2. Verify Ornith has no active or terminally pending request.
3. Request a MoE-only shrink to the parked slot count and verify that KV page
   count remains 65,536.
4. Start Gemma FreeToken through the control daemon and wait for health plus the
   expected model identity.
5. If GPU readiness fails, stop the partial GPU child and health-check CPU
   llama.cpp.
6. Commit `GEMMA_ACTIVE_GPU` or `GEMMA_ACTIVE_CPU`, then forward Gemma FIFO.

### Gemma to waiting Ornith

1. Keep admitting Gemma because the active model has strict affinity priority.
2. Once the Gemma active request and Gemma FIFO are both empty, enter
   `GEMMA_DRAINING` and close Gemma admission.
3. Stop Gemma GPU gracefully; the CPU fallback remains idle and receives no
   request.
4. Expand Ornith's MoE cache, verify readiness, and commit the Ornith lease.
5. Forward the head of the Ornith FIFO.

The symmetric rule applies when Gemma waits behind active Ornith. There is no
fixed idle TTL: a waiting other-model queue triggers switching immediately
after the active queue drains. With no waiting work, the current owner remains
warm until the next request identifies whether a switch is necessary.

## Startup and recovery

Startup order is significant because FreeToken records a baseline free-VRAM
budget before allocating model caches:

1. Start Ornith alone on the otherwise free GPU.
2. Verify the full 64K active profile.
3. Park Ornith with a MoE-only shrink while preserving 64K KV.
4. Start the CPU Gemma fallback and the FreeToken control daemon.
5. Start the arbiter and expose the public endpoint.
6. Do not preload Gemma GPU until its first lease unless the co-residency gate
   proves that preloading adds no material memory risk.

After arbiter restart, it reconciles actual child PIDs, health, model identity,
Ornith cache geometry, and GPU occupancy before accepting traffic. It never
infers ownership solely from stale state on disk.

## Failure and streaming behavior

- A backend is not routable until both health and model identity checks pass.
- A switch timeout fails the waiting queue with 503 and leaves the last known
  healthy owner unchanged when possible.
- If Gemma GPU fails before response bytes are committed, the request may be
  retried once on CPU under the same Gemma lease.
- If any streaming bytes were committed, the arbiter never replays the request
  automatically because that could duplicate spoken or visible output.
- Ornith failure is reported as Ornith failure; Gemma is not substituted for a
  request that selected Ornith.
- Client disconnect propagates cancellation upstream. Scheduler ownership is
  released only after terminal acknowledgement, not merely after socket close.
- Queue length and per-request wait time are bounded by configuration. Overflow
  receives 429; queue timeout receives 504. Request bodies are not persisted.
- A failed destructive cache rebuild invokes existing rollback behavior. The
  arbiter does not route to Ornith until the old or requested geometry is proven
  healthy.

## Metrics and observability

The arbiter exposes aggregate health and Prometheus-compatible metrics for:

- current lease owner and state;
- active backend (`ornith-gpu`, `gemma-gpu`, or `gemma-cpu`);
- per-model queued and active request counts;
- queue wait, TTFT, generation time, and total latency;
- switch direction, duration, result, and failure reason;
- Ornith active/parked MoE slots and invariant KV pages;
- Gemma GPU startup/readiness time and CPU-fallback count;
- cancellations, queue overflow, timeout, and partial-stream failures.

Structured logs carry a request ID, selected public model, lease epoch, queue
position, backend, and state transition. They never log prompt content by
default.

## Verification strategy

### Deterministic tests

- FIFO ordering within each model.
- Active-model affinity when the other queue becomes non-empty.
- Ornith tie-break when both queues arrive in `IDLE`.
- Exactly one execution backend active across concurrent request races.
- No model switch before terminal acknowledgement of the previous request.
- Queued cancellation, active streaming cancellation, overflow, and timeout.
- Gemma GPU readiness failure selecting CPU only under a Gemma lease.
- No automatic retry after streamed bytes are committed.
- Stable aggregate `/v1/models` independent of current owner.
- Crash reconciliation and stale-child cleanup.

### Live RTX 2070 gates

1. Boot FreeToken Gemma alone on SM75 and verify one deterministic Russian and
   one English response with reasoning disabled.
2. Record cold/warm load, TTFT, decode tok/s, VRAM, RAM, swap, and temperature.
3. Boot Ornith first, park it with 64K KV unchanged, and prove a prefix-cache hit
   survives MoE shrink and re-expansion.
4. Start Gemma GPU beside parked Ornith and prove no OOM. If it fails, implement
   host PLE and repeat before accepting a runtime fallback.
5. Exercise Gemma -> queued Ornith and Ornith -> queued Gemma with streaming,
   cancellations, and several same-model requests.
6. Prove only one backend generates at a time from arbiter metrics and backend
   logs.
7. Reboot the user-service stack and repeat model listing plus one request to
   each model through the public endpoint.

All commands, configurations, positive and negative results, and raw artifacts
are appended to `TESTLOG.md` and `CHANGELOG.md`; user-facing operating status is
updated in `README.md`. Raw evidence lives under `benchmarks/results/`, never
under `/tmp`.

## Rollout and rollback

The existing Gemma llama.cpp service remains the rollback path until all live
gates pass. New services use private ports and do not replace public `:1919`
until the arbiter passes isolated integration tests. Deployment then proceeds
as one reversible unit: install private services, verify them, stop the old
public listener, start the arbiter on `:1919`, and run API smoke tests. On any
failure, stop the arbiter and restore the prior single-model service without
changing model files or deleting benchmark evidence.
