# Two-Model Lease Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `ornith-35b` and `gemma-4-e2b` through one OpenAI-compatible `:1919/v1` endpoint while executing only one model at a time with FIFO queues and active-model affinity priority.

**Architecture:** A torch-free FastAPI arbiter owns admission, two FIFO queues, one execution lease, and an HTTPX streaming proxy. Ornith remains resident on a private FreeToken port and changes only its MoE slot cache; an existing FreeToken daemon starts/stops Gemma GPU, while CPU llama.cpp is selected only if Gemma owns the lease and GPU readiness fails.

**Tech Stack:** Python 3.10+, FastAPI, HTTPX async streaming, FreeToken daemon/control APIs, llama.cpp OpenAI API, pytest, systemd user services.

**Spec:** `docs/superpowers/specs/2026-08-30-two-model-lease-scheduler-design.md`

## Global Constraints

- Public endpoint is `0.0.0.0:1919`; private backends bind loopback.
- Public model IDs are exactly `ornith-35b` and `gemma-4-e2b`.
- Only one model executes inference at a time; no mid-token migration or replay after streamed bytes.
- Active-model requests have affinity priority; Ornith breaks an `IDLE` tie; each queue is FIFO.
- Ornith keeps exactly 65,536 KV pages while parked; only `moe_cache_size` changes.
- Gemma GPU primary is FreeToken E2B; llama.cpp CPU is fallback only under a Gemma lease.
- Tests run with `PYTHONPATH=$PWD/python:$PWD /home/random/freetoken-turing/.venv/bin/python -m pytest`.
- Every live positive or negative result is appended to `TESTLOG.md` and `CHANGELOG.md`, with raw artifacts under `benchmarks/results/`.

---

### Task 1: Port Gemma 4 E-series support

**Files:**
- Create: `tests/models/test_gemma4_e_series.py`
- Modify: `python/freetoken/engine/engine.py`
- Modify: `python/freetoken/models/config.py`
- Modify: `python/freetoken/models/gemma4/attention.py`
- Modify: `python/freetoken/models/gemma4/config.py`
- Modify: `python/freetoken/models/gemma4/gguf.py`
- Modify: `python/freetoken/models/gemma4/model.py`
- Modify: `python/freetoken/models/gemma4/moe.py`
- Modify: `python/freetoken/models/gemma4/weight.py`

**Interfaces:**
- Consumes: upstream behavior from commits `6486907` and `abd3b14`, not their unrelated descendants.
- Produces: dense E2B `ModelConfig`, PLE injection, shared-KV attention, double-wide MLP selection, and graph-safe GGUF PLE.

- [ ] **Step 1: Write failing dense E2B metadata tests**

```python
def test_dense_e2b_metadata_does_not_require_expert_keys(monkeypatch):
    config = parse_gguf_config(_e2b_shim(monkeypatch))
    assert config.moe_enabled is False
    assert config.per_layer_hidden_size == 256
    assert config.num_kv_shared_layers == 20
    assert config.intermediate_size_by_layer[:15] == (6144,) * 15
    assert config.intermediate_size_by_layer[15:] == (12288,) * 20
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD/python:$PWD /home/random/freetoken-turing/.venv/bin/python -m pytest -q tests/models/test_gemma4_e_series.py`

Expected: fail because E-series fields/dense parsing do not exist.

- [ ] **Step 3: Port only the two relevant upstream deltas**

Apply the behavior of `6486907`, then `abd3b14`; reconcile names with current `ModelConfig` and preserve current Turing GGUF/MoE paths. Do not import commits `efecc23`, `392f613`, or `36da469`.

- [ ] **Step 4: Add PLE/shared-KV behavioral tests and verify GREEN**

```python
def test_shared_kv_layers_build_q_only_attention():
    layer = Gemma4Attention(_e2b_config(), layer_id=34)
    assert layer.k_proj is None
    assert layer.v_proj is None

def test_gguf_ple_defaults_to_graph_safe_gpu_embedding():
    model = Gemma4ForCausalLM(_e2b_gguf_config())
    assert model.supports_cuda_graph is True
```

Run the new file plus existing Gemma/dispatch tests and commit the green cycle.

---

### Task 2: Implement the lease and FIFO scheduler

**Files:**
- Create: `python/freetoken/arbiter/__init__.py`
- Create: `python/freetoken/arbiter/model.py`
- Create: `python/freetoken/arbiter/scheduler.py`
- Create: `tests/arbiter/test_scheduler.py`

**Interfaces:**
- Produces: `ModelId`, `LeaseState`, `QueuedRequest`, and `LeaseScheduler`.
- `await LeaseScheduler.acquire(model_id, request_id)` returns an async lease context.
- `LeaseScheduler.snapshot()` returns owner, state, active count, and per-model FIFO depths.

- [ ] **Step 1: Write RED tests for FIFO, affinity, tie-break, and cancellation**

```python
async def test_active_model_drains_before_other_model():
    scheduler = LeaseScheduler(tie_break=ModelId.ORNITH)
    first = await scheduler.acquire(ModelId.GEMMA, "g1")
    waiting_o = asyncio.create_task(scheduler.acquire(ModelId.ORNITH, "o1"))
    waiting_g = asyncio.create_task(scheduler.acquire(ModelId.GEMMA, "g2"))
    await first.release()
    assert (await waiting_g).model_id is ModelId.GEMMA
    assert not waiting_o.done()
```

- [ ] **Step 2: Verify RED**

Run the focused scheduler test and confirm import failure for the missing package.

- [ ] **Step 3: Implement the minimal condition-based state machine**

Use one `asyncio.Condition`, one `deque` per model, monotonic enqueue sequence, an atomic switch fence when the active queue becomes empty, and terminal release acknowledgement before ownership changes.

- [ ] **Step 4: Verify GREEN and commit**

Run all arbiter scheduler tests including simultaneous arrivals, queued cancellation, queue limit, timeout, and snapshot invariants.

---

### Task 3: Add lifecycle orchestration and OpenAI streaming proxy

**Files:**
- Create: `python/freetoken/arbiter/backends.py`
- Create: `python/freetoken/arbiter/proxy.py`
- Create: `python/freetoken/arbiter/app.py`
- Create: `python/freetoken/arbiter/server.py`
- Create: `tests/arbiter/test_backends.py`
- Create: `tests/arbiter/test_app.py`
- Modify: `pyproject.toml`
- Modify: `python/freetoken/cli.py`

**Interfaces:**
- `BackendController.prepare(model_id) -> ActiveBackend` performs cache/lifecycle/readiness work before lease commit.
- `BackendController.release(model_id)` parks/stops the previous GPU backend.
- `proxy_openai(request, backend, client)` returns a streaming or buffered FastAPI response and always closes the HTTPX response.
- `build_arbiter_app(config, controller, client)` exposes `/health`, `/metrics`, `/v1/models`, `/v1/chat/completions`, and `/v1/responses`.

- [ ] **Step 1: Add `httpx>=0.27,<1` and write RED API tests**

```python
def test_models_always_lists_both_public_ids(client):
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [
        "ornith-35b", "gemma-4-e2b"
    ]

def test_unknown_model_is_404_without_backend_transition(client, controller):
    response = client.post("/v1/chat/completions", json={"model": "unknown", "messages": []})
    assert response.status_code == 404
    assert controller.calls == []
```

- [ ] **Step 2: Verify RED**

Run `tests/arbiter/test_backends.py` and `tests/arbiter/test_app.py` and confirm missing arbiter modules/routes.

- [ ] **Step 3: Implement controller and streaming proxy**

Use HTTPX manual streaming (`client.send(..., stream=True)`), `aiter_raw()`, and a `BackgroundTask(response.aclose)`. Strip hop-by-hop headers, rewrite the upstream `model` field to the backend's expected ID, and release the lease only after stream close/terminal response.

- [ ] **Step 4: Implement switch sequencing**

For Gemma: MoE-only park Ornith, start/health-check FreeToken Gemma through daemon, then CPU fallback on pre-commit GPU failure. For Ornith: stop Gemma GPU, expand Ornith MoE slots, verify unchanged KV pages and model identity, then commit readiness.

- [ ] **Step 5: Verify GREEN and commit**

Cover stream cancellation, no replay after bytes, pre-commit CPU fallback, switch failure rollback, one-active-backend invariant, and aggregate metrics.

---

### Task 4: Add source-controlled service topology

**Files:**
- Create: `deploy/systemd/freetoken-arbiter.service`
- Create: `deploy/systemd/freetoken-gemma-daemon.service`
- Create: `deploy/systemd/llama-gemma-cpu.service`
- Modify: `deploy/systemd/freetoken-ornith.service`
- Create: `tests/deploy/test_two_model_systemd.py`

**Interfaces:**
- Public arbiter listens on `0.0.0.0:1919`.
- Ornith listens on `127.0.0.1:19191`.
- Gemma GPU daemon listens on `127.0.0.1:1900` and starts serve on `127.0.0.1:19192`.
- Gemma CPU listens on `127.0.0.1:19193` with zero GPU layers.

- [ ] **Step 1: Write RED unit-contract tests**

Assert unique ports, loopback-only private services, startup ordering, no `Conflicts=` between public model choices, exact model paths, 64K Ornith geometry, and CPU Gemma `--gpu-layers 0`.

- [ ] **Step 2: Verify RED**

Run the new deployment test and confirm missing units/private-port mismatch.

- [ ] **Step 3: Add units and arbiter CLI arguments**

Keep existing live units untouched until all isolated tests pass. Validate source units with `systemd-analyze --user verify`.

- [ ] **Step 4: Verify GREEN and commit**

Run deployment tests, shell syntax checks, and systemd verification.

---

### Task 5: Live rollout, measurements, and handoff

**Files:**
- Modify: `README.md`
- Modify: `TESTLOG.md`
- Modify: `CHANGELOG.md`
- Create: `benchmarks/results/two-model-arbiter-2026-08-31/` artifacts

**Interfaces:**
- Produces a boot-enabled, rollback-safe one-endpoint service and raw evidence proving both model IDs, ordering, switching, GPU/CPU selection, and one-active-model execution.

- [ ] **Step 1: Gate FreeToken Gemma alone**

Measure cold/warm readiness, Russian/English output, no-thinking behavior, TTFT, decode tok/s, VRAM/RAM/swap, and SM75 errors on private `:19192`.

- [ ] **Step 2: Gate Ornith park/expand without losing 64K prefix cache**

Record cache geometry and hit counters before park, after MoE-only shrink, and after expansion.

- [ ] **Step 3: Gate co-residency and implement host PLE if required**

Start Gemma GPU beside parked Ornith. If OOM occurs, return to Task 1 with a new RED host-Q6_K-PLE test; do not silently accept llama.cpp GPU as success.

- [ ] **Step 4: Install services atomically and exercise queues**

Run Gemma→Ornith and Ornith→Gemma requests, same-model FIFO bursts, simultaneous idle arrivals, cancellations, and CPU fallback injection. Capture arbiter/backend metrics proving no overlapping generation.

- [ ] **Step 5: Update permanent records, verify, commit, and push**

Run focused/full relevant tests, `git diff --check`, live API smokes, and systemd status checks. Append all results, commit only scoped changes, pull/rebase safely, push the branch, and verify origin is current.
