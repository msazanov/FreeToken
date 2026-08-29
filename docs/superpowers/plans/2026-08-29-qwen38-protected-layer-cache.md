# Qwen3.8 Protected Layer Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Qwen3.8's decode-only global-LRU thrash with an opt-in protected per-layer MoE cache and measure it against the complete fixed-seed 64K control.

**Architecture:** Keep the 256 existing GPU expert slots and every model calculation unchanged. During Qwen file-backed decode only, reserve contiguous per-layer slot ranges and a shared transient tail: at 256 slots, 48 layers receive five protected slots each and the remaining sixteen slots serve rank-6-to-10 cold experts. Prefill remains the existing dynamic file-backed path; router IDs remain authoritative and are only rewritten to physical cache slots after admission.

**Tech Stack:** Python 3.12, PyTorch, Triton 3.6, FreeToken GGUF Q4_K_M offload cache, RTX 2070 Mobile SM75.

**Spec:** `docs/superpowers/specs/2026-08-28-qwen38-turing-runtime-optimization-design.md`, E11. Control: `benchmarks/results/qwen38-e1e4-64k-ws16-top10-router/context-65536.json` at `ee93760`.

## Global Constraints

- Target Qwen3.8 Flash Next Q4_K_M file-backed GGUF on RTX 2070 Mobile / 8 GiB VRAM, i7-8750H / 32 GiB RAM / NVMe.
- Retain FreeToken, page size 4, `tq4-nc`, 16 MiB QSA workspace, top-10 router, greedy seed `20260828`, one running request and no MTP/speculation.
- Do not change router scores, selected top-k IDs, weights, quantization, QSA/GDN/PLE state, or KV representation.
- New policy is decode-only. Prefill retains current dynamic file-backed admission and copy choreography.
- Support it only for `quant_format == "qwen4_gguf"`, GPU decode, and one request. Reject unsupported configurations early.
- Geometry: `P = max(1, (cache_size - 16) // num_layers)`; `T = cache_size - P * num_layers`; require `T >= minimum_cache_size`. At 256/48/10 this is `P=5`, `T=16`.
- Keep `lru` default/rollback. Do not add a RAM L2 in this change.
- No performance claim until 1K and 64K matched artifacts, exact output digests, and updates to `TESTLOG.md`, `CHANGELOG.md`, and `README.md` exist.

---

### Task 1: Policy geometry, validation and telemetry

**Files:**
- Modify: `python/freetoken/engine/config.py`, `python/freetoken/server/args.py`, `python/freetoken/engine/engine.py`, `python/freetoken/moe/offload_cache.py`
- Modify: `tests/moe/test_telemetry.py`, `tests/moe/test_offload.py`

**Interfaces:**
- Produces `EngineConfig.moe_cache_policy` values `lru` or `protected_layer`.
- Produces `OffloadMoeCache.protected_slots_per_layer`, `transient_slots`, and `cache_policy_geometry() -> dict[str, int | str]`.
- Extends `telemetry_snapshot()["cache"]` with scalar policy geometry only.

- [ ] **Step 1: Write the failing tests.**

```python
def test_protected_layer_geometry_uses_all_256_slots():
    cache = OffloadMoeCache(
        num_layers=48, num_experts=512, cache_size=256,
        device=torch.device("cpu"), quant_format="qwen4_gguf",
        minimum_cache_size=10, cache_policy="protected_layer",
    )
    assert cache.protected_slots_per_layer == 5
    assert cache.transient_slots == 16
    assert cache.cache_policy_geometry()["protected_slot_count"] == 240

def test_protected_layer_rejects_non_qwen_layout():
    with pytest.raises(ValueError, match="qwen4_gguf"):
        OffloadMoeCache(1, 4, 16, torch.device("cpu"), cache_policy="protected_layer")
```

Assert parser acceptance and terminal telemetry geometry too.

- [ ] **Step 2: Verify the tests fail.**

Run: `PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_telemetry.py tests/moe/test_offload.py -q -k protected_layer`

Expected: fail because the policy and geometry do not exist.

- [ ] **Step 3: Implement the minimal contract.**

Add the parser choice. Validate layout, GPU decode, request limit, and transient capacity in `OffloadMoeCache`. Return scalar geometry from the terminal snapshot without retaining IDs or token history.

- [ ] **Step 4: Verify focused tests pass.**

Run: `PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_telemetry.py tests/moe/test_offload.py -q -k 'protected_layer or telemetry'`

Expected: pass.

- [ ] **Step 5: Commit.**

Run: `git add python/freetoken/engine/config.py python/freetoken/server/args.py python/freetoken/engine/engine.py python/freetoken/moe/offload_cache.py tests/moe/test_telemetry.py tests/moe/test_offload.py && git commit -m "feat: add Qwen protected cache geometry"`

### Task 2: Decode-only protected/transient admission

**Files:**
- Modify: `python/freetoken/moe/offload_kernels.py`, `python/freetoken/moe/offload_cache.py`
- Modify: `tests/moe/test_offload.py`, `tests/models/test_qwen4exp_gguf_experts.py`

**Interfaces:**
- Produces `ensure_experts_protected_layer(cache, layer_id: int, expert_ids: torch.Tensor) -> None`.
- Protected range for layer `L` is `[L * P, (L + 1) * P)`; transient range is `[num_layers * P, cache_size)`.
- Existing `num_indices`, `evict_slots`, `src_indices`, `slot_for_id`, `id_of_slot`, and `usage` keep their copy contract.

- [ ] **Step 1: Write failing CPU/CUDA parity tests.**

```python
def test_protected_layer_keeps_top_ranked_experts_across_other_layers():
    cache = make_qwen_protected_cpu_cache(layers=2, experts=8, slots=16, topk=2)
    first = torch.tensor([[7, 3]], dtype=torch.int32)
    cache.ensure_experts(0, first)
    cache.ensure_experts(1, torch.tensor([[1, 2]], dtype=torch.int32))
    repeated = torch.tensor([[7, 3]], dtype=torch.int32)
    cache.ensure_experts(0, repeated)
    assert cache.num_indices.item() == 0
    assert (repeated >= 0).all()
```

Assert rank-1-to-P misses use only their layer range; rank-tail misses use transient slots; a call in another layer cannot evict protected ownership; legacy LRU remains unchanged. Add CUDA-gated Qwen-sized 48x512 top-10 parity against the CPU path.

- [ ] **Step 2: Verify the tests fail.**

Run: `PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_offload.py tests/models/test_qwen4exp_gguf_experts.py -q -k protected_layer`

Expected: fail because Qwen still dispatches global dynamic hybrid-LRU.

- [ ] **Step 3: Implement deterministic admission.**

Dispatch it only when policy is `protected_layer`, bank is file-backed, and trace phase is `decode`; all prefill retains dynamic LRU. Build the unique route set in rank order. Admit first `P` routed experts into the layer's protected range, evicting the least-used non-active slot within that range. Admit the remaining active experts into transient slots, evicting the least-used non-active transient slot. Before an eviction clear the old mapping; write the new mapping, compact source index, destination slot, and usage exactly as the existing `copy_missing` protocol requires. CPU and Triton use the same `(usage, slot)` tie break.

- [ ] **Step 4: Verify CPU/CUDA parity.**

Run: `PYTHONPATH=python /home/random/dev/qwen/freetoken/python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/moe/test_offload.py tests/models/test_qwen4exp_gguf_experts.py -q -k 'protected_layer or qwen4_file_backed_lru_admission'`

Expected: pass; CUDA parity skips only with no CUDA.

- [ ] **Step 5: Commit.**

Run: `git add python/freetoken/moe/offload_kernels.py python/freetoken/moe/offload_cache.py tests/moe/test_offload.py tests/models/test_qwen4exp_gguf_experts.py && git commit -m "feat: protect Qwen decode experts by layer"`

### Task 3: Exact-output runner and matched A/B

**Files:**
- Modify: `benchmarks/qwen38_turing_profile.py`, `tests/benchmarks/test_qwen38_turing_profile.py`
- Create: `benchmarks/results/qwen38-protected-layer-1k/*`, `benchmarks/results/qwen38-protected-layer-64k/*`
- Modify: `TESTLOG.md`, `CHANGELOG.md`, `README.md`

**Interfaces:**
- Produces `response_sha256` from final deterministic response bytes and retains `server_stats.moe.cache.policy`.
- Consumes matching model, prompt fixture, `temperature=0`, seed `20260828`, `tq4-nc`, page size 4, QSA workspace 16 MiB, and 255 decode tokens.

- [ ] **Step 1: Write a failing runner test.**

```python
def test_result_records_response_digest_and_cache_policy():
    record = make_result_record(response_bytes=b"fixed output", stats={
        "moe": {"cache": {"policy": "protected_layer"}}
    })
    assert record["response_sha256"] == hashlib.sha256(b"fixed output").hexdigest()
    assert record["server_stats"]["moe"]["cache"]["policy"] == "protected_layer"
```

- [ ] **Step 2: Verify it fails, then implement it.**

Run: `PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q`

Expected before implementation: fail because no response digest exists. Hash only final response bytes; never write the full prompt or use `/tmp`.

- [ ] **Step 3: Verify runner tests pass.**

Run: `PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q`

Expected: pass.

- [ ] **Step 4: Run 1K control/candidate and compare digests.**

Run the existing 1K command twice, changing only `--moe-cache-policy lru` to `protected_layer`. Require equal response digests before comparing speed, hit/miss, evictions, H2D bytes, I/O/faults, GPU utilization and VRAM.

- [ ] **Step 5: Run 64K control/candidate and document.**

Use the complete 64K control configuration, 1,024-token prefill chunks and 255 decode tokens. Change only the cache policy for candidate. Append both commands, artifacts, output-digest comparison, acceptance decision and every failure to `TESTLOG.md`; add user-facing results to `CHANGELOG.md` and a concise Qwen benchmark table to `README.md`.

- [ ] **Step 6: Commit and publish.**

Run: `git add benchmarks/qwen38_turing_profile.py tests/benchmarks/test_qwen38_turing_profile.py benchmarks/results TESTLOG.md CHANGELOG.md README.md && git commit -m "bench: compare Qwen protected layer cache"`

Then run: `git fetch upstream --prune` and `git push origin feat/qwen4exp-gguf-turing`.

## Plan Self-Review

- E11 geometry, request guard, rollback, and 256-slot target are isolated in Tasks 1-2.
- CPU/CUDA deterministic admission and existing three-bank copy invariants are tested in Task 2.
- Matched 1K/64K, output equivalence, expert/H2D/I/O/GPU measurement, three documentation files, commits and push are covered in Task 3.
- RAM-L2, prediction, page-size changes, QSA changes, and hybrid-radix are explicitly excluded, leaving one changed variable in the A/B.
