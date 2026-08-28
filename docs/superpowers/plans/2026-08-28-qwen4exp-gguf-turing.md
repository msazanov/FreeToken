# Qwen3.8 Flash Next GGUF on Turing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, text-only FreeToken path for AtomicChat Qwen3.8 Flash Next
`AD-4.27bpw-Q4_K_M-M64` on the RTX 2070 runtime under `/home/random/dev/qwen`.

**Architecture:** Port only the text execution primitives from the Qwen4 source
branch into the current Ornith GGUF branch. Register `qwen4exp`, parse its GGUF
metadata, and implement a packed-weight/PLE boundary without merging the older
source branch. Keep cache reuse conservative until PLE/QSA state can be restored.

**Tech Stack:** Python 3, PyTorch, Triton, FreeToken GGUF reader and CUDA GGML
kernels, pytest, Hugging Face model files.

**Spec:** `docs/superpowers/specs/2026-08-28-qwen4exp-gguf-turing-design.md`

## Global Constraints

- Do all runtime and model work below `/home/random/dev/qwen`.
- Do not stop, reconfigure, or import code from the active Ornith service.
- Text-only first: vision/mmproj and MTP are out of scope.
- Do not use NVFP4 kernels; target Turing SM 7.5 and native packed GGUF kernels.
- PLE stays mmap-backed on NVMe; never materialize the whole table in RAM/VRAM.
- Qwen top-10 router must use the numerically equivalent Torch fallback.
- No radix-cache claim until PLE and QSA state are restored and covered by tests.
- Follow RED → GREEN → REFACTOR and commit each green task separately.

---

### Task 1: Isolated runtime and Qwen4 text-core source boundary

**Files:**
- Create: `docs/models/qwen3.8-flash-next-q4km-turing.md`
- Modify: `python/freetoken/models/register.py`
- Test: `tests/models/test_qwen4_exp.py`

**Interfaces:**
- Consumes: source commit `ad752c9` from `qwen4-source/feature/qwen3.8-flash-next`.
- Produces: `Qwen4ExpForCausalLM` registration for a text-only model package.

- [ ] **Step 1: Write the failing registry test.**

```python
def test_qwen4_text_model_registry_entry():
    spec = get_model_spec("Qwen4ExpForCausalLM")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"
```

- [ ] **Step 2: Verify RED.**

Run: `pytest tests/models/test_qwen4_exp.py::test_qwen4_text_model_registry_entry -q`

Expected: FAIL because the Qwen4 package and registry entry do not exist.

- [ ] **Step 3: Port the minimal text-only Qwen4 model package.**

Copy only `args.py`, `config.py`, `model.py`, `mrope.py`, QSA attention/cache
pieces and their direct dependencies from `ad752c9`; omit `vision.py`, vision
imports and MTP loading. Register the model with `Qwen4ExpForCausalLM`.

- [ ] **Step 4: Verify GREEN.**

Run: `pytest tests/models/test_qwen4_exp.py -q`

Expected: PASS for registry, QSA geometry, MRoPE, router and PLE-token-history
unit tests that do not need a checkpoint.

- [ ] **Step 5: Commit.**

```bash
git add python/freetoken/models/qwen4_exp python/freetoken/attention python/freetoken/kvcache python/freetoken/models/register.py tests/models/test_qwen4_exp.py docs/models/qwen3.8-flash-next-q4km-turing.md
git commit -m "feat: add Qwen4 text runtime primitives"
```

### Task 2: Qwen4Exp GGUF configuration dispatch

**Files:**
- Modify: `python/freetoken/models/gguf/config.py`
- Create: `python/freetoken/models/qwen4_exp/gguf.py`
- Modify: `python/freetoken/models/qwen4_exp/__init__.py`
- Modify: `python/freetoken/models/register.py`
- Test: `tests/models/test_qwen4exp_gguf.py`

**Interfaces:**
- Consumes: `GgufConfigShim`, Qwen4 `ModelConfig`, metadata keys prefixed with
  `qwen4exp.`.
- Produces: `parse_gguf_config(shim: GgufConfigShim) -> ModelConfig` and the
  `Qwen4ExpGGUFForCausalLM` registry specification.

- [ ] **Step 1: Write a failing metadata-fixture test.**

```python
def test_qwen4exp_gguf_dispatches_and_preserves_qsa_and_ple_geometry(monkeypatch):
    shim = make_qwen4exp_shim()
    config = parse_gguf_config(shim)
    assert config.model_type == "qwen4exp"
    assert config.num_experts_per_tok == 10
    assert config.qwen4_args.ngram_size == 3
    assert config.requires_naive_cache
```

- [ ] **Step 2: Verify RED.**

Run: `pytest tests/models/test_qwen4exp_gguf.py::test_qwen4exp_gguf_dispatches_and_preserves_qsa_and_ple_geometry -q`

Expected: FAIL because `qwen4exp` is absent from `GGUF_ARCH_TO_REGISTRY`.

- [ ] **Step 3: Implement metadata parsing.**

Map every required AtomicChat header key: block count, hybrid layer interval,
QSA/indexer geometry, GDN geometry, MoE counts, hyper-connection parameters,
MRoPE sections and PLE multipliers/offsets/vocab sizes. Force text-only vision
configuration to `None`, route top-10 via Torch fallback and set conservative
cache/graph capability flags.

- [ ] **Step 4: Verify GREEN.**

Run: `pytest tests/models/test_qwen4exp_gguf.py tests/models/test_gguf_dispatch.py -q`

Expected: PASS; missing required metadata raises a message containing its
`qwen4exp.*` key.

- [ ] **Step 5: Commit.**

```bash
git add python/freetoken/models/gguf/config.py python/freetoken/models/qwen4_exp python/freetoken/models/register.py tests/models/test_qwen4exp_gguf.py
git commit -m "feat: dispatch Qwen4Exp GGUF metadata"
```

### Task 3: Packed Q4_K_M weights and isolated PLE loader

**Files:**
- Create: `python/freetoken/models/qwen4_exp/gguf_weights.py`
- Modify: `python/freetoken/models/qwen4_exp/gguf.py`
- Modify: `python/freetoken/models/qwen4_exp/model.py`
- Test: `tests/models/test_qwen4exp_gguf.py`

**Interfaces:**
- Consumes: `iter_gguf_tensors(model_path)`, `GgufTensor.packed()`, existing
  `GGUFLinear`/GGUF expert-bank interfaces.
- Produces: `iter_gguf_weights(model_path, device, *, include_moe_experts,
  include_non_moe)` and `Qwen4GgufPleStore` that holds mapped shard views.

- [ ] **Step 1: Write failing packed-weight and PLE isolation tests.**

```python
def test_qwen4_gguf_weight_loader_keeps_ple_tensor_mapped():
    entries = list(iter_gguf_weights(fixture_path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    assert "model.layers.1.ple.ple_embedding.ngram_embedding" not in dict(entries)
    assert Qwen4GgufPleStore.from_gguf(fixture_path).is_mmap_backed
```

- [ ] **Step 2: Verify RED.**

Run: `pytest tests/models/test_qwen4exp_gguf.py::test_qwen4_gguf_weight_loader_keeps_ple_tensor_mapped -q`

Expected: FAIL because no Qwen4 GGUF weight loader exists.

- [ ] **Step 3: Implement native GGUF name mapping and offload banks.**

Map Qwen4 GGUF tensors to the text model's fused projections. Route routed
expert tensors through the existing heterogeneous packed expert-bank code;
do not dequantize entire expert layers. Build a PLE store over shard-2 packed
views, gathering only rows required for the active n-grams.

- [ ] **Step 4: Verify GREEN.**

Run: `pytest tests/models/test_qwen4exp_gguf.py tests/models/test_gguf_type_tables.py -q`

Expected: PASS; unsupported tensor quant types identify both tensor and GGML
type; PLE test proves no whole-table tensor allocation.

- [ ] **Step 5: Commit.**

```bash
git add python/freetoken/models/qwen4_exp tests/models/test_qwen4exp_gguf.py
git commit -m "feat: load Qwen4 GGUF weights and PLE"
```

### Task 4: Cache-state contract, guarded download and runtime benchmark

**Files:**
- Create: `tests/kvcache/test_qwen4exp_state_contract.py`
- Create: `benchmarks/run_qwen4exp_gguf_smoke.py`
- Create: `benchmarks/2026-08-28-qwen4exp-q4km-turing.md`
- Create: `scripts/download-qwen4exp-q4km.sh`
- Create: `scripts/serve-qwen4exp-q4km.sh`

**Interfaces:**
- Consumes: `/home/random/dev/qwen/models/atomicchat-qwen38-q4km`,
  `Qwen4GgufPleStore`, QSA cache objects and FreeToken server arguments.
- Produces: a separate text-only server on a port other than 1919 and a
  machine-readable benchmark record.

- [ ] **Step 1: Write a failing cache contract test.**

```python
def test_qwen4exp_disallows_radix_resume_without_ple_and_qsa_state():
    config = make_qwen4exp_config()
    assert config.requires_naive_cache
```

- [ ] **Step 2: Verify RED.**

Run: `pytest tests/kvcache/test_qwen4exp_state_contract.py -q`

Expected: FAIL until the model exposes an explicit cache-state contract.

- [ ] **Step 3: Implement safe serve/download tooling.**

Download only the 33 requested files from AtomicChat into the dedicated model
directory with resume and checksum verification. The service script refuses
port 1919, exports `HF_HOME=/home/random/dev/qwen/cache`, sets model/log paths
under `/home/random/dev/qwen`, uses `--moe-backend offload`, and records all
runtime metrics.

- [ ] **Step 4: Verify GREEN and run opt-in integration.**

Run: `pytest tests/kvcache/test_qwen4exp_state_contract.py -q`

Then run: `scripts/download-qwen4exp-q4km.sh --verify-only` and
`benchmarks/run_qwen4exp_gguf_smoke.py --model-path ... --context 1024`.

Expected: all unit tests pass; the smoke record contains either a successful
strict text response or the exact unsupported operation without touching
Ornith.

- [ ] **Step 5: Commit.**

```bash
git add tests/kvcache scripts benchmarks
git commit -m "bench: add guarded Qwen4 GGUF runtime probe"
```
