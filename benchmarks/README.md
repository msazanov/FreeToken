# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

**`tq3_4s_kernel_bench.py`** — target-SM75 batch-one microbenchmark for the
packed Ornith TQ3_4S expert MMVQ kernel. It retains all CUDA-event samples for
the real 512x2048 gate/up and 2048x512 down matrices, compares numerical error
against exact FP32 CPU materialization, and times the exact CUDA
dequantize-plus-MM fallback. These are kernel milliseconds, not model tok/s.

```bash
MAX_JOBS=12 PYTHONPATH=python:. python benchmarks/tq3_4s_kernel_bench.py \
  --output benchmarks/results/ornith35-tq3-sm75-kernel/task4-mmvq.json
```

**`fit_tq3_4s_dp4a.py`** — CPU-only exhaustive reconstruction of the eight
signed int8 levels and shared scale used by the SM75 DP4A approximation. It
derives Gaussian Lloyd-Max bin masses from the authoritative centroids and
enumerates every reachable rounded table in the declared scale interval.

**`tq3_4s_moe_bench.py`** — resident-slot, batch-one top-8 routed SwiGLU
microbenchmark at Ornith's real H=2048/I=512 geometry. It saves every event and
an exact-FP32 quality comparison, plus hashes for the Python API/activation and
CUDA implementation it exercises. This isolates one hot MoE layer; cache
misses, PCIe/NVMe, non-MoE layers, prefill and model tok/s are deliberately
excluded.

**`tq3_4s_prefill_bench.py`** — FP16 sweep across 1/6/7/16/64/128/256/512/
1024 input tokens for a real-shape TQ3 dense projection and resident top-8
Ornith MoE layer. It labels the dense MMVQ→exact-materialized dispatch boundary,
retains every CUDA-event sample and records peak allocated/reserved VRAM. It is
still a layer benchmark: cache misses, PCIe/NVMe, attention/GDN and model tok/s
are excluded.

The first real checkpoint intake, failed generic-MHA TQ4 startup, successful
16K INT8 startup, deterministic short prompts and cold/warm 1K repository tasks
are preserved together under
`benchmarks/results/ornith35-tq3-sm75-smoke-task6-v1/`. The two
`compression-1024.json` files were produced by `ornith_context_bench.py` and
include SSE TTFT, full output, cache geometry and one-second GPU/host samples.
Warm-cache artifacts separate cached and newly processed prompt tokens; their
canonical `prefill_tps_estimate` is null rather than the misleading total
prompt/TTFT quotient. `run-provenance.json` pins the exact dirty source,
software stack, model revision and checkpoint used by the successful server.
Future `ornith_context_bench.py` runs require `--model-sha256`; this deliberately
uses the intake gate's precomputed digest instead of rereading tens of GiB and
perturbing the page cache during a context sweep. The runner also records local
model file stats, GPU UUID/name, compute capability and NVIDIA driver.
