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
