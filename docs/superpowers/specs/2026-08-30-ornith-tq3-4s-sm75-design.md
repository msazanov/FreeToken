# Ornith TQ3_4S on SM75 Design

## Goal

Serve `YTan2000/Ornith-1.5-35B-A3B-TQ3_4S` through FreeToken on the
RTX 2070 (SM 7.5), preserving the packed TQ3_4S expert weights from NVMe through
RAM and the VRAM expert cache. The first live milestone isolates weight decode
with FreeToken's known-good generic-MHA `int8` KV-cache format.

## Why this experiment is different from 3-bit KV

TQ3_4S is GGML type 46 for model weights: each block stores 32 transformed weight
indices in 16 bytes, or exactly 4.0 stored bits per weight after scale overhead.
The closest TurboQuant KV format is TQ3_0, a separate block layout and runtime
algorithm. A TQ3_4S weight kernel cannot read a TQ3_0 KV page, and KV quantization
does not reduce expert-transfer bytes.

The original plan named `tq4-nc` as the constant control. Real startup proved
that this branch has packed TQ4 storage and a dedicated QSA reader but no generic
MHA packed decoder; it failed at CUDA-graph warmup on physical-width versus
logical-width head dimensions. The corrected weight experiment therefore keeps
`int8` KV constant. Only after weight parity and live decode measurements pass
will a separate KV A/B compare memory, prefill, decode, and expert-cache effects.

## TQ3_4S representation

The format is copied from `turbo-tan/llama.cpp-tq3` rather than inferred from its
name:

- enum: `GGML_TYPE_TQ3_4S = 46`;
- block: 32 logical weights in 16 bytes;
- payload: four encoded scales plus twelve packed 3-bit indices;
- codebook: the eight TQ3 Lloyd-Max centroids;
- transform: a fixed signed 32-point Walsh-Hadamard transform (WHT) applied during
  quantization and inverted during materialized dequantization;
- vector path: activations receive the matching WHT before Q8_1 quantization, then
  the packed weight dot product uses Turing-native DP4A.

Metadata-only support is explicitly invalid: decoding the packed indices without
the paired activation transform produces plausible-shaped but numerically corrupt
output.

## Runtime path

The initial vertical slice targets FreeToken's decode path:

1. The GGUF reader recognizes type 46 even when the installed `gguf-py` table is
   older than the producing fork.
2. Python type tables expose the exact 32/16 geometry.
3. A pure-Torch reference decoder provides deterministic CPU oracle values.
4. CUDA materialized dequantization matches that oracle.
5. The small-batch linear MMVQ path applies transformed Q8_1 activations and uses
   a TQ3_4S DP4A vector dot product.
6. The fused MoE vector path uses the same primitive while reading only the
   selected packed expert slots.

Large-batch MMQ/prefill is a second milestone. Until a matching SM75 kernel is
verified, TQ3_4S prefill may use a correctness-first materialized fallback in
bounded chunks; it must never silently use an untransformed Q8_1 path.

## Cache geometry

The official Q4_K_M control uses a maximum 2,039,808-byte expert slot. TQ3_4S uses
1,572,864 bytes per slot, 22.89% less. Under the same packed VRAM budget, the
first-order capacity rises from 1,429 to about 1,943 slots. Live tests cover both:

- fixed 1,429 slots, isolating lower transfer/dequantization cost;
- automatic capacity, testing whether roughly 36% more slots reduce misses.

The offload cache stays opaque and packed. No expert is expanded to FP16/BF16 in
RAM or copied expanded across PCIe.

## Correctness and quality gates

The implementation must pass, in this order:

1. Exact type/row-byte and reader compatibility tests.
2. Golden single-block CPU decode copied from the producing fork's algorithm.
3. CUDA dequantization parity against the CPU oracle.
4. Dense MMVQ and fused-MoE parity against materialized BF16 matmul on SM75.
5. Header-only Ornith tensor reconciliation without downloading model payloads.
6. Short deterministic real-model prompts with finite logits and coherent output.
7. Identical-seed A/B quality prompts against Q4_K_M before speed claims.

Fast-vector approximations in the source fork use integer levels that are not
bit-for-bit identical to its materialized centroid decoder. Their error must be
measured; they are not accepted solely because upstream uses them.

## Measurements

Every live run records immutable artifacts under `benchmarks/results/` and appends
the hypothesis and outcome to `TESTLOG.md` and `CHANGELOG.md`. Required fields are:

- cold and warm prefill tok/s and TTFT;
- decode tok/s sampled through the whole generation;
- expert requests, hits, misses, evictions, bytes copied, and miss latency;
- slot count and per-layer expert working set;
- GPU utilization, VRAM, power and temperature;
- CPU utilization, RAM, swap, NVMe reads and major faults;
- output hash, seed, prompt identity, and quality-sanity result.

## KV follow-up

After the TQ3_4S weight A/B, compare current `int8` with a separate TQ3_0-style
KV prototype at 16K, 64K, and 122880 context. Do not reuse the TQ3_4S weight
codec. Prefer an asymmetric first prototype—INT8 K plus TQ3_0 V—because K errors
directly perturb attention scores and the checkpoint publisher likewise keeps K
at Q8 while using TQ3 only for V. The 3-bit KV mode is accepted only if the VRAM
saved creates enough additional expert slots to improve end-to-end decode or
enables a useful context that otherwise cannot run. Lower KV bytes alone are not
a success criterion.

## Non-goals

- No llama.cpp runtime or fallback.
- No Blackwell NVFP4/TQ3 kernels on SM75.
- No full-model download before metadata and kernel parity gates pass.
- No simultaneous change of weight quantization and KV quantization in one A/B.
- No speed claim from prefill-only telemetry.
