# Ornith Q4_K_M on Turing Design

## Goal

Serve the official `ornith-ai/Ornith-1.5-35B-A3B-GGUF` Q4_K_M checkpoint with
FreeToken on an RTX 2070 (SM 7.5), retaining packed GGUF weights in RAM and the GPU
expert cache. llama.cpp is not a runtime option.

## Starting point

The implementation starts from FlashML-org/FreeToken PR #131 (`feat/generic-gguf`).
That branch already provides the qwen35moe GGUF adapter, Q4_K/Q6_K CUDA kernels,
long-prefill launch chunking, tokenizer fixes, and direct packed expert execution.
Turing support comes from PR #24 plus the SM75 fixes already present in PR #131.

The official Q4_K_M file has 40 served layers and one ignored MTP block. Its expert
layout is:

- `gate` and `up`: Q4_K in every served layer;
- `down`: Q6_K in 20 layers and Q4_K in 20 layers;
- all three expert stacks remain packed and are never materialized as FP16/BF16.

PR #131 rejects that file because one GPU slot tensor currently assumes that every
host layer has the same row shape and every cached expert has a quant-derived compact
stride.

## Cache layout

Keep one LRU slot namespace and one GPU tensor per bank. For each bank, allocate every
GPU slot using the maximum packed expert byte count found among its layers. Host banks
retain their native compact layouts. A cache miss copies only the native byte count of
the selected layer into the beginning of the destination slot; padding is neither
stored in RAM nor transferred over PCIe.

The fused index-copy descriptor therefore carries two independent values:

- `feature_bytes[layer, bank]`: bytes actually copied for one expert;
- `destination_stride_bytes[bank]`: distance between GPU cache slots.

For uniform formats they are equal, preserving current behavior. For mixed Q4_K_M
down layers the destination stride is the Q6_K expert size while Q4_K layers copy only
their smaller native payload.

## Kernel dispatch

The GGUF MoE CUDA launcher receives the cache tensor's actual expert stride in bytes.
The kernel uses this stride only to locate the selected slot; rows inside that slot
retain the existing quant-specific compact layout. Quant type is selected per layer
for `gate_up` and `down` independently.

This adds no dequantization stage, no second GPU pool, and no padded PCIe transfer.
Uniform-bank calls remain source-compatible and use the same arithmetic.

## Prefill and cache behavior

Decode miss copies, whole-layer materialization, double-buffered prefill, and hit-D2D
prefill all use the same per-layer copy geometry. The first implementation targets the
GPU offload backend. CPU and hybrid decode must reject genuinely mixed per-layer GGUF
banks with a precise error until the CPU executor accepts a quant type per layer.

Cache budgeting prices the maximum GPU slot geometry, because that is the memory
actually allocated. Host-bank estimates continue to sum native layer sizes.

## Turing constraints

The build must target SM 7.5 and avoid kernels requiring Ampere-only features. Q4_K and
Q6_K expert arithmetic remains in the vendored Q8_1 activation path using DP4A. The
existing SM75 attention fixes and the long-prefill `gridDim.z` fix are retained.

## Validation

Validation proceeds in increasing cost:

1. Pure geometry tests for mixed per-layer source sizes and maximum cache stride.
2. Copy tests proving native source bytes land at the correct padded destination slot.
3. CUDA kernel comparisons for Q4_K and Q6_K cached slots against dequantize + matmul.
4. Real-checkpoint tensor reconciliation for the local official Q4_K_M file.
5. FreeToken server smoke test at a small context, then 122880 with INT8 KV.
6. Identical-prompt comparison against the saved NVFP4 baseline: load time, RAM/VRAM,
   prefill, decode, PCIe/cache misses, and output sanity.

## Non-goals

- No llama.cpp runtime or fallback.
- No MTP speculative decoding in the first serving milestone.
- No re-quantization of the official Q4_K_M checkpoint.
- No silent fallback to BF16/FP16 expert materialization.
