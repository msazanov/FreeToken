# Ornith 1.5 35B A3B Q4_K_M on Turing

First end-to-end run of the official mixed-quant GGUF through FreeToken on an RTX 2070.

## Runtime

- GPU: NVIDIA GeForce RTX 2070, SM 7.5, 8192 MiB VRAM.
- Host RAM: 31 GiB; swap: 31 GiB.
- Fork: `msazanov/FreeToken`.
- Branch: `feat/ornith-q4km-turing`.
- Implementation head before this report: `d91a119`.
- Upstream GGUF base: PR #131 head `bb432e8`.
- Model: official `Ornith-1.5-35B-Q4_K_M.gguf`.
- Model bytes: 21,713,462,848 (21 GiB on disk).
- Served layers: 40; experts per layer: 256.
- Expert types: gate/up Q4_K on all layers; down Q6_K on 20 layers and Q4_K on 20 layers.

## Added support

The official `Q4_K_M` cannot use one compact expert stride for every layer because its
down projection changes between Q6_K and Q4_K. This branch adds:

- per-layer GGUF expert type metadata;
- compact native-width host banks;
- a max-width shared GPU slot pool per bank;
- fused copies with separate source width and destination slot stride;
- an MMVQ expert kernel that addresses slots using the tensor's real stride;
- per-layer quant-type dispatch;
- exact GPU-cache budget accounting for the widest layer;
- the existing local INT8/FP8 KV-cache work rebased onto the PR #131 GGUF implementation.

The expert-level padding stays only in VRAM. Q4_K host layers are not expanded to Q6_K
in RAM, and only native bytes cross PCIe.

## Working launch

```text
PYTHONPATH=/home/random/freetoken-turing/.worktrees/ornith-q4km-turing/python \
/home/random/freetoken-turing/.venv/bin/ft serve \
  --model-path /home/random/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-GGUF/snapshots/fbbaed45c2f0e200276ffa51701a24d45dc7f57e/Ornith-1.5-35B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 1919 \
  --served-model-name "Ornith 1.5 35b" \
  --moe-backend offload \
  --expert-load serial \
  --memory-ratio 0.85 \
  --max-running-requests 1 \
  --max-seq-len-override 122880 \
  --num-tokens 122880 \
  --kv-reserve-tokens 122880 \
  --moe-cache-auto \
  --disable-moe-prefill-overlap \
  --max-prefill-length 640 \
  --cuda-graph-max-bs 1 \
  --disable-pynccl \
  --cache-type radix \
  --enable-cache-report \
  --attention-backend triton \
  --kv-cache-dtype int8
```

Active transient unit at capture time:

```text
freetoken-ornith-q4km-122k.service
```

## Observed geometry and memory

- Native context exposed by the API: 122,880 tokens.
- INT8 KV allocation: 122,880 tokens, 1.18 GiB.
- Auto-sized MoE cache: 1,429 expert slots.
- VRAM used by the worker: about 6,934 MiB (`nvidia-smi`).
- Free VRAM after graph capture: 1.06 GiB.
- Service memory after startup: about 19.8 GiB RAM, 3.4 GiB attributed swap.
- Serial expert-bank build: about 27 seconds.
- End-to-end startup to ready: about 92 seconds, including config scan, bank load,
  CUDA graph capture, and prefill warmup.

## Correctness and performance smoke tests

- OpenAI `/v1/models` reports `context_length=122880`.
- Russian factual prompt returned `Столица Японии — Токио.` with HTTP 200.
- Arithmetic prompt returned `17 умножить на 6 будет 102.` with HTTP 200.
- A 255-token reasoning-limited run sustained 33.8-37.2 generated token/s after the
  first decode interval; an earlier short run reached 39.5 token/s.
- A cold 1,416-token synthetic prompt completed in 12.74 seconds: about 111 prompt
  token/s end-to-end.
- A cold 2,353-token synthetic prompt completed in 19.71 seconds: about 119 prompt
  token/s end-to-end.
- Long 640-token scheduler chunks commonly reported about 113-150 input token/s after
  the first cold chunk.

These prefill prompts are not byte-identical to the earlier NVFP4 baseline, so the figures
show the Q4_K_M runtime is healthy but are not yet a controlled A/B quality or speed verdict.

## Verification

- Mixed-stride CUDA copy test passed on SM75 and verifies that only the compact source
  prefix is written while destination padding remains untouched.
- GGUF MMVQ stride test passed on SM75 and produces bit-identical outputs between a
  compact bank and the same bank placed in padded expert slots.
- Targeted mixed-GGUF, cache-budget, dispatch, and CPU-format tests passed.
- Quantized-KV integration suite: 32 passed.
- Broad relevant suite: 93 passed; 13 failures were pre-existing environment gaps in
  that development worktree (`_pinned_tensor` not installed there and FlashInfer absent),
  not mixed-GGUF failures. The runtime uses the already-built pinned extension.

## Known follow-ups

- Install `triton_kernels` to remove the pure-PyTorch `fused_topk` fallback.
- Run a controlled, identical-prompt NVFP4 versus Q4_K_M benchmark and quality set.
- Mixed per-layer GGUF is currently intended for `--moe-backend offload`; the CPU/hybrid
  executor still requires one supported quant format for all expert banks and rejects this
  checkpoint clearly.
