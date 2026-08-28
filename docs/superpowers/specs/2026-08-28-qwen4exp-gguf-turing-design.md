# Qwen3.8 Flash Next GGUF on Turing — Design

## Goal

Run the text path of AtomicChat's Qwen3.8 Flash Next `AD-4.27bpw-Q4_K_M-M64`
through FreeToken on RTX 2070 Mobile, with resident compute split across VRAM/RAM
and the isolated PLE table read from NVMe. The running Ornith service remains
untouched.

## Runtime boundary

All new runtime state belongs below `/home/random/dev/qwen`:

- code worktree: `/home/random/dev/qwen/freetoken`;
- model: `/home/random/dev/qwen/models/atomicchat-qwen38-q4km`;
- Hugging Face/Xet cache: `/home/random/dev/qwen/cache`;
- service and benchmark logs: `/home/random/dev/qwen/logs`.

The runtime starts text-only. Vision/mmproj and MTP are deliberately disabled;
they do not contribute to the coding-agent workload and increase the first
integration surface.

## Architecture

The current Ornith branch already provides a packed, multi-shard GGUF reader,
mixed Q4_K/IQ kernels, and offloaded expert banks. Qwen3.8 Flash Next needs a
separate `qwen4exp` model adapter because it adds QSA, Gated DeltaNet geometry,
Gated Residual, PLE and a top-10 router.

We use the Qwen4 implementation from FreeToken PR #232 as a source, not a
merge base: it predates and removes the current GGUF machinery. The implementation
ports its text-only model/QSA pieces into this branch, then adds a native GGUF
config and weight adapter. GGUF tensor names and metadata are the source of
truth; safetensors/FTW conversion is not part of the first boot path.

## PLE and cache invariants

The Q4_K_M file isolates PLE in shard 2. It must remain mmap-backed on NVMe and
must never be copied wholesale to RAM or VRAM. Per-request PLE convolution
state and incomplete-QSA state are cache state: a radix prefix may be reused
only when both can be restored. Until state restoration is implemented and
tested, the Qwen runtime explicitly selects the naive cache rather than
silently reporting false prefix reuse.

## Hardware and correctness constraints

- Target GPU is Turing SM 7.5; no Blackwell-only NVFP4 kernel is permitted.
- Experts use FreeToken's existing packed GGUF Q4_K/IQ offload path.
- The top-10 router uses the tested Torch fallback because ten is not a power
  of two for the external Triton top-k kernel.
- `Q5_1`, `IQ2_S`, `IQ3_S`, `IQ4_NL`, `Q4_K`, and `Q8_0` are accepted only
  through existing GGUF dispatch tables.
- Every new behavior starts with a failing unit test. GPU and full-model tests
  are opt-in and must record the exact revision, model checksum, context,
  prefill rate, TTFT, decode rate, expert-cache miss rate, RAM and VRAM.

## Acceptance criteria

1. A metadata-only fixture with `general.architecture=qwen4exp` dispatches to
   `Qwen4ExpGGUFForCausalLM` and rejects incomplete/unknown metadata clearly.
2. Qwen4 text-only config produces correct QSA/linear layer groups, PLE
   geometry, top-10 routing and safe cache capability flags.
3. The Q4_K_M shard directory validates as a complete 33-shard GGUF and maps
   all referenced tensors without materializing the PLE table.
4. A guarded local smoke test can boot the text model with FreeToken and log
   failures precisely; it cannot affect the Ornith service on port 1919.
5. Radix reuse is enabled only after PLE and QSA state restoration is covered
   by tests; otherwise the runtime advertises naive-cache mode honestly.
