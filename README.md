<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/xzwSnMdsX"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 

## RTX 2070 Qwen3.8 research status

This fork keeps reproducible Turing results for Qwen3.8 Flash Next on RTX 2070
Mobile 8 GiB, i7-8750H, 32 GiB RAM and NVMe. The Q4_K_M 64K control completed
at 38.16 prefill tok/s and 1.614 end-to-end decode tok/s; it established that
the 256-slot global cache thrashes across layers (zero decode L1 hits) rather
than exposing a fresh-NVMe bottleneck. See [TESTLOG.md](TESTLOG.md) for raw
artifacts and failures retained as evidence.

The separate Qwen3.8 REAP-256 GGUF candidate has 256 instead of 512 experts per
layer. Its matched 1K/16K/64K stable-LRU speed controls are complete. They show
a repeated decode-speed advantage, but are not a quality comparison and do not
show a decode-cache hit: a cache-policy A/B remains a separate experiment.

Each new profile now includes a prompt-private SHA-256 of the final visible
answer. Incomplete or error SSE streams are rejected before an artifact is
published, so throughput comparisons cannot accidentally use a partial output.

The REAP checkpoint first passed a header-only static gate:
it is a two-shard Qwen4Exp model with 48 layers, 256 experts/layer and top-10;
its expert quant types are covered by the fork's GPU MoE-vector kernel. Runtime
compatibility and performance were deliberately left unclaimed until the full
shards verified and a fixed-seed live control completed.

That verification and the first live control are now complete: the REAP-256
checkpoint finished the isolated fixed-seed 1K FreeToken run on the RTX 2070
with stable global LRU.  Its end-to-end prefill/decode were 14.63 / 2.175 tok/s
versus 12.10 / 1.834 tok/s for the Q4_K_M control.  This is a speed observation,
not a quality verdict. Decode still had zero L1 hits, so pruning does not by
itself solve global cache thrash; the complete artifact is
`benchmarks/results/qwen38-reap256-1k-lru/context-1024.json`.

The matched 16K control is also complete: 16,396 input tokens and 254 generated
tokens took 567.03 s, with 36.72 prefill tok/s and 2.100 end-to-end decode
tok/s. Against Q4_K_M's 616.26 s, 35.21 and 1.686 respectively, that is -8.0%
wall time, +4.3% prefill and +24.5% decode. The evidence is deliberately
qualified: REAP's decode trace is still `0 / 121,920` L1 hits/misses and its
H2D volume is larger (276.94 GB vs 252.24 GB). It is a stable-LRU model control,
not proof that the cache problem has been solved. The raw artifact is
`benchmarks/results/qwen38-reap256-16k-lru/context-16384.json`.

The final matched 64K control completed 65,548 input and 254 generated tokens
in 1,838.30 s: 38.229 prefill tok/s and 2.046 end-to-end decode tok/s. Compared
with Q4_K_M's 1,874.98 s, 38.162 and 1.614, that is -2.0% wall time, +0.2%
prefill and +26.7% decode. Decode remains a complete global-LRU miss (`0 /
121,920` hits/misses) and REAP read more physical bytes in this run; the result
does not diagnose an NVMe improvement. Artifact:
`benchmarks/results/qwen38-reap256-64k-lru/context-65536.json`.

## Cross-model RTX 2070 context-speed registry

`benchmarks/results/model-context-speed.jsonl` is the append-only, plot-ready
source of truth for every completed end-to-end context-speed point. The runner
now appends future terminal artifacts automatically; a duplicate artifact is
rejected rather than rewritten. Each row binds actual input/output tokens,
prefill and decode throughput, seed, quantization, runtime profile, revision and
the immutable raw JSON path.

The first normalized cross-model slice uses the same fixed prompt generator,
temperature `0` and seed `20260828`. It is a useful hardware comparison, not a
one-variable A/B: Qwen needs `tq4-nc`/256-slot LRU while Ornith uses the
validated 122,880-token INT8-KV/1,429-slot profile. Each model's tokenizer and
chat template also give a slightly different actual prompt length.

| Requested context | Qwen3.8 REAP-256 Q3_K_XL prefill / decode | Ornith 1.5 35b Q4_K_M prefill / decode | Ornith decode factor |
| ---: | ---: | ---: | ---: |
| 1K | 14.63 / 2.17 tok/s | 63.54 / 29.08 tok/s | 13.4× |
| 16K | 36.72 / 2.10 tok/s | 91.75 / 21.00 tok/s | 10.0× |
| 64K | 38.23 / 2.05 tok/s | 53.45 / 13.59 tok/s | 6.6× |

At 16K, the direct chunk A/B has now completed: `1024` measured 97.83 prefill
tok/s and 21.48 decode tok/s, versus 91.75 / 21.00 with `640` (+6.6% / +2.3%),
with only about 40 MiB more sampled VRAM. The terminal output lengths differed
(108 versus 127), so this is a throughput/capacity result, not exact-output
equivalence. Raw artifacts:
`benchmarks/results/ornith35-q4km-r2/context-1024.json` and
`benchmarks/results/ornith35-q4km-r3/context-{16384,65536}.json`.

The PLE loader also accepts the REAP checkpoint's `IQ4_NL`
`per_layer_token_embd.weight`: its gate follows the exact quantized types
handled by native `ggml_dequantize`. `Q5_1` remains covered by a regression
test, while unsupported types are rejected and the packed row shape is still
validated. See
`.superpowers/sdd/2026-08-29-qwen38-reap256-ple-iq4nl-report.md` for the
RED/GREEN record.

The follow-up geometry gate is also recorded: for `IQ4_NL`, the loader rejects
`ple_embed_dim` values that are not divisible by 256, matching the native
dequantizer's complete-block writes. The fixture now matches Qwen's exact
`ngram_size=3`, `heads_per_ngram=8` layout (16 total heads × 160 values,
`ple_embed_dim=2560`) with 90 packed bytes per row, and drives the complete
host PLE route through n-gram IDs, row selection, dequantization and reshape.
The CUDA gate uses a deterministic dequantization reference. CPU tests passed;
the CUDA test was cleanly skipped where no NVIDIA driver was available. See
`.superpowers/sdd/2026-08-29-qwen38-reap256-ple-iq4nl-geometry/task-ple-iq4nl-geometry-report.md`.
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## RTX 2070 fork mission

This fork focuses on useful, fast and high-quality local inference on a constrained
mobile workstation: **RTX 2070 Mobile (8 GiB VRAM), Intel i7-8750H, 32 GiB DDR4
and 1 TB NVMe**. The current primary target is **Ornith 1.5 35B A3B**: its MoE
architecture is the most effective candidate found so far for this hardware when
served by FreeToken with CPU/RAM/NVMe-assisted expert offload.

Every performance or quality claim in this fork must be reproducible. See
[TESTLOG.md](TESTLOG.md) for raw benchmark records and [CHANGELOG.md](CHANGELOG.md)
for hypotheses, changes, successful experiments and rejected experiments.

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).

## RTX 2070 live benchmark ledger

Every local performance attempt is retained in
`benchmarks/results/benchmark-events.jsonl`: successful artifacts, startup
incidents, OOMs, timeouts and manually aborted runs are all first-class events.
The prompt-private runner stores no prompt, visible answer or reasoning text.
For a terminal stream it records TTFT, exact final usage, a SHA-256 of visible
content, phase-labelled one-second GPU/VRAM/power/temperature/process-I/O
samples, and timestamped counts of SSE deltas. The latter distinguish content
from reasoning deltas without retaining either text.

Rebuild the all-results graph after a run:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/benchmark_ledger.py
PYTHONPATH=python .venv/bin/python benchmarks/render_benchmark_dashboard.py
```

The generated `benchmarks/results/benchmark-dashboard.html` contains every
ledger event, including red failure marks, alongside the speed/context and
prefill-chunk plots. The first forced long Ornith trace (`16,400` input +
`4,095` generated tokens, Q4_K_M, INT8 KV, chunk `1024`) measured 99.86 prefill
tok/s, 19.52 decode tok/s and 164.24 s TTFT; see `TESTLOG.md` for scope and
interpretation before comparing it with short EOS-limited runs.

### Candidate quantizations under investigation

The production control remains the official `Q4_K_M`. A Hugging Face intake on
2026-08-30 identified four plausible next FreeToken/Turing experiments:

1. AtomicChat `AD-Q4_K-IQ4_XS` (20.13 GB) — its publisher reports a materially
   better BF16-reference divergence than stock Q4_K_M at about 1 GB less disk
   space. Both `Q4_K` and `IQ4_XS` have native FreeToken GGUF kernels.
2. `UD-Q4_K_S` / `UD-IQ4_XS` from the Unsloth-Dynamic-style Ornith release —
   per-tensor, imatrix-calibrated GGUF layouts that deliberately remove the
   untrained MTP block. They need a live quality and throughput comparison; no
   third-party FreeToken result is assumed.
3. REAP-50 source weights — 128 rather than 256 routed experts while retaining
   top-8 routing. Re-quantizing that source to a native GGUF K-quant is a
   high-risk/high-upside cache-working-set experiment. Its published NVFP4A16
   build is not selected directly because NVFP4 is non-native on SM75.
4. APEX Compact (16.54 GB) — role-aware mixed precision that compresses routed
   experts more heavily; it needs a GGUF type/startup gate before download.

`TQ3_4S` is not directly loadable yet, but the exact tensor audit promoted it
to the highest-upside **runtime-port** candidate. Its served expert slot is
1,572,864 bytes, 22.89% below Q4_K_M, while its dense served weights are also
smaller. Under the same packed-weight VRAM budget this projects roughly 1,943
slots instead of the measured 1,429; that is a first-order capacity estimate,
not a speed result. Correct execution requires the matching activation WHT,
Lloyd-Max centroids and native SM75 `dp4a` vector-dot path from
`turbo-tan/llama.cpp-tq3`.

The follow-up header audit also changes how the direct AD/UD candidates should
be read. Atomic AD has the strongest published same-corpus quality evidence and
smaller expert transfers, but its Q8 non-expert weights consume about 0.94 GiB
more than the served Q4_K_M control; at a fixed total GPU packed-weight budget
it may therefore hold fewer, not more, expert slots. `UD-Q4_K_S` keeps the same
maximum slot stride as Q4_K_M because three Q6_K down-projection layers set the
global pool stride. These are quality/compatibility experiments rather than
assumed speed wins. Qwen-GGUF MXFP4 remains unsupported by the generic loader,
and Ornith MTP is a separate speculative-decoding project. Detailed evidence
and test gates are in `TESTLOG.md`.

Before the type-46 port, this branch also absorbed FreeToken upstream through
`58f4b9e` (including the official Qwen3.8 runtime) while preserving the existing
GGUF/Turing implementation as a separate compatibility path. The combined CPU
gate is `245 passed, 139 skipped`; the merge fixed two silent GGUF parity bugs:
Qwen3.8 now renormalizes its selected top-10 expert weights and the GDN output
gates use the upstream string contracts (`sigmoid` for Qwen3.8, `silu` for
Ornith/Qwen3.5).

The first `TQ3_4S` weight A/B deliberately keeps the current KV format unchanged.
`TQ3_4S` model weights and a three-bit TurboQuant KV cache are different codecs:
the former stores offline-transformed weight blocks, while KV must encode dynamic
attention vectors online. At 122,880 tokens a hypothetical three-bit KV codec
using the denser `TURBO3_0` layout would save about 140.625 MiB versus `tq4-nc`,
worth roughly 70 additional TQ3 expert slots; the weight port itself projects
about 514 additional slots. Therefore KV
is a later isolated A/B, not part of the initial weight result.

The metadata/CPU gate for `TQ3_4S` is now implemented. FreeToken recognizes
GGML type 46 as `32 values / 16 bytes`, can read it even when the installed
`gguf-py` enum is stale, and has a literal-byte pure-Torch authority for E3M5,
3-bit centroid unpacking and inverse signed WHT. The real sparse Ornith header
enumerated all 753 tensors, including 381 TQ3 tensors totaling
17,230,725,120 packed bytes.

The next correctness slice is also complete: the exact materialized CUDA
dequantizer was compiled natively for `sm_75` and exercised on the RTX 2070 in
FP32, FP16 and BF16. Fixed random blocks matched the CPU authority bit-for-bit
(`max_abs=0`, `mean_abs=0`); the combined GGUF/SM75 gate is `77 passed, 1
skipped`. This only establishes the numeric GPU authority. MMVQ, fused MoE,
real-model generation, quality and tokens/second remain unproven.

The following dense batch-one MMVQ slice is now measured on the same RTX 2070.
Type 46 rotates each activation block with the matching signed normalized WHT,
quantizes it to Q8_1, then uses Turing-native `dp4a` directly against the packed
weights. The donor fork's reused symmetric levels produced about 6.7% relative
L2 error against the exact asymmetric TQ3_4S centroids. A least-squares-fitted
int8 table reduced real expert-matrix relative L2 to 0.55-0.61% without adding
instructions to the PRMT+DP4A path.

| Real Ornith matrix | dtype | packed MMVQ p50 | exact dequant + MM p50 | kernel speedup | relative L2 |
|---|---:|---:|---:|---:|---:|
| gate/up 512x2048 | FP16 | 0.025872 ms | 0.137312 ms | 5.307x | 0.605% |
| gate/up 512x2048 | BF16 | 0.025136 ms | 0.137616 ms | 5.475x | 0.580% |
| down 2048x512 | FP16 | 0.025472 ms | 0.139152 ms | 5.463x | 0.615% |
| down 2048x512 | BF16 | 0.025040 ms | 0.139088 ms | 5.555x | 0.555% |

All 800 ordered timing samples and content hashes for every benchmark/kernel
source are in
`benchmarks/results/ornith35-tq3-sm75-kernel-task4-fitted-v3/tq3-mmvq.json`.
The canonical run alternates fast and fallback order on every sample and carries
an executable quality gate. The earlier v1 (5.85-6.15x) and v2 (5.70-5.74x)
runs are retained as intermediate artifacts rather than overwritten. The fitted
constants are independently reproducible from `fit_tq3_4s_dp4a.py` and its v3
JSON result.
This is not yet an Ornith tok/s result: fused selected-expert MoE, large-batch
prefill, real checkpoint loading, routing/cache behavior and model quality are
still separate gates. The KV control therefore remains `tq4-nc`; changing it at
the same time would hide whether any later end-to-end gain came from weights or
KV storage.
