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

The next separate candidate is Qwen3.8 REAP-256 GGUF: 256 instead of 512
experts per layer. It is being checked as a model-compression control through
the stable LRU runtime first; no performance or quality result is claimed until
the model's GGUF metadata passes and matched 1K/16K/64K profiles complete.

Each new profile now includes a prompt-private SHA-256 of the final visible
answer. Incomplete or error SSE streams are rejected before an artifact is
published, so throughput comparisons cannot accidentally use a partial output.

The downloading REAP checkpoint has already passed a header-only static gate:
it is a two-shard Qwen4Exp model with 48 layers, 256 experts/layer and top-10;
its expert quant types are covered by the fork's GPU MoE-vector kernel. Runtime
compatibility and performance remain unclaimed until the full shards verify.

The PLE loader also accepts the REAP checkpoint's `IQ4_NL`
`per_layer_token_embd.weight`: its gate follows the exact quantized types
handled by native `ggml_dequantize`. `Q5_1` remains covered by a regression
test, while unsupported types are rejected and the packed row shape is still
validated. See
`.superpowers/sdd/2026-08-29-qwen38-reap256-ple-iq4nl-report.md` for the
RED/GREEN record.
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
