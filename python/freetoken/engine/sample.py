from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    generators: List[torch.Generator | None] | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def _sample_with_generators(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generators: List[torch.Generator | None],
) -> torch.Tensor:
    """Torch fallback for requests that require a reproducible RNG stream.

    The kernel samplers consume their own RNG state and cannot accept a
    per-request generator. This path is intentionally opt-in (only when at
    least one request supplied ``seed``), preserving the fast kernel path for
    ordinary traffic while making a seed a real sampling control.
    """
    samples: list[torch.Tensor] = []
    for row, generator in enumerate(generators):
        scores = logits[row] / temperatures[row]
        if top_k is not None:
            k = int(top_k[row].item())
            if k < scores.numel():
                cutoff = torch.topk(scores, k).values[-1]
                scores = scores.masked_fill(scores < cutoff, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        if top_p is not None and float(top_p[row].item()) < 1.0:
            sorted_probs, _ = probs.sort(descending=True)
            # The kernel samples every probability >= the first cumulative
            # crossover's probability. Applying the resulting threshold to the
            # unsorted row keeps ties at that cutoff together.
            crossover = torch.searchsorted(
                sorted_probs.cumsum(dim=-1), top_p[row], right=False
            )
            threshold = sorted_probs[crossover]
            keep = probs >= threshold
            filtered = torch.zeros_like(probs)
            filtered[keep] = probs[keep]
            probs = filtered / filtered.sum()
        samples.append(torch.multinomial(probs, 1, generator=generator))
    return torch.cat(samples)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        generators = None
        if any(p.seed is not None for p in params):
            generators = []
            for req, params in zip(batch.reqs, params, strict=True):
                if params.seed is not None and req.sampling_generator is None:
                    req.sampling_generator = torch.Generator(device=self.device)
                    req.sampling_generator.manual_seed(params.seed)
                generators.append(req.sampling_generator)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, generators=generators)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            if args.generators is not None:
                return _sample_with_generators(
                    logits.float(), args.temperatures, args.top_k, args.top_p, args.generators
                )
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
