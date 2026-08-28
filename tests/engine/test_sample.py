from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.engine import sample as sample_module
from freetoken.engine.sample import Sampler, _sample_with_generators


def test_seeded_top_p_retains_first_crossover_token(monkeypatch):
    """Nucleus filtering follows the kernel's inclusive crossover boundary."""
    observed: list[torch.Tensor] = []

    def capture_multinomial(probs, num_samples, *, generator):
        observed.append(probs.clone())
        return torch.zeros(num_samples, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    _sample_with_generators(
        torch.log(torch.tensor([[0.6, 0.4]])),
        torch.tensor([1.0]),
        top_k=None,
        top_p=torch.tensor([0.8]),
        generators=[torch.Generator().manual_seed(7)],
    )

    assert torch.allclose(observed[0], torch.tensor([0.6, 0.4]))


def test_seeded_top_p_retains_all_tokens_tied_at_cutoff(monkeypatch):
    """The kernel keeps every token at the first crossover probability."""
    observed: list[torch.Tensor] = []

    def capture_multinomial(probs, num_samples, *, generator):
        observed.append(probs.clone())
        return torch.zeros(num_samples, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    _sample_with_generators(
        torch.log(torch.tensor([[0.4, 0.3, 0.3]])),
        torch.tensor([1.0]),
        top_k=None,
        top_p=torch.tensor([0.5]),
        generators=[torch.Generator().manual_seed(7)],
    )

    assert torch.allclose(observed[0], torch.tensor([0.4, 0.3, 0.3]))


def test_sampler_entry_point_is_deterministic_for_seeded_cpu_request(monkeypatch):
    """Sampler.prepare + Sampler.sample preserve one seeded request RNG stream."""
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype, device=device),
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _name: nullcontext())

    def sample_twice() -> tuple[torch.Tensor, torch.Tensor]:
        req = SimpleNamespace(
            sampling_params=SamplingParams(temperature=1.0, top_p=1.0, seed=1234),
            sampling_generator=None,
        )
        sampler = Sampler(device=torch.device("cpu"), vocab_size=3)
        batch = SimpleNamespace(reqs=[req])
        logits = torch.tensor([[0.2, 0.5, 0.3]])
        first = sampler.sample(logits, sampler.prepare(batch))
        second = sampler.sample(logits, sampler.prepare(batch))
        assert req.sampling_generator is not None
        return first, second

    assert sample_twice() == sample_twice()
