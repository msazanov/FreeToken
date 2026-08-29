from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _PleTensor:
    def __init__(self, ggml_type: int, shape: tuple[int, int], row_bytes: int):
        self.name = "per_layer_token_embd.weight"
        self.ggml_type = ggml_type
        self.shape = shape
        self.row_bytes = row_bytes
        self._packed = torch.zeros((shape[0], row_bytes), dtype=torch.uint8)

    def packed(self) -> torch.Tensor:
        return self._packed


def _ple_embedding():
    from freetoken.models.qwen4_exp.model import _HostNGramEmbedding

    embedding = _HostNGramEmbedding(
        SimpleNamespace(
            qwen4_args=SimpleNamespace(
                ngram_size=2,
                heads_per_ngram=1,
                ple_embed_dim=32,
                eos_token_id=0,
                split_ngram_parts=1,
            )
        ),
        layer_id=0,
    )
    embedding.layer_multipliers.copy_(torch.tensor([1, 1]))
    embedding.ngram_heads_vocab_sizes.copy_(torch.tensor([4]))
    embedding.ngram_heads_offsets.copy_(torch.tensor([0]))
    return embedding


def _load_ple_tensor(monkeypatch, tensor: _PleTensor):
    from freetoken.models.gguf import reader

    monkeypatch.setattr(reader, "is_gguf_path", lambda _path: True)
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: iter((tensor,)))
    embedding = _ple_embedding()
    embedding.load_host_weights("fixture.gguf")
    return embedding


def test_qwen4_ple_accepts_q5_1_table(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_Q5_1

    embedding = _load_ple_tensor(monkeypatch, _PleTensor(GGML_Q5_1, (4, 32), 24))

    assert embedding._gguf_type == GGML_Q5_1
    assert embedding._gguf_table.shape == (4, 24)


def test_qwen4_ple_accepts_iq4_nl_table(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_IQ4_NL

    embedding = _load_ple_tensor(monkeypatch, _PleTensor(GGML_IQ4_NL, (4, 32), 18))

    assert embedding._gguf_type == GGML_IQ4_NL
    assert embedding._gguf_table.shape == (4, 18)


def test_qwen4_ple_rejects_unsupported_table_type(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_F16

    with pytest.raises(RuntimeError, match=r"supported.*F16"):
        _load_ple_tensor(monkeypatch, _PleTensor(GGML_F16, (4, 32), 64))


@pytest.mark.parametrize(
    ("shape", "row_bytes"),
    [
        ((4, 31), 24),
        ((4, 32), 23),
    ],
)
def test_qwen4_ple_rejects_unexpected_row_shape(monkeypatch, shape, row_bytes):
    from freetoken.models.gguf.dequant import GGML_Q5_1

    with pytest.raises(RuntimeError, match="Unexpected Qwen4-Exp GGUF PLE shape"):
        _load_ple_tensor(monkeypatch, _PleTensor(GGML_Q5_1, shape, row_bytes))
