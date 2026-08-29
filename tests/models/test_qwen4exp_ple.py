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
    from freetoken.models.qwen4_exp.gguf_model import _HostNGramEmbedding

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


def _real_geometry_ple_embedding(*, ple_embed_dim: int = 2560):
    from freetoken.models.qwen4_exp.gguf_model import _HostNGramEmbedding

    # This is the production Qwen3.8 Flash Next layout: two n-gram groups
    # (2-gram and 3-gram), each with eight heads.
    ngram_size = 3
    heads_per_ngram = 8
    ngram_heads = 16  # (ngram_size - 1) * heads_per_ngram
    embedding = _HostNGramEmbedding(
        SimpleNamespace(
            qwen4_args=SimpleNamespace(
                ngram_size=ngram_size,
                heads_per_ngram=heads_per_ngram,
                ple_embed_dim=ple_embed_dim,
                eos_token_id=0,
                split_ngram_parts=1,
            )
        ),
        layer_id=0,
    )
    embedding.layer_multipliers.copy_(torch.tensor([1, 1, 1]))
    embedding.ngram_heads_vocab_sizes.copy_(torch.ones(ngram_heads, dtype=torch.long))
    embedding.ngram_heads_offsets.copy_(torch.arange(ngram_heads, dtype=torch.long))
    return embedding


def _load_ple_tensor(monkeypatch, tensor: _PleTensor, embedding=None):
    from freetoken.models.gguf import reader

    monkeypatch.setattr(reader, "is_gguf_path", lambda _path: True)
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: iter((tensor,)))
    embedding = _ple_embedding() if embedding is None else embedding
    embedding.load_host_weights("fixture.gguf")
    return embedding


def test_qwen4_ple_accepts_q5_1_table(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_Q5_1

    embedding = _load_ple_tensor(monkeypatch, _PleTensor(GGML_Q5_1, (4, 32), 24))

    assert embedding._gguf_type == GGML_Q5_1
    assert embedding._gguf_table.shape == (4, 24)


def test_qwen4_ple_accepts_iq4_nl_table(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_IQ4_NL

    embedding = _load_ple_tensor(
        monkeypatch,
        _PleTensor(GGML_IQ4_NL, (16, 160), 90),
        _real_geometry_ple_embedding(),
    )

    assert embedding.ngram_heads == 16
    assert embedding.head_dim == 160
    assert embedding.embedding_dim == 2560
    assert embedding._gguf_type == GGML_IQ4_NL
    assert embedding._gguf_table.shape == (16, 90)


def test_qwen4_ple_forward_uses_actual_ngram_layout(monkeypatch):
    from freetoken.kernel import gguf as gguf_kernel
    from freetoken.models.gguf.dequant import GGML_IQ4_NL
    from freetoken.models.qwen4_exp import gguf_model as qwen4_model

    table = _PleTensor(GGML_IQ4_NL, (16, 160), 90)
    table._packed[:, 0] = torch.arange(16, dtype=torch.uint8)
    embedding = _load_ple_tensor(monkeypatch, table, _real_geometry_ple_embedding())
    request = SimpleNamespace(
        input_ids=torch.tensor([7, 11]),
        cached_len=0,
        device_len=2,
        extend_len=2,
    )
    batch = SimpleNamespace(
        is_decode=False,
        reqs=[request],
        input_ids=request.input_ids,
    )
    monkeypatch.setattr(qwen4_model, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    calls = []

    def fake_dequantize(rows, ggml_type, row_count, width, dtype):
        calls.append((rows.clone(), ggml_type, row_count, width, dtype))
        return rows[:, :1].to(dtype).expand(-1, width).contiguous()

    monkeypatch.setattr(gguf_kernel, "ggml_dequantize", fake_dequantize)

    result = embedding.forward(torch.device("cpu"), torch.float16)

    assert result.shape == (2, 2560)
    assert result.dtype == torch.float16
    assert len(calls) == 1
    rows, ggml_type, row_count, width, dtype = calls[0]
    assert (ggml_type, row_count, width, dtype) == (GGML_IQ4_NL, 32, 160, torch.float16)
    assert torch.equal(rows[:, 0], torch.arange(16, dtype=torch.uint8).repeat(2))
    assert torch.equal(result[0].view(16, 160)[:, 0], torch.arange(16, dtype=torch.float16))
    assert torch.equal(result[1].view(16, 160)[:, 0], torch.arange(16, dtype=torch.float16))


def test_qwen4_ple_rejects_iq4_nl_non_block_aligned_embedding_dim(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_IQ4_NL

    # The row is valid IQ4_NL (32 values, 18 bytes); only the aggregate
    # embedding is unsafe because the CUDA kernel emits complete 256-value
    # blocks before the eventual view into embedding_dim.
    with pytest.raises(RuntimeError, match=r"IQ4_NL PLE.*32.*256"):
        _load_ple_tensor(
            monkeypatch,
            _PleTensor(GGML_IQ4_NL, (1, 32), 18),
        )


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


def test_iq4_nl_dequantizes_real_ple_geometry_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the IQ4_NL dequantization gate")

    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import GGML_IQ4_NL

    # Five valid IQ4_NL blocks per row: fp16 scale 1.0 followed by 16 packed
    # nibbles. Distinct bytes make accidental row/stride mistakes observable.
    packed = torch.zeros((16, 90), dtype=torch.uint8)
    quant_bytes = torch.arange(16 * 16, dtype=torch.uint8).view(16, 16)
    for block in range(5):
        start = block * 18
        packed[:, start] = 0
        packed[:, start + 1] = 0x3C  # little-endian fp16(1.0)
        packed[:, start + 2 : start + 18] = quant_bytes + block * 16

    values = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32,
    )
    expected_blocks = []
    for block in range(5):
        start = block * 18 + 2
        q4 = packed[:, start : start + 16]
        expected_blocks.append(torch.cat((values[q4 & 0x0F], values[q4 >> 4]), dim=1))
    expected = torch.cat(expected_blocks, dim=1).to(torch.float16).cuda()
    packed = packed.cuda().contiguous()

    first = ggml_dequantize(packed, GGML_IQ4_NL, 16, 160, torch.float16)
    second = ggml_dequantize(packed, GGML_IQ4_NL, 16, 160, torch.float16)
    torch.cuda.synchronize()

    assert first.shape == (16, 160)
    assert first.dtype == torch.float16
    assert first.device.type == "cuda"
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.equal(first, expected)
