from __future__ import annotations

from types import SimpleNamespace


def test_gemma4_tool_response_opener_is_an_end_of_generation_token(monkeypatch):
    """A parsed Gemma tool call must stop before its raw response marker leaks as content."""

    from freetoken.models.gguf import tokenizer as gguf_tokenizer

    tokens = ["<unk>", "<eos>", "<turn|>", "<|tool_response>"]
    monkeypatch.setattr(
        gguf_tokenizer,
        "load_gguf_metadata",
        lambda _path: {
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.eos_token_id": 1,
        },
    )
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _path: "gemma4")

    stop_ids = gguf_tokenizer.gguf_eos_token_ids(
        "gemma.gguf", SimpleNamespace(eos_token_id=1)
    )

    assert stop_ids == {1, 2, 3}
