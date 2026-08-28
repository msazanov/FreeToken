"""Encoder/decoder round-trips for the ZMQ control messages (no GPU).

Every message that crosses api -> tokenizer -> scheduler -> tokenizer -> api must survive the
wire with its fields intact; these pin the ones carrying state a later consumer reads back
(rebuild control, prompt admission, per-reply token deltas and KV usage).
"""

from __future__ import annotations

from freetoken.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserReply,
)
from freetoken.core import SamplingParams


def test_cache_rebuild_msg_roundtrip():
    msg = CacheRebuildMsg(request_id="abc", moe_cache_size=8, num_pages=1024, mode="if_idle")
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("abc", 8, 1024, "if_idle")


def test_cache_rebuild_backend_msg_roundtrip():
    msg = CacheRebuildBackendMsg(request_id="r1", moe_cache_size=None, num_pages=256, mode="drain")
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, CacheRebuildBackendMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("r1", None, 256, "drain")


def test_cache_rebuild_result_msg_roundtrip():
    msg = CacheRebuildResultMsg(request_id="r2", status="ok", moe_cache_size=16, num_pages=512)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildResultMsg)
    assert (out.request_id, out.status, out.moe_cache_size, out.num_pages, out.error) == (
        "r2", "ok", 16, 512, None,
    )


def test_cache_rebuild_reply_roundtrip():
    msg = CacheRebuildReply(request_id="r3", status="failed", error="boom")
    out = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))
    assert isinstance(out, CacheRebuildReply)
    assert (out.request_id, out.status, out.error) == ("r3", "failed", "boom")


def test_prompt_admitted_msg_roundtrip():
    msg = PromptAdmittedMsg(uid=42, prompt_tokens=1234, cached_tokens=500)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, PromptAdmittedMsg)
    assert (out.uid, out.prompt_tokens, out.cached_tokens) == (42, 1234, 500)


def test_user_reply_token_deltas_round_trip():
    moe_stats = {"schema_version": 1, "miss": {"miss_rate": 0.5}}
    msg = UserReply(
        uid=7,
        incremental_output="hello",
        finished=False,
        prompt_tokens_delta=11,
        completion_tokens_delta=3,
        cached_tokens=4,
        kv_used_pages=40,
        kv_total_pages=512,
        gpu_mem_bytes=64 * (1 << 30),
        moe_stats=moe_stats,
    )

    decoded = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))

    assert isinstance(decoded, UserReply)
    assert decoded.uid == 7
    assert decoded.incremental_output == "hello"
    assert decoded.finished is False
    assert decoded.prompt_tokens_delta == 11
    assert decoded.completion_tokens_delta == 3
    assert decoded.cached_tokens == 4
    assert decoded.kv_used_pages == 40
    assert decoded.kv_total_pages == 512
    assert decoded.gpu_mem_bytes == 64 * (1 << 30)
    assert decoded.moe_stats == moe_stats


def test_detokenize_msg_carries_kv_usage_round_trip():
    moe_stats = {"schema_version": 1, "miss": {"miss_rate": 0.5}}
    msg = DetokenizeMsg(
        uid=3, next_token=42, finished=True,
        kv_used_pages=10, kv_total_pages=256, gpu_mem_bytes=1 << 30,
        mamba_used_slots=7, mamba_total_slots=64,
        swa_used_tokens=8448, swa_total_tokens=76800,
        moe_stats=moe_stats,
    )
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(decoded, DetokenizeMsg)
    assert (decoded.kv_used_pages, decoded.kv_total_pages, decoded.gpu_mem_bytes) == (10, 256, 1 << 30)
    assert (decoded.mamba_used_slots, decoded.mamba_total_slots) == (7, 64)
    assert (decoded.swa_used_tokens, decoded.swa_total_tokens) == (8448, 76800)
    assert decoded.moe_stats == moe_stats


def test_old_user_reply_payload_decodes_with_new_optional_field():
    payload = {
        "__type__": "UserReply",
        "uid": 7,
        "incremental_output": "done",
        "finished": True,
    }

    decoded = BaseFrontendMsg.decoder(payload)

    assert isinstance(decoded, UserReply)
    assert decoded.moe_stats is None


def test_old_detokenize_payload_decodes_with_new_optional_field():
    payload = {
        "__type__": "DetokenizeMsg",
        "uid": 7,
        "next_token": 42,
        "finished": True,
    }

    decoded = BaseTokenizerMsg.decoder(payload)

    assert isinstance(decoded, DetokenizeMsg)
    assert decoded.moe_stats is None


def test_absent_moe_stats_is_omitted_from_new_wire_messages():
    user_reply = UserReply(uid=7, incremental_output="", finished=True)
    detokenize = DetokenizeMsg(uid=7, next_token=42, finished=True)

    assert "moe_stats" not in BaseFrontendMsg.encoder(user_reply)
    assert "moe_stats" not in BaseTokenizerMsg.encoder(detokenize)


def test_frontend_batch_omits_absent_nested_moe_stats_and_reads_old_payload():
    encoded = BaseFrontendMsg.encoder(
        BatchFrontendMsg(data=[UserReply(uid=7, incremental_output="", finished=True)])
    )

    assert "moe_stats" not in encoded["data"][0]

    decoded = BaseFrontendMsg.decoder(
        {
            "__type__": "BatchFrontendMsg",
            "data": [
                {
                    "__type__": "UserReply",
                    "uid": 7,
                    "incremental_output": "done",
                    "finished": True,
                }
            ],
        }
    )

    assert isinstance(decoded, BatchFrontendMsg)
    assert decoded.data[0].moe_stats is None


def test_tokenizer_batch_omits_absent_nested_moe_stats_and_reads_old_payload():
    encoded = BaseTokenizerMsg.encoder(
        BatchTokenizerMsg(data=[DetokenizeMsg(uid=7, next_token=42, finished=True)])
    )

    assert "moe_stats" not in encoded["data"][0]

    decoded = BaseTokenizerMsg.decoder(
        {
            "__type__": "BatchTokenizerMsg",
            "data": [
                {
                    "__type__": "DetokenizeMsg",
                    "uid": 7,
                    "next_token": 42,
                    "finished": True,
                }
            ],
        }
    )

    assert isinstance(decoded, BatchTokenizerMsg)
    assert decoded.data[0].moe_stats is None


def test_client_dicts_with_the_wire_tag_key_survive_intact():
    """Tool JSON Schemas and chat_template_kwargs are free-form client data. A field literally
    named ``__type__`` (a common discriminator) must not be read back as a serialized class --
    that used to kill the tokenizer worker on an unknown/incompatible name."""
    hostile = [
        {"__type__": "AbortMsg"},                                    # a real class name
        {"__type__": "NoSuchClassAnywhere"},                         # an unknown one
        {"type": "object", "properties": {"__type__": {"type": "string"}}},
        {"__raw_dict__": {"a": 1}},                                  # collides with the escape key
        {"deep": {"__type__": "AbortMsg", "l": [{"__type__": "x"}]}},
    ]
    for payload in hostile:
        msg = TokenizeMsg(
            uid=1, text="hi", sampling_params=SamplingParams(),
            chat_template_kwargs=payload,
            tools=[{"type": "function", "function": {"name": "f", "parameters": payload}}],
        )
        out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
        assert isinstance(out, TokenizeMsg)
        assert out.chat_template_kwargs == payload
        assert out.tools[0]["function"]["parameters"] == payload
