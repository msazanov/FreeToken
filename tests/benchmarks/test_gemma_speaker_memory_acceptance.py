from __future__ import annotations

import json

import httpx


class _Response:
    def __init__(self, body=None, *, status_code=200, text=""):
        self._body = body
        self.status_code = status_code
        self.text = text

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Client:
    def __init__(self, response):
        self.response = response

    def post(self, _url, *, json):
        return self.response


class _BrokenStreamingClient:
    def stream(self, *_args, **_kwargs):
        raise httpx.ConnectError("refused")


class _BrokenPostClient:
    def post(self, *_args, **_kwargs):
        raise httpx.ConnectError("refused")


class _MalformedStreamResponse:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_lines(self):
        return iter(("data: not-json", "data: [DONE]"))


class _MalformedStreamingClient:
    def stream(self, *_args, **_kwargs):
        return _MalformedStreamResponse()


def test_tool_probe_turns_non_json_upstream_failure_into_failed_result():
    from benchmarks.gemma_speaker_memory_acceptance import _tool_probe

    response = _Response(
        json.JSONDecodeError("bad", "<html>", 0),
        status_code=502,
        text="<html>bad gateway</html>",
    )

    result = _tool_probe(_Client(response), "http://test", "gemma-4-e2b", 1, 0.2, 128)

    assert result["passed"] is False
    assert result["http_status"] == 502
    assert result["response"]["error"]["body"] == "<html>bad gateway</html>"


def test_tool_probe_turns_buffered_connection_failure_into_failed_result():
    from benchmarks.gemma_speaker_memory_acceptance import _tool_probe

    result = _tool_probe(_BrokenPostClient(), "http://test", "gemma-4-e2b", 1, 0.2, 128)

    assert result["passed"] is False
    assert result["http_status"] is None
    assert result["response"]["error"]["type"] == "ConnectError"


def test_tool_probe_rejects_non_object_tool_arguments_without_crashing():
    from benchmarks.gemma_speaker_memory_acceptance import _tool_probe

    response = _Response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "speaker_memory_remember_name",
                                    "arguments": "[]",
                                }
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    result = _tool_probe(_Client(response), "http://test", "gemma-4-e2b", 1, 0.2, 128)

    assert result["passed"] is False


def test_warm_latency_median_excludes_first_cold_transition():
    from benchmarks.gemma_speaker_memory_acceptance import _latency_metrics

    metrics = _latency_metrics([82.0, 3.1, 3.2, 3.3, 3.4])

    assert metrics["tool_latency_p50_s"] == 3.3
    assert metrics["tool_latency_warm_p50_s"] == 3.25


def test_ttft_probe_turns_connection_failure_into_failed_result():
    from benchmarks.gemma_speaker_memory_acceptance import _ttft_probe

    result = _ttft_probe(_BrokenStreamingClient(), "http://test", "gemma-4-e2b")

    assert result["passed"] is False
    assert result["http_status"] is None
    assert result["error"]["type"] == "ConnectError"


def test_ttft_probe_rejects_malformed_sse_without_crashing():
    from benchmarks.gemma_speaker_memory_acceptance import _ttft_probe

    result = _ttft_probe(_MalformedStreamingClient(), "http://test", "gemma-4-e2b")

    assert result["passed"] is False
    assert result["http_status"] == 200
    assert result["content"] == ""
