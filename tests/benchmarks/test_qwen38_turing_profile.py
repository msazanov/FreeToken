from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


# The shared benchmark venv is editable against a sibling checkout. Exercise
# this task's source tree even when pytest is invoked through that interpreter.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def test_aggregate_trace_keeps_only_phase_layer_counters():
    from freetoken.moe.offload_cache import AggregateRouteCopyTrace

    trace = AggregateRouteCopyTrace(num_layers=2)
    trace.record_route(
        phase="prefill",
        layer_id=1,
        expert_ids=(41, 7, 41, 9),
        l1_hits=1,
        l1_misses=2,
        evictions=1,
    )
    trace.record_copy(phase="prefill", layer_id=1, records=3, nbytes=768)
    trace.record_route(
        phase="decode",
        layer_id=0,
        expert_ids=(5, 5),
        l1_hits=1,
        l1_misses=0,
        evictions=0,
    )

    snapshot = trace.snapshot()

    assert snapshot == {
        "prefill": {
            "layers": [
                {
                    "layer": 0,
                    "route_references": 0,
                    "route_unique": 0,
                    "l1_hits": 0,
                    "l1_misses": 0,
                    "copy_records": 0,
                    "copy_bytes": 0,
                    "evictions": 0,
                },
                {
                    "layer": 1,
                    "route_references": 4,
                    "route_unique": 3,
                    "l1_hits": 1,
                    "l1_misses": 2,
                    "copy_records": 3,
                    "copy_bytes": 768,
                    "evictions": 1,
                },
            ]
        },
        "decode": {
            "layers": [
                {
                    "layer": 0,
                    "route_references": 2,
                    "route_unique": 1,
                    "l1_hits": 1,
                    "l1_misses": 0,
                    "copy_records": 0,
                    "copy_bytes": 0,
                    "evictions": 0,
                },
                {
                    "layer": 1,
                    "route_references": 0,
                    "route_unique": 0,
                    "l1_hits": 0,
                    "l1_misses": 0,
                    "copy_records": 0,
                    "copy_bytes": 0,
                    "evictions": 0,
                },
            ]
        },
    }
    encoded = json.dumps(snapshot, sort_keys=True)
    assert "expert" not in encoded
    assert "41" not in encoded


def test_process_counter_delta_parses_proc_records_and_preserves_missing_values():
    from benchmarks.qwen38_turing_profile import parse_proc_counters, process_counter_delta

    before = parse_proc_counters(
        "rchar: 10\nread_bytes: 1024\n",
        "123 (freetoken) R 0 0 0 0 0 0 11 0 7 0 0\n",
    )
    after = parse_proc_counters(
        "rchar: 20\nread_bytes: 4096\n",
        "123 (freetoken) R 0 0 0 0 0 0 15 0 10 0 0\n",
    )

    assert process_counter_delta(before, after) == {
        "io_read_bytes": 3072,
        "major_faults": 3,
        "minor_faults": 4,
    }
    assert process_counter_delta(before, None) == {
        "io_read_bytes": None,
        "major_faults": None,
        "minor_faults": None,
    }


def test_proc_and_gpu_samples_include_rss_io_power_and_temperature():
    from benchmarks.qwen38_turing_profile import parse_gpu_sample, parse_proc_counters

    counters = parse_proc_counters(
        "read_bytes: 4096\n",
        "123 (freetoken) R 0 0 0 0 0 0 15 0 10 0 0\n",
        "Name:\tfreetoken\nVmRSS:\t123456 kB\n",
    )

    assert counters["rss_kib"] == 123456
    assert parse_gpu_sample("97, 7048, 84.50, 86") == {
        "utilization_percent": 97.0,
        "memory_used_mib": 7048.0,
        "power_w": 84.5,
        "temperature_c": 86.0,
    }


def test_stream_completion_hashes_final_content_only_and_ignores_reasoning(monkeypatch):
    from benchmarks import qwen38_turing_profile as profile

    events = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"reasoning_content": "дум"}}]},
        {"choices": [{"delta": {"content": "ответ"}}]},
        {"choices": [{"delta": {"reasoning_content": "!"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    ]
    stream = [
        (f"data: {json.dumps(event, ensure_ascii=False)}\n").encode("utf-8")
        for event in events
    ] + [b"data: [DONE]\n"]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(stream)

    monkeypatch.setattr(profile.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    metrics = profile._stream_completion(
        "http://127.0.0.1:1",
        {"stream": True},
        timeout_s=1.0,
    )

    assert metrics["response_sha256"] == hashlib.sha256("ответ".encode("utf-8")).hexdigest()
    assert metrics["prompt_tokens"] == 12
    assert metrics["completion_tokens"] == 3
    assert "response_text" not in metrics


def test_stream_completion_keeps_timed_delta_counts_without_content(monkeypatch):
    """A long run records progress, but never the prompt, answer, or reasoning."""
    from benchmarks import qwen38_turing_profile as profile

    events = [
        {"choices": [{"delta": {"reasoning_content": "private chain"}}]},
        {"choices": [{"delta": {"content": "private answer"}}]},
        {"choices": [{"delta": {"content": "more private answer"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 16, "completion_tokens": 3}},
    ]
    stream = [
        (f"data: {json.dumps(event)}\n").encode("utf-8") for event in events
    ] + [b"data: [DONE]\n"]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(stream)

    ticks = iter((100.0, 101.0, 102.0, 103.0, 104.0, 105.0))
    monkeypatch.setattr(profile.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(profile.time, "monotonic", lambda: next(ticks))

    metrics = profile._stream_completion(
        "http://127.0.0.1:1", {"stream": True}, timeout_s=1.0, trace_stride_events=2
    )

    assert metrics["first_token_elapsed_s"] == 1.0
    assert metrics["generation_events"] == [
        {
            "elapsed_s": 2.0,
            "delta_events": 2,
            "content_delta_events": 1,
            "reasoning_delta_events": 1,
        },
        {
            "elapsed_s": 4.0,
            "delta_events": 3,
            "content_delta_events": 2,
            "reasoning_delta_events": 1,
            "completion_tokens": 3,
            "terminal": True,
        },
    ]
    assert "private" not in json.dumps(metrics, ensure_ascii=False)


def test_phase_samples_mark_prefill_and_decode_without_clock_leakage():
    from benchmarks.qwen38_turing_profile import phase_samples

    annotated = phase_samples(
        [
            {"monotonic_s": 9.0, "utilization_percent": 10.0},
            {"monotonic_s": 11.5, "utilization_percent": 99.0},
        ],
        request_started_at=10.0,
        first_token_at=11.0,
    )

    assert annotated == [
        {"elapsed_s": 0.0, "phase": "prefill", "utilization_percent": 10.0},
        {"elapsed_s": 1.5, "phase": "decode", "utilization_percent": 99.0},
    ]


def test_parse_args_exposes_forced_generation_and_trace_controls():
    from benchmarks.qwen38_turing_profile import parse_args

    args = parse_args(["--model-path", "/models/test.gguf", "--ignore-eos", "--trace-stride-events", "64"])

    assert args.ignore_eos is True
    assert args.trace_stride_events == 64


def test_stream_completion_raises_on_sse_error(monkeypatch):
    from benchmarks import qwen38_turing_profile as profile

    stream = [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
        b'data: {"error":{"message":"generation failed","code":"backend_error"}}\n',
        b"data: [DONE]\n",
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(stream)

    monkeypatch.setattr(profile.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="generation failed"):
        profile._stream_completion("http://127.0.0.1:1", {"stream": True}, timeout_s=1.0)


def test_stream_completion_raises_when_done_sentinel_is_missing(monkeypatch):
    from benchmarks import qwen38_turing_profile as profile

    stream = [
        b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"stop"}]}\n',
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(stream)

    monkeypatch.setattr(profile.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match=r"\[DONE\]"):
        profile._stream_completion("http://127.0.0.1:1", {"stream": True}, timeout_s=1.0)


def test_stream_completion_raises_when_terminal_finish_reason_is_missing(monkeypatch):
    from benchmarks import qwen38_turing_profile as profile

    stream = [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
        b"data: [DONE]\n",
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(stream)

    monkeypatch.setattr(profile.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="terminal finish_reason"):
        profile._stream_completion("http://127.0.0.1:1", {"stream": True}, timeout_s=1.0)


def test_result_record_promotes_digest_without_retaining_response_body(tmp_path):
    from benchmarks.qwen38_turing_profile import make_result_record

    digest = hashlib.sha256(b"fixed output").hexdigest()
    record = make_result_record(
        context_tokens=1024,
        seed=20260828,
        port=41234,
        pid=123,
        metrics={
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "response_sha256": digest,
        },
        process={"io_read_bytes": 0, "major_faults": 0, "minor_faults": 0},
        gpu_samples=[],
        server_stats={"moe": {"cache": {"policy": "protected_layer"}}},
        artifact=tmp_path / "context-1024.json",
    )

    assert record["response_sha256"] == digest
    assert record["server_stats"]["moe"]["cache"]["policy"] == "protected_layer"
    assert "response_text" not in record
    assert "prompt" not in record


def test_child_environment_disables_parent_request_body_logging(tmp_path):
    from benchmarks.qwen38_turing_profile import child_environment

    environment = child_environment(
        tmp_path,
        inherited={"FREETOKEN_API_LOG_DIR": "/unsafe/request-logs", "KEEP": "yes"},
    )

    assert "FREETOKEN_API_LOG_DIR" not in environment
    assert environment["KEEP"] == "yes"
    assert environment["PYTHONPATH"].split(":")[0] == str(tmp_path / "python")


def test_wait_for_server_requires_exact_ok_health_status(monkeypatch):
    from benchmarks import qwen38_turing_profile as profile

    class RunningProcess:
        returncode = None

        def poll(self):
            return None

    payloads = iter(
        (
            {"status": "loading", "phase": "weights"},
            {"status": "OK"},
            {"status": "ok"},
        )
    )
    calls: list[tuple[str, str, float]] = []
    sleeps: list[float] = []

    def get_json(origin: str, endpoint: str, timeout_s: float):
        calls.append((origin, endpoint, timeout_s))
        assert endpoint == "/health"
        return next(payloads)

    monkeypatch.setattr(profile, "_get_json", get_json)
    monkeypatch.setattr(profile.time, "sleep", sleeps.append)

    profile._wait_for_server("http://127.0.0.1:41234", RunningProcess(), timeout_s=1.0)

    assert calls == [
        ("http://127.0.0.1:41234", "/health", 2),
        ("http://127.0.0.1:41234", "/health", 2),
        ("http://127.0.0.1:41234", "/health", 2),
    ]
    assert sleeps == [0.25, 0.25]


def test_wait_for_server_reports_health_error_payload(monkeypatch):
    from benchmarks import qwen38_turing_profile as profile

    class RunningProcess:
        returncode = None

        def poll(self):
            return None

    times = iter((0.0, 0.0, 2.0))

    def get_json(_origin: str, endpoint: str, timeout_s: float):
        assert endpoint == "/health"
        return {"status": "error", "message": "backend worker exited"}

    monkeypatch.setattr(profile, "_get_json", get_json)
    monkeypatch.setattr(profile.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(profile.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError) as raised:
        profile._wait_for_server("http://127.0.0.1:41234", RunningProcess(), timeout_s=1.0)

    assert "Qwen health reported startup error" in str(raised.value)
    assert "backend worker exited" in str(raised.value)


def test_wait_for_server_reports_owned_child_exit_before_health_probe(monkeypatch):
    from benchmarks import qwen38_turing_profile as profile

    class ExitedProcess:
        returncode = 47

        def poll(self):
            return self.returncode

    calls: list[tuple[str, str, float]] = []

    def get_json(origin: str, endpoint: str, timeout_s: float):
        calls.append((origin, endpoint, timeout_s))
        return {"status": "ok"}

    monkeypatch.setattr(profile, "_get_json", get_json)

    with pytest.raises(RuntimeError, match=r"Qwen process exited during startup with 47"):
        profile._wait_for_server("http://127.0.0.1:41234", ExitedProcess(), timeout_s=1.0)

    assert calls == []
