from __future__ import annotations

import json
from pathlib import Path


def test_normalize_legacy_artifact_keeps_speed_without_response_text(tmp_path):
    from benchmarks.benchmark_ledger import normalize_artifact

    artifact = tmp_path / "compression-16384.json"
    artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-27T13:01:08+00:00",
                "requested_context_tokens": 16_384,
                "model": "Ornith 1.5 35b",
                "metrics": {
                    "prompt_tokens": 16_373,
                    "completion_tokens": 255,
                    "prefill_tps_estimate": 121.95,
                    "decode_tps": 20.56,
                    "response_text": "private prompt output must not be copied",
                },
            }
        ),
        encoding="utf-8",
    )

    event = normalize_artifact(artifact)

    assert event["status"] == "success"
    assert event["model"] == "Ornith 1.5 35b"
    assert event["actual_context_tokens"] == 16_373
    assert event["prefill_tps"] == 121.95
    assert event["decode_tps"] == 20.56
    assert "response_text" not in json.dumps(event)


def test_normalize_live_artifact_promotes_first_token_elapsed_as_ttft(tmp_path):
    from benchmarks.benchmark_ledger import normalize_artifact

    artifact = tmp_path / "context-16384.json"
    artifact.write_text(
        json.dumps(
            {
                "requested_context_tokens": 16_384,
                "metrics": {
                    "prompt_tokens": 16_400,
                    "first_token_elapsed_s": 148.35,
                    "prompt_tps": 110.55,
                    "decode_tps": 19.66,
                },
            }
        ),
        encoding="utf-8",
    )

    assert normalize_artifact(artifact)["ttft_s"] == 148.35


def test_append_event_preserves_success_and_failure_once(tmp_path):
    from benchmarks.benchmark_ledger import append_event, make_failure_event

    ledger = tmp_path / "benchmark-events.jsonl"
    success = {
        "event_id": "artifact:/result/a.json",
        "status": "success",
        "artifact": "/result/a.json",
    }
    failure = make_failure_event(
        attempt_id="prefill-2048",
        model="Ornith 1.5 35b",
        requested_context_tokens=16_384,
        parameters={"max_prefill_length": 2048},
        failure_kind="oom",
        error="CUDA out of memory",
        stdout_log="/result/stdout.log",
        stderr_log="/result/stderr.log",
    )

    assert append_event(ledger, success) is True
    assert append_event(ledger, success) is False
    assert append_event(ledger, failure) is True
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["success", "oom"]
    assert events[1]["parameters"] == {"max_prefill_length": 2048}


def test_classify_failure_distinguishes_oom_from_startup_error():
    from benchmarks.benchmark_ledger import classify_failure

    assert classify_failure("RuntimeError: CUDA out of memory") == "oom"
    assert classify_failure("ModuleNotFoundError: benchmarks") == "startup_error"
    assert classify_failure("usage: runner --startup-timeout-s 900\nargument --server-arg: expected one argument") == "startup_error"


def test_parse_chunk_values_requires_ascending_unique_positive_values():
    from benchmarks.ornith_prefill_sweep import parse_chunk_values

    assert parse_chunk_values("1024,1280,2048") == (1024, 1280, 2048)
    try:
        parse_chunk_values("1024,1024")
    except ValueError as exc:
        assert "strictly ascending" in str(exc)
    else:
        raise AssertionError("duplicate chunks must be rejected")
