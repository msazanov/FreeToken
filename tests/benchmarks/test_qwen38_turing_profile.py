from __future__ import annotations

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
