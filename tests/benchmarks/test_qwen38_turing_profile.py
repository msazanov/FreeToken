from __future__ import annotations

import json
import sys
from pathlib import Path


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
