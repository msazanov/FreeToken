import json
import hashlib
from pathlib import Path

import pytest


def test_load_jsonl_preserves_every_valid_measurement(tmp_path: Path):
    from benchmarks.plot_context_results import load_jsonl

    registry = tmp_path / "measurements.jsonl"
    rows = [
        {
            "model": "Ornith 1.5 35b",
            "quantization": "Q4_K_M",
            "actual_context_tokens": 1012,
            "prefill_tps": 130.4,
            "decode_tps": 25.3,
            "artifact": "q4.json",
        },
        {
            "model": "Ornith 1.5 35b",
            "quantization": "TQ3_4S",
            "actual_context_tokens": 1012,
            "prefill_tps": 161.9,
            "decode_tps": 35.1,
            "artifact": "tq3.json",
        },
    ]
    registry.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert load_jsonl(registry) == rows


def test_context_svg_contains_both_metrics_and_every_series():
    from benchmarks.plot_context_results import render_context_svg

    svg = render_context_svg(
        [
            {
                "model": "Ornith 1.5 35b",
                "quantization": "Q4_K_M",
                "actual_context_tokens": 1012,
                "prefill_tps": 130.4,
                "decode_tps": 25.3,
                "runtime_profile": "fixed1429",
            },
            {
                "model": "Ornith 1.5 35b",
                "quantization": "TQ3_4S",
                "actual_context_tokens": 1012,
                "prefill_tps": 161.9,
                "decode_tps": 35.1,
                "runtime_profile": "auto2633",
            },
        ]
    )

    assert svg.startswith("<svg")
    assert "Prefill, tok/s" in svg
    assert "Decode, tok/s" in svg
    assert "Q4_K_M" in svg
    assert "TQ3_4S" in svg
    assert svg.count("data-point=") == 4


def test_context_ticks_merge_near_identical_actual_token_counts():
    from benchmarks.plot_context_results import context_ticks

    ticks = context_ticks([1012, 1036, 1040, 16396, 16400, 65548])

    assert len(ticks) == 3
    assert ticks[0] == pytest.approx((1012 + 1036 + 1040) / 3)
    assert ticks[1] == pytest.approx((16396 + 16400) / 2)


def test_weight_ab_svg_labels_slot_count_and_transfer_pressure():
    from benchmarks.plot_context_results import render_weight_ab_svg

    svg = render_weight_ab_svg(
        {
            "runs": [
                {
                    "label": "Q4 fixed",
                    "slots": 1429,
                    "prefill_tps": 130.4,
                    "decode_tps": 25.3,
                    "decode_miss_rate": 0.444,
                    "decode_transfer_bytes_per_output_token": 271_700_000,
                },
                {
                    "label": "TQ3 auto",
                    "slots": 2633,
                    "prefill_tps": 161.6,
                    "decode_tps": 35.1,
                    "decode_miss_rate": 0.276,
                    "decode_transfer_bytes_per_output_token": 139_000_000,
                },
            ]
        }
    )

    assert svg.startswith("<svg")
    assert "2633 slots" in svg
    assert "Cache miss, %" in svg
    assert "Transfer, MB/output token" in svg
    assert svg.count("data-point=") == 8


def test_task7_summary_matches_raw_artifacts_and_reported_deltas():
    root = Path(__file__).parents[2]
    summary_path = (
        root
        / "benchmarks/results/ornith35-tq3-weight-ab-task7-v2-system/summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = {run["label"]: run for run in summary["runs"]}

    for run in runs.values():
        raw = json.loads((summary_path.parent / run["artifact"]).read_text(encoding="utf-8"))
        decode_copy = sum(
            layer["copy_bytes"]
            for layer in raw["runtime_after"]["stats"]["moe"]["trace"]["decode"]["layers"]
        )
        prefill_copy = sum(
            layer["copy_bytes"]
            for layer in raw["runtime_after"]["stats"]["moe"]["trace"]["prefill"]["layers"]
        )
        response_sha = hashlib.sha256(raw["metrics"]["response_text"].encode()).hexdigest()

        assert run["decode_tps"] == raw["metrics"]["decode_tps"]
        assert run["prefill_tps"] == raw["metrics"]["prefill_tps_estimate"]
        assert run["decode_copy_bytes"] == decode_copy
        assert run["prefill_copy_bytes"] == prefill_copy
        assert run["total_copy_bytes"] == decode_copy + prefill_copy
        assert run["response_sha256"] == response_sha

    q4 = runs["Q4_K_M fixed"]
    fixed = runs["TQ3_4S fixed"]
    auto = runs["TQ3_4S auto"]
    comparisons = summary["comparisons_percent"]
    assert comparisons["tq3_fixed_vs_q4"]["decode_tps"] == pytest.approx(
        100 * (fixed["decode_tps"] / q4["decode_tps"] - 1), abs=1e-6
    )
    assert comparisons["tq3_auto_vs_tq3_fixed"]["decode_tps"] == pytest.approx(
        100 * (auto["decode_tps"] / fixed["decode_tps"] - 1), abs=1e-6
    )
    assert comparisons["tq3_auto_vs_q4"]["decode_tps"] == pytest.approx(
        100 * (auto["decode_tps"] / q4["decode_tps"] - 1), abs=1e-6
    )
    assert comparisons["tq3_fixed_vs_q4"]["ttft"] == pytest.approx(
        100 * (fixed["ttft_s"] / q4["ttft_s"] - 1), abs=1e-6
    )
    assert comparisons["tq3_auto_vs_tq3_fixed"]["ttft"] == pytest.approx(
        100 * (auto["ttft_s"] / fixed["ttft_s"] - 1), abs=1e-6
    )
    assert comparisons["tq3_auto_vs_q4"]["ttft"] == pytest.approx(
        100 * (auto["ttft_s"] / q4["ttft_s"] - 1), abs=1e-6
    )
