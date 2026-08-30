import json
import hashlib
import subprocess
import sys
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


def test_comparison_manifest_assigns_rows_to_explicit_cohorts_and_exclusions():
    from benchmarks.plot_context_results import classify_comparison_rows

    rows = [
        {
            "artifact": "/checkout/benchmarks/results/model/qwen-1k.json",
            "model": "Qwen3.8 Flash Next REAP-256",
            "quantization": "Q3_K_XL",
            "requested_context_tokens": 1024,
            "actual_context_tokens": 1036,
            "prefill_tps": 14.6,
            "decode_tps": 2.17,
        },
        {
            "artifact": "/checkout/benchmarks/results/legacy/short.json",
            "model": "Ornith",
            "quantization": "Q4",
            "actual_context_tokens": 16400,
            "prefill_tps": 97.8,
            "decode_tps": 21.5,
        },
    ]
    manifest = {
        "cohorts": {
            "model_context": [
                {
                    "artifact": "model/qwen-1k.json",
                    "series": "Qwen3.8 REAP-256 · Q3_K_XL",
                    "category": "1K",
                }
            ]
        },
        "excluded": [
            {
                "artifact": "legacy/short.json",
                "reason": "different completion length",
            }
        ],
    }

    classified = classify_comparison_rows(rows, manifest)

    assert classified["cohorts"]["model_context"][0]["series"] == "Qwen3.8 REAP-256 · Q3_K_XL"
    assert classified["cohorts"]["model_context"][0]["row"] is rows[0]
    assert classified["excluded"][0]["reason"] == "different completion length"


def test_comparison_manifest_rejects_unclassified_ledger_rows():
    from benchmarks.plot_context_results import classify_comparison_rows

    rows = [
        {
            "artifact": "/checkout/benchmarks/results/new/unreviewed.json",
            "model": "Ornith",
            "quantization": "Q4",
            "actual_context_tokens": 1024,
            "prefill_tps": 1.0,
            "decode_tps": 1.0,
        }
    ]

    with pytest.raises(ValueError, match="unclassified.*new/unreviewed.json"):
        classify_comparison_rows(rows, {"cohorts": {}, "excluded": []})


def test_real_comparison_manifest_accounts_for_every_ledger_row():
    from benchmarks.plot_context_results import classify_comparison_rows, load_jsonl

    root = Path(__file__).parents[2]
    rows = load_jsonl(root / "benchmarks/results/model-context-speed.jsonl")
    manifest = json.loads(
        (root / "benchmarks/comparison_cohorts.json").read_text(encoding="utf-8")
    )

    classified = classify_comparison_rows(rows, manifest)

    assert {name: len(entries) for name, entries in classified["cohorts"].items()} == {
        "model_context": 6,
        "tq3_weight_cache_ab": 6,
        "prefill_block_sweep": 8,
    }
    assert len(classified["excluded"]) == 1
    assert sum(map(len, classified["cohorts"].values())) + len(classified["excluded"]) == len(rows) == 21


def test_comparison_manifest_rejects_duplicate_assignment():
    from benchmarks.plot_context_results import classify_comparison_rows

    row = {
        "artifact": "/checkout/benchmarks/results/shared/run.json",
        "model": "Ornith",
        "quantization": "Q4",
        "actual_context_tokens": 1024,
        "prefill_tps": 1.0,
        "decode_tps": 1.0,
    }
    manifest = {
        "cohorts": {
            "first": [{"artifact": "shared/run.json"}],
            "second": [{"artifact": "shared/run.json"}],
        },
        "excluded": [],
    }

    with pytest.raises(ValueError, match="assigned more than once.*shared/run.json"):
        classify_comparison_rows([row], manifest)


def test_comparison_manifest_rejects_duplicate_ledger_artifact():
    from benchmarks.plot_context_results import classify_comparison_rows

    row = {
        "artifact": "/checkout/benchmarks/results/shared/run.json",
        "model": "Ornith",
        "quantization": "Q4",
        "actual_context_tokens": 1024,
        "prefill_tps": 1.0,
        "decode_tps": 1.0,
    }
    manifest = {
        "cohorts": {"one": [{"artifact": "shared/run.json"}]},
        "excluded": [],
    }

    with pytest.raises(ValueError, match="duplicate ledger artifact.*shared/run.json"):
        classify_comparison_rows([row, dict(row)], manifest)


def test_comparison_manifest_rejects_forged_baseline_series():
    from benchmarks.plot_context_results import classify_comparison_rows

    row = {
        "artifact": "/checkout/benchmarks/results/qwen/run.json",
        "model": "Qwen3.8 Flash Next REAP-256",
        "quantization": "Q3_K_XL",
        "requested_context_tokens": 1024,
        "actual_context_tokens": 1036,
        "prefill_tps": 14.6,
        "decode_tps": 2.17,
    }
    manifest = {
        "cohorts": {
            "model_context": [
                {
                    "artifact": "qwen/run.json",
                    "series": "Ornith 1.5 35B · Q4_K_M",
                    "category": "1K",
                }
            ]
        },
        "excluded": [],
    }

    with pytest.raises(ValueError, match="baseline series mismatch.*qwen/run.json"):
        classify_comparison_rows([row], manifest)


def test_comparison_manifest_rejects_forged_baseline_category():
    from benchmarks.plot_context_results import classify_comparison_rows

    row = {
        "artifact": "/checkout/benchmarks/results/qwen/run.json",
        "model": "Qwen3.8 Flash Next REAP-256",
        "quantization": "Q3_K_XL",
        "requested_context_tokens": 1024,
        "actual_context_tokens": 1036,
        "prefill_tps": 14.6,
        "decode_tps": 2.17,
    }
    manifest = {
        "cohorts": {
            "model_context": [
                {
                    "artifact": "qwen/run.json",
                    "series": "Qwen3.8 REAP-256 · Q3_K_XL",
                    "category": "64K",
                }
            ]
        },
        "excluded": [],
    }

    with pytest.raises(ValueError, match="baseline category mismatch.*qwen/run.json"):
        classify_comparison_rows([row], manifest)


def test_objective_comparison_svg_is_one_decode_speed_vs_context_plot():
    from benchmarks.plot_context_results import load_jsonl, render_comparison_svg

    root = Path(__file__).parents[2]
    rows = load_jsonl(root / "benchmarks/results/model-context-speed.jsonl")
    manifest = json.loads(
        (root / "benchmarks/comparison_cohorts.json").read_text(encoding="utf-8")
    )

    svg = render_comparison_svg(rows, manifest)

    assert svg.startswith("<svg")
    assert "Скорость генерации в зависимости от размера контекста" in svg
    assert 'data-x-metric="decode_tps"' in svg
    assert 'data-x-scale="linear"' in svg
    assert 'data-y-metric="actual_context_tokens"' in svg
    assert 'data-y-scale="log2"' in svg
    assert 'data-axis="context" transform="rotate(-90' in svg
    assert svg.count('data-measurement="') == 21
    assert svg.count('data-model-family="ornith"') == 18
    assert svg.count('data-model-family="qwen"') == 3
    assert svg.count('data-series-line="') == 2
    assert svg.count("<polyline") == 2
    assert "Ornith Q4_K_M · базовая линия" in svg
    assert "Qwen REAP Q3_K_XL · базовая линия" in svg
    assert "Ornith TQ3_4S · fixed 1429" in svg
    assert "Ornith TQ3_4S · auto 2633" in svg
    assert "Ornith Q4_K_M · prefill-block sweep" in svg
    assert "p1024" in svg and "p4096" in svg
    assert "точка = один сохранённый end-to-end прогон" in svg
    assert "Линия соединяет только одинаковую конфигурацию" in svg


def test_svg_coordinates_and_line_membership_match_the_real_ledger():
    import xml.etree.ElementTree as ET

    from benchmarks.plot_context_results import load_jsonl, render_comparison_svg

    root = Path(__file__).parents[2]
    rows = load_jsonl(root / "benchmarks/results/model-context-speed.jsonl")
    manifest = json.loads(
        (root / "benchmarks/comparison_cohorts.json").read_text(encoding="utf-8")
    )
    svg_root = ET.fromstring(render_comparison_svg(rows, manifest))

    marks = [element for element in svg_root.iter() if "data-measurement" in element.attrib]
    actual_points = {
        element.attrib["data-artifact"]: (
            int(element.attrib["data-context-tokens"]),
            float(element.attrib["data-decode-tps"]),
        )
        for element in marks
    }
    expected_points = {
        row["artifact"].split("benchmarks/results/", 1)[1]: (
            int(row["actual_context_tokens"]),
            float(row["decode_tps"]),
        )
        for row in rows
    }
    assert actual_points == expected_points

    lines = {
        element.attrib["data-series-line"]: set(
            element.attrib["data-series-artifacts"].split("|")
        )
        for element in svg_root.iter()
        if "data-series-line" in element.attrib
    }
    expected_ornith = {
        entry["artifact"]
        for entry in manifest["cohorts"]["model_context"]
        if entry["series"].startswith("Ornith")
    }
    expected_qwen = {
        entry["artifact"]
        for entry in manifest["cohorts"]["model_context"]
        if entry["series"].startswith("Qwen")
    }
    assert lines == {
        "Ornith Q4_K_M · базовая линия": expected_ornith,
        "Qwen REAP Q3_K_XL · базовая линия": expected_qwen,
    }

    prefill_marks = [
        element
        for element in marks
        if "prefill-block sweep" in element.attrib["data-measurement"]
    ]
    assert len(prefill_marks) == 8
    assert len({element.attrib["data-marker-shape"] for element in prefill_marks}) >= 4
    assert len({element.attrib["data-marker-color"] for element in prefill_marks}) >= 4


def test_registry_cli_renders_the_explicit_comparison_manifest(tmp_path: Path):
    root = Path(__file__).parents[2]
    output = tmp_path / "comparison.svg"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/plot_context_results.py"),
            "--registry",
            str(root / "benchmarks/results/model-context-speed.jsonl"),
            "--comparison-manifest",
            str(root / "benchmarks/comparison_cohorts.json"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    svg = output.read_text(encoding="utf-8")
    assert "Скорость генерации в зависимости от размера контекста" in svg
    assert svg.count('data-measurement="') == 21
    assert svg.count('data-series-line="') == 2


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
