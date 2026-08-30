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


def test_live_decode_extractor_uses_every_stable_stdout_sample(tmp_path: Path):
    from benchmarks.plot_context_results import extract_live_decode_samples

    artifact = tmp_path / "context-16384.json"
    artifact.write_text("{}\n", encoding="utf-8")
    artifact.with_suffix(".stdout.log").write_text(
        "Decode batch, #running-req: 1, #token: 16441, "
        "gen throughput (token/s): 0.21, #queue-req: 0\n"
        "Decode batch, #running-req: 1, #token: 16481, "
        "gen throughput (token/s): 20.61, #queue-req: 0\n"
        "Decode batch, #running-req: 1, #token: 16521, "
        "gen throughput (token/s): 21.34, #queue-req: 0\n",
        encoding="utf-8",
    )

    samples = extract_live_decode_samples(artifact)

    assert samples == [
        {
            "context_tokens": 16481,
            "decode_tps": 20.61,
            "completion_tokens": None,
            "source_kind": "server_stdout",
            "source_line": 2,
        },
        {
            "context_tokens": 16521,
            "decode_tps": 21.34,
            "completion_tokens": None,
            "source_kind": "server_stdout",
            "source_line": 3,
        },
    ]


def test_live_decode_extractor_uses_stable_runtime_stats_samples(tmp_path: Path):
    from benchmarks.plot_context_results import extract_live_decode_samples

    artifact = tmp_path / "compression-1024.json"
    artifact.write_text(
        json.dumps(
            {
                "runtime_samples": [
                    {
                        "server_stats": {
                            "kv": {"used_pages": 1013, "page_size": 4},
                            "throughput": {"decode_tps": 341.4},
                            "requests": {"completion_tokens_total": 1},
                        }
                    },
                    {
                        "server_stats": {
                            "kv": {"used_pages": 1043, "page_size": 4},
                            "throughput": {"decode_tps": 29.8},
                            "requests": {"completion_tokens_total": 31},
                        }
                    },
                    {
                        "server_stats": {
                            "kv": {"used_pages": 1074, "page_size": 4},
                            "throughput": {"decode_tps": 29.9},
                            "requests": {"completion_tokens_total": 62},
                        }
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = extract_live_decode_samples(artifact)

    assert samples == [
        {
            "context_tokens": 4172,
            "decode_tps": 29.8,
            "completion_tokens": 31,
            "source_kind": "runtime_stats",
            "source_line": 2,
        },
        {
            "context_tokens": 4296,
            "decode_tps": 29.9,
            "completion_tokens": 62,
            "source_kind": "runtime_stats",
            "source_line": 3,
        },
    ]


def test_real_live_decode_sources_cover_every_run_and_all_stable_samples():
    from benchmarks.plot_context_results import (
        collect_live_comparison_runs,
        load_jsonl,
        load_live_jsonl,
    )

    root = Path(__file__).parents[2]
    results_root = root / "benchmarks/results"
    portable = load_live_jsonl(results_root / "model-context-speed-live.jsonl")
    if not all(
        (results_root / row["source_artifact"]).exists()
        for row in portable
    ):
        pytest.skip("primary stdout logs are local evidence ignored by Git")
    rows = load_jsonl(root / "benchmarks/results/model-context-speed.jsonl")
    manifest = json.loads(
        (root / "benchmarks/comparison_cohorts.json").read_text(encoding="utf-8")
    )

    runs = collect_live_comparison_runs(
        rows,
        manifest,
        results_root=results_root,
    )

    assert len(runs) == 21
    assert sum(len(run["samples"]) for run in runs) == 900
    assert all(run["samples"] for run in runs)
    assert {
        run["samples"][0]["source_kind"] for run in runs
    } == {"server_stdout", "runtime_stats"}
    assert {
        len(run["samples"])
        for run in runs
        if run["cohort"] == "prefill_block_sweep"
    } == {101}


def test_checked_in_live_registry_exactly_matches_raw_live_sources():
    from benchmarks.plot_context_results import (
        collect_live_comparison_runs,
        flatten_live_comparison_runs,
        load_jsonl,
        load_live_jsonl,
    )

    root = Path(__file__).parents[2]
    results_root = root / "benchmarks/results"
    rows = load_jsonl(results_root / "model-context-speed.jsonl")
    portable = load_live_jsonl(results_root / "model-context-speed-live.jsonl")
    if not all(
        (results_root / row["source_artifact"]).exists()
        for row in portable
    ):
        pytest.skip("primary stdout logs are local evidence ignored by Git")
    manifest = json.loads(
        (root / "benchmarks/comparison_cohorts.json").read_text(encoding="utf-8")
    )
    raw_runs = collect_live_comparison_runs(
        rows,
        manifest,
        results_root=results_root,
    )

    assert portable == flatten_live_comparison_runs(raw_runs)


def test_live_run_resolution_rebases_stale_absolute_artifact_paths(tmp_path: Path):
    from benchmarks.plot_context_results import collect_live_comparison_runs

    results_root = tmp_path / "benchmarks/results"
    artifact = results_root / "model/run.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    artifact.with_suffix(".stdout.log").write_text(
        "Decode batch, #token: 1001, gen throughput (token/s): 0.1\n"
        "Decode batch, #token: 1041, gen throughput (token/s): 20.0\n",
        encoding="utf-8",
    )
    stale_artifact = tmp_path / "old/checkout/benchmarks/results/model/run.json"
    stale_artifact.parent.mkdir(parents=True)
    stale_artifact.write_text("{}\n", encoding="utf-8")
    stale_artifact.with_suffix(".stdout.log").write_text(
        "Decode batch, #token: 1001, gen throughput (token/s): 0.1\n"
        "Decode batch, #token: 1041, gen throughput (token/s): 1.0\n",
        encoding="utf-8",
    )
    row = {
        "artifact": str(stale_artifact),
        "model": "Ornith",
        "quantization": "Q4",
        "actual_context_tokens": 1000,
        "prefill_tps": 1.0,
        "decode_tps": 20.0,
    }
    manifest = {
        "cohorts": {"custom": [{"artifact": "model/run.json"}]},
        "excluded": [],
    }

    runs = collect_live_comparison_runs(
        [row],
        manifest,
        results_root=results_root,
    )

    assert runs[0]["artifact_path"] == artifact
    assert runs[0]["samples"][0]["context_tokens"] == 1041
    assert runs[0]["samples"][0]["decode_tps"] == 20.0


def test_live_runs_can_use_portable_registry_without_companion_logs(tmp_path: Path):
    from benchmarks.plot_context_results import collect_live_comparison_runs

    results_root = tmp_path / "results"
    artifact = results_root / "model/run.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    row = {
        "artifact": "model/run.json",
        "model": "Ornith",
        "quantization": "Q4",
        "actual_context_tokens": 1000,
        "prefill_tps": 1.0,
        "decode_tps": 20.0,
    }
    manifest = {
        "cohorts": {"custom": [{"artifact": "model/run.json"}]},
        "excluded": [],
    }
    live_rows = [
        {
            "schema_version": 1,
            "artifact": "model/run.json",
            "cohort": "custom",
            "model": "Ornith",
            "quantization": "Q4",
            "sample_index": 0,
            "context_tokens": 1041,
            "decode_tps": 20.0,
            "completion_tokens": None,
            "source_kind": "server_stdout",
            "source_artifact": "model/run.stdout.log",
            "source_line": 42,
            "source_sha256": "0" * 64,
        }
    ]

    runs = collect_live_comparison_runs(
        [row],
        manifest,
        results_root=results_root,
        live_rows=live_rows,
    )

    assert runs[0]["samples"] == [
        {
            "context_tokens": 1041,
            "decode_tps": 20.0,
            "completion_tokens": None,
            "source_kind": "server_stdout",
            "source_line": 42,
        }
    ]


def test_objective_comparison_svg_is_one_decode_speed_vs_context_plot():
    from benchmarks.plot_context_results import (
        load_jsonl,
        load_live_jsonl,
        render_comparison_svg,
    )

    root = Path(__file__).parents[2]
    rows = load_jsonl(root / "benchmarks/results/model-context-speed.jsonl")
    manifest = json.loads(
        (root / "benchmarks/comparison_cohorts.json").read_text(encoding="utf-8")
    )

    svg = render_comparison_svg(
        rows,
        manifest,
        results_root=root / "benchmarks/results",
        live_rows=load_live_jsonl(
            root / "benchmarks/results/model-context-speed-live.jsonl"
        ),
    )

    assert svg.startswith("<svg")
    assert "Живая скорость decode по мере роста KV-контекста" in svg
    assert 'data-x-metric="decode_tps"' in svg
    assert 'data-x-scale="linear"' in svg
    assert 'data-y-metric="live_context_tokens"' in svg
    assert 'data-y-scale="log2"' in svg
    assert 'data-axis="context" transform="rotate(-90' in svg
    assert svg.count('data-live-measurement="') == 900
    assert svg.count('data-model-family="ornith"') == 885
    assert svg.count('data-model-family="qwen"') == 15
    assert svg.count('data-run-trace="') == 21
    assert svg.count('data-baseline-summary-line="') == 0
    assert svg.count('data-summary-measurement="') == 0
    assert svg.count('data-prefill-legend="') == 8
    assert "128K</text>" not in svg
    assert "64K</text>" in svg
    assert "Ornith Q4_K_M · live" in svg
    assert "Qwen REAP Q3_K_XL · live" in svg
    assert "Ornith TQ3_4S · fixed 1429" in svg
    assert "Ornith TQ3_4S · auto 2633" in svg
    assert "Ornith Q4_K_M · prefill-block sweep" in svg
    assert "p1024" in svg and "p4096" in svg
    assert "900 стабильных живых decode-срезов" in svg
    assert "Первый переходный интервал после prefill исключён" in svg


def test_svg_live_coordinates_and_trace_membership_match_raw_sources():
    import xml.etree.ElementTree as ET

    from benchmarks.plot_context_results import (
        collect_live_comparison_runs,
        load_jsonl,
        load_live_jsonl,
        render_comparison_svg,
    )

    root = Path(__file__).parents[2]
    rows = load_jsonl(root / "benchmarks/results/model-context-speed.jsonl")
    manifest = json.loads(
        (root / "benchmarks/comparison_cohorts.json").read_text(encoding="utf-8")
    )
    results_root = root / "benchmarks/results"
    live_rows = load_live_jsonl(results_root / "model-context-speed-live.jsonl")
    runs = collect_live_comparison_runs(
        rows,
        manifest,
        results_root=results_root,
        live_rows=live_rows,
    )
    svg_root = ET.fromstring(
        render_comparison_svg(
            rows,
            manifest,
            results_root=results_root,
            live_rows=live_rows,
        )
    )

    marks = [
        element
        for element in svg_root.iter()
        if "data-live-measurement" in element.attrib
    ]
    actual_points = {
        (
            element.attrib["data-artifact"],
            element.attrib["data-source-kind"],
            int(element.attrib["data-source-line"]),
        ): (
            int(element.attrib["data-context-tokens"]),
            float(element.attrib["data-decode-tps"]),
        )
        for element in marks
    }
    expected_points = {
        (
            run["artifact"],
            sample["source_kind"],
            int(sample["source_line"]),
        ): (
            int(sample["context_tokens"]),
            float(sample["decode_tps"]),
        )
        for run in runs
        for sample in run["samples"]
    }
    assert actual_points == expected_points

    traces = {
        element.attrib["data-run-trace"]: int(element.attrib["data-live-samples"])
        for element in svg_root.iter()
        if "data-run-trace" in element.attrib
    }
    assert traces == {run["artifact"]: len(run["samples"]) for run in runs}

    assert not any(
        "data-baseline-summary-line" in element.attrib
        or "data-summary-measurement" in element.attrib
        for element in svg_root.iter()
    )

    prefill_marks = [
        element
        for element in marks
        if "prefill-block sweep" in element.attrib["data-live-measurement"]
    ]
    assert len(prefill_marks) == 8 * 101
    assert len({element.attrib["data-marker-shape"] for element in prefill_marks}) >= 4
    assert len({element.attrib["data-marker-color"] for element in prefill_marks}) == 8


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
    assert "Живая скорость decode по мере роста KV-контекста" in svg
    assert svg.count('data-live-measurement="') == 900
    assert svg.count('data-run-trace="') == 21
    assert svg.count('data-baseline-summary-line="') == 0
    assert svg.count('data-summary-measurement="') == 0


def test_registry_cli_uses_portable_live_ledger_when_raw_log_is_absent(
    tmp_path: Path,
):
    root = Path(__file__).parents[2]
    results_root = tmp_path / "results"
    artifact = results_root / "model/run.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    registry = results_root / "model-context-speed.jsonl"
    registry.write_text(
        json.dumps(
            {
                "artifact": "/stale/checkout/benchmarks/results/model/run.json",
                "model": "Ornith",
                "quantization": "Q4",
                "requested_context_tokens": 1024,
                "actual_context_tokens": 1024,
                "completion_tokens": 64,
                "prefill_tps": 100.0,
                "decode_tps": 20.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (results_root / "model-context-speed-live.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "model/run.json",
                "cohort": "custom",
                "model": "Ornith",
                "quantization": "Q4",
                "sample_index": 0,
                "context_tokens": 1041,
                "decode_tps": 20.0,
                "completion_tokens": None,
                "source_kind": "server_stdout",
                "source_artifact": "model/run.stdout.log",
                "source_line": 42,
                "source_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cohorts": {"custom": [{"artifact": "model/run.json"}]},
                "excluded": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "portable.svg"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/plot_context_results.py"),
            "--registry",
            str(registry),
            "--comparison-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text(encoding="utf-8").count(
        'data-live-measurement="'
    ) == 1


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
