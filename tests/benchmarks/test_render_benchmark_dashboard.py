from __future__ import annotations

import json


def test_dashboard_keeps_every_success_and_failure_from_ledger(tmp_path):
    from benchmarks.render_benchmark_dashboard import render_dashboard

    ledger = tmp_path / "benchmark-events.jsonl"
    output = tmp_path / "dashboard.html"
    rows = [
        {"event_id": "artifact:a", "status": "success", "model": "Ornith", "requested_context_tokens": 16384, "prefill_tps": 98.0, "decode_tps": 21.5, "parameters": {"max_prefill_length": 1024}},
        {"event_id": "attempt:b", "status": "oom", "model": "Ornith", "requested_context_tokens": 16384, "parameters": {"max_prefill_length": 2048}, "error": "CUDA out of memory"},
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    render_dashboard(ledger, output)

    page = output.read_text(encoding="utf-8")
    assert "artifact:a" in page
    assert "attempt:b" in page
    assert "max_prefill_length" in page
    assert "CUDA out of memory" in page
