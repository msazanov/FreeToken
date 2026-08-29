"""Append-only catalogue for every benchmark observation, including failures."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _model_name(record: dict[str, Any]) -> str:
    model = record.get("model")
    if isinstance(model, str):
        return model
    if isinstance(model, dict) and isinstance(model.get("id"), str):
        return model["id"]
    server_model = ((record.get("server_stats") or {}).get("model") or {})
    if isinstance(server_model.get("id"), str):
        return server_model["id"]
    return "unknown"


def _gpu_summary(record: dict[str, Any]) -> dict[str, float | None]:
    samples = record.get("gpu_samples") or record.get("runtime_samples") or []
    if not isinstance(samples, list):
        samples = []
    utils = [sample.get("utilization_percent") for sample in samples if isinstance(sample, dict)]
    memory = [sample.get("memory_used_mib") for sample in samples if isinstance(sample, dict)]
    numeric_utils = [float(value) for value in utils if isinstance(value, (int, float))]
    numeric_memory = [float(value) for value in memory if isinstance(value, (int, float))]
    return {
        "gpu_util_mean_percent": None if not numeric_utils else sum(numeric_utils) / len(numeric_utils),
        "gpu_util_peak_percent": None if not numeric_utils else max(numeric_utils),
        "gpu_vram_peak_mib": None if not numeric_memory else max(numeric_memory),
    }


def normalize_artifact(artifact: Path) -> dict[str, Any]:
    """Reduce any historical result JSON to prompt-private graphable fields."""
    artifact = artifact.resolve()
    record = json.loads(artifact.read_text(encoding="utf-8"))
    metrics = record.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "event_id": f"artifact:{artifact}",
        "status": "success",
        "timestamp_utc": record.get("timestamp_utc"),
        "artifact": str(artifact),
        "model": _model_name(record),
        "requested_context_tokens": record.get("requested_context_tokens"),
        "actual_context_tokens": _first(metrics, "prompt_tokens", "context_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "ttft_s": metrics.get("ttft_s"),
        "prefill_tps": _first(metrics, "prompt_tps", "prefill_tps_estimate"),
        "decode_tps": metrics.get("decode_tps"),
        "elapsed_s": _first(metrics, "elapsed_s", "wall_s"),
        "sampling": record.get("sampling") or {},
        "parameters": ((record.get("slice") or {}).get("parameters") or {}),
        "runtime_parameters": ((record.get("slice") or {}).get("runtime_parameters") or {}),
        **_gpu_summary(record),
    }


def make_failure_event(
    *,
    attempt_id: str,
    model: str,
    requested_context_tokens: int,
    parameters: dict[str, Any],
    failure_kind: str,
    error: str,
    stdout_log: str,
    stderr_log: str,
) -> dict[str, Any]:
    return {
        "event_id": f"attempt:{attempt_id}",
        "status": failure_kind,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "requested_context_tokens": requested_context_tokens,
        "parameters": parameters,
        "error": error[-4000:],
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
    }


def classify_failure(text: str) -> str:
    lower = text.lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return "oom"
    if "timed out" in lower or "timeout" in lower:
        return "timeout"
    if "aborted" in lower or "killed" in lower:
        return "aborted"
    return "startup_error"


def append_event(ledger: Path, event: dict[str, Any]) -> bool:
    """Write an event exactly once, keyed by immutable artifact or attempt ID."""
    ledger = ledger.resolve()
    event_id = str(event["event_id"])
    seen: set[str] = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(str(json.loads(line).get("event_id")))
    if event_id in seen:
        return False
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(event, sort_keys=True) + "\n")
    return True


def backfill_artifacts(results_root: Path, ledger: Path) -> int:
    added = 0
    for artifact in sorted(results_root.rglob("*.json")):
        if artifact.name in {"summary.json"} or artifact.name.endswith(".jsonl"):
            continue
        if append_event(ledger, normalize_artifact(artifact)):
            added += 1
    return added


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--ledger", type=Path, default=Path(__file__).parent / "results" / "benchmark-events.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(backfill_artifacts(args.results_root, args.ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
