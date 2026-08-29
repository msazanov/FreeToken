"""Append-only index of reproducible context-speed benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _entry_from_record(
    record: dict[str, Any],
    *,
    artifact: Path,
    model: str,
    quantization: str,
    runtime_profile: str,
    source_kind: str,
    git_commit: str,
) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    return {
        "timestamp_utc": record.get("timestamp_utc"),
        "artifact": str(artifact),
        "model": model,
        "quantization": quantization,
        "runtime_profile": runtime_profile,
        "source_kind": source_kind,
        "git_commit": git_commit,
        "requested_context_tokens": record.get("requested_context_tokens"),
        "actual_context_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "prefill_tps": metrics.get("prompt_tps", metrics.get("prefill_tps_estimate")),
        "decode_tps": metrics.get("decode_tps"),
        "elapsed_s": metrics.get("elapsed_s"),
        "sampling": record.get("sampling") or {},
    }


def append_artifact(
    artifact: Path,
    registry: Path,
    *,
    model: str,
    quantization: str,
    runtime_profile: str,
    source_kind: str,
    git_commit: str,
) -> bool:
    """Append an artifact once and return whether the registry changed."""
    artifact = artifact.resolve()
    registry = registry.resolve()
    record = json.loads(artifact.read_text(encoding="utf-8"))
    existing_artifacts: set[str] = set()
    if registry.exists():
        for line in registry.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_artifacts.add(str(json.loads(line).get("artifact")))
    if str(artifact) in existing_artifacts:
        return False
    registry.parent.mkdir(parents=True, exist_ok=True)
    entry = _entry_from_record(
        record,
        artifact=artifact,
        model=model,
        quantization=quantization,
        runtime_profile=runtime_profile,
        source_kind=source_kind,
        git_commit=git_commit,
    )
    with registry.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(entry, sort_keys=True) + "\n")
    return True
