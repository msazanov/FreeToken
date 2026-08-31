from __future__ import annotations

import subprocess
import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
DEPLOY = ROOT / "deploy" / "systemd"


def _unit(name: str) -> str:
    return (DEPLOY / name).read_text(encoding="utf-8")


def test_arbiter_topology_has_one_public_port_and_private_backends() -> None:
    arbiter = _unit("freetoken-arbiter.service")
    ornith = _unit("freetoken-ornith.service")
    daemon = _unit("freetoken-daemon.service")
    cpu = _unit("llama-gemma-cpu.service")

    assert "--host 0.0.0.0" in arbiter
    assert "--port 1919" in arbiter
    assert "--ornith-url http://127.0.0.1:19191" in arbiter
    assert "--gemma-gpu-url http://127.0.0.1:19192" in arbiter
    assert "--gemma-cpu-url http://127.0.0.1:19193" in arbiter
    assert "Requires=freetoken-ornith.service freetoken-daemon.service" in arbiter

    assert "--host 127.0.0.1" in ornith
    assert "--port 19191" in ornith
    assert "--host 0.0.0.0" not in ornith
    assert re.search(r"(?m)^\s+--port 1919\s", ornith) is None
    assert "--max-seq-len-override 65536" in ornith
    assert "--num-tokens 65536" in ornith
    assert "--kv-reserve-tokens 65536" in ornith

    assert "ft daemon" in daemon
    assert "--port 1900" in daemon
    assert "KillMode=process" in daemon

    assert "--host 127.0.0.1" in cpu
    assert "--port 19193" in cpu
    assert "--gpu-layers 0" in cpu
    assert "--alias gemma-4-e2b" in cpu


def test_all_source_controlled_units_pass_systemd_verify() -> None:
    units = sorted(DEPLOY.glob("*.service"))
    result = subprocess.run(
        ["systemd-analyze", "verify", "--user", *(str(unit) for unit in units)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
