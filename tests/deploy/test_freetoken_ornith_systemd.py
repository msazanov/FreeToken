from pathlib import Path
import re
import subprocess


UNIT_PATH = Path(__file__).parents[2] / "deploy/systemd/freetoken-ornith.service"


def test_production_unit_pins_the_fast_ornith_64k_profile() -> None:
    unit = UNIT_PATH.read_text(encoding="utf-8")

    required = (
        "Ornith-1.5-35B-A3B-TQ3_4S.gguf",
        "--host 127.0.0.1",
        "--port 19191",
        "--dtype bfloat16",
        "--max-seq-len-override 65536",
        "--num-tokens 65536",
        "--kv-reserve-tokens 65536",
        "--kv-cache-dtype int8",
        "--moe-cache-auto",
        "--moe-cache-policy lru",
        "--max-prefill-length 2560",
        "--max-running-requests 1",
        "--cache-type radix",
        "--enable-cache-report",
        "--moe-collect-stats",
    )
    for argument in required:
        assert argument in unit

    assert "122880" not in unit
    assert "--host 0.0.0.0" not in unit
    assert re.search(r"(?m)^\s+--port 1919\s", unit) is None
    assert "Restart=on-failure" in unit
    assert "WantedBy=default.target" in unit


def test_production_unit_has_no_systemd_validation_warnings() -> None:
    result = subprocess.run(
        ["systemd-analyze", "verify", "--user", str(UNIT_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
