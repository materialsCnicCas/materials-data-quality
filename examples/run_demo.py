"""Run a deterministic, no-API-key end-to-end quality-assessment demo."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "mp20_demo_1000.csv"
DEFAULT_OUTPUT = ROOT / "runtime_outputs" / f"demo_{datetime.now():%Y%m%d_%H%M%S}"
OUTPUT = Path(os.environ.get("DATAVALUE_DEMO_OUTPUT", DEFAULT_OUTPUT)).resolve()


def main() -> int:
    if not INPUT.exists():
        print(
            "Demo input is missing. Recreate it with:\n"
            "  python scripts/create_demo_subset.py",
            file=sys.stderr,
        )
        return 2

    environment = os.environ.copy()
    environment["SRD_CLUSTER_MODEL_DIR"] = str(OUTPUT / "cluster_model")
    command = [
        sys.executable,
        str(ROOT / "main_v2.py"),
        str(INPUT),
        "--profile",
        "paper",
        "--system-mode",
        "manual",
        "--non-interactive",
        "--disable-quality-memory",
        "--enable-audit-report",
        "--output-dir",
        str(OUTPUT),
    ]
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, env=environment)
    if completed.returncode == 0:
        print(f"Demo completed. Results: {OUTPUT}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
