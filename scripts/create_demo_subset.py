"""Create the deterministic demonstration subset from the frozen mp_20 input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "mp20" / "mp_20.csv"
DESTINATION = ROOT / "examples" / "mp20_demo_1000.csv"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    if len(frame) < 1000:
        raise ValueError("The source input contains fewer than 1,000 records.")
    subset = frame.sample(n=1000, random_state=42).reset_index(drop=True)
    subset.to_csv(DESTINATION, index=False, encoding="utf-8")
    print(f"Wrote {len(subset)} records to {DESTINATION}")


if __name__ == "__main__":
    main()
