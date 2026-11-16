"""Static and data-integrity checks for the public release."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "main_v2.py",
    "project/llm_config.py",
    "data/mp20/mp_20.csv",
    "data/alloys/alloy_case_study_185_records.xlsx",
    "data/alloys/alloy_reference_315_records.xlsx",
    "experiments/results/validation_summary.csv",
    "experiments/results/mace_summary.csv",
    "experiments/results/memory_summary.csv",
    "experiments/results/robustness_summary.csv",
]
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
MACHINE_PATH_PATTERNS = (
    re.compile(r"C:\\Users\\[^\\\r\n]+", re.IGNORECASE),
    re.compile("/" + "home/" + r"[^/\s]+/", re.IGNORECASE),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing required file: {relative}")

    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("SRD_CLUSTER_MODEL_DIR", str(ROOT / "runtime_outputs" / "models"))
    try:
        importlib.import_module("project")
        importlib.import_module("project.llm_config")
        importlib.import_module("main_v2")
    except Exception as exc:
        errors.append(f"import failure: {exc}")

    text_suffixes = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".env", ".example"}
    allowed_path_roots = {
        ROOT / "experiments" / "mace_evidence",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if SECRET_PATTERN.search(text):
            errors.append(f"possible credential: {path.relative_to(ROOT)}")
        for pattern in MACHINE_PATH_PATTERNS:
            match = pattern.search(text)
            if match and not any(path.is_relative_to(prefix) for prefix in allowed_path_roots):
                errors.append(
                    f"machine-specific path in {path.relative_to(ROOT)}: {match.group(0)}"
                )

    mp20 = ROOT / "data" / "mp20" / "mp_20.csv"
    expected_hash = "c0f55956e7d0615c15d0c2044865465cdb02846d69e0b3091d7b9c8ffb65c63a"
    if mp20.exists() and sha256(mp20) != expected_hash:
        errors.append("mp_20.csv hash does not match the manuscript input hash")

    manifest = ROOT / "manifests" / "SHA256SUMS.txt"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.is_file():
                errors.append(f"manifest entry is missing: {relative}")
            elif sha256(path) != expected:
                errors.append(f"manifest hash mismatch: {relative}")

    if errors:
        print("Release verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Release verification passed.")
    print(f"Checked {len(REQUIRED)} required files and scanned source/data text for secrets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
