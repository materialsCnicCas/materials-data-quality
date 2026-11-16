"""Evaluate the paper's generic detector on user-supplied ODDS matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project.src.odds_adaptive_anomaly import ODDSAdaptiveAnomalyDetector  # noqa: E402


LABEL_NAMES = ("label", "labels", "y", "target", "outlier", "is_outlier", "class")


def _resolve_label(values: list[str]) -> str | None:
    lowered = {value.lower(): value for value in values}
    for candidate in LABEL_NAMES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    label_column = _resolve_label(list(frame.columns))
    if label_column is None:
        raise ValueError(f"{path.name}: no label column among {LABEL_NAMES}")
    labels = pd.to_numeric(frame.pop(label_column), errors="coerce")
    numeric = frame.select_dtypes(include=[np.number])
    valid = labels.notna().to_numpy()
    return numeric.loc[valid].to_numpy(float), labels.loc[valid].to_numpy(int)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        label_key = _resolve_label(keys)
        if label_key is None:
            raise ValueError(f"{path.name}: no label array among {LABEL_NAMES}")
        labels = np.asarray(archive[label_key]).reshape(-1)
        feature_keys = [key for key in keys if key != label_key]
        if "X" in feature_keys:
            features = np.asarray(archive["X"])
        elif "x" in feature_keys:
            features = np.asarray(archive["x"])
        else:
            features = np.column_stack([np.asarray(archive[key]) for key in feature_keys])
    return features.astype(float), labels.astype(int)


def load_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    content = loadmat(path)
    label_key = _resolve_label([key for key in content if not key.startswith("__")])
    if label_key is None:
        raise ValueError(f"{path.name}: no label array among {LABEL_NAMES}")
    labels = np.asarray(content[label_key]).reshape(-1)
    feature_key = next(key for key in ("X", "x", "data") if key in content)
    return np.asarray(content[feature_key], dtype=float), labels.astype(int)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".npz":
        return load_npz(path)
    if suffix == ".mat":
        return load_mat(path)
    if suffix == ".npy":
        matrix = np.load(path)
        if matrix.ndim != 2:
            raise ValueError(f"{path.name}: .npy input must be two-dimensional")
        return matrix[:, :-1].astype(float), matrix[:, -1].astype(int)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime_outputs" / "odds_auc.csv")
    args = parser.parse_args()

    files = [args.data_dir] if args.data_dir.is_file() else sorted(
        path for path in args.data_dir.rglob("*")
        if path.suffix.lower() in {".csv", ".npz", ".mat", ".npy"}
    )
    if not files:
        raise FileNotFoundError(f"No supported dataset files found under {args.data_dir}")

    rows = []
    for path in files:
        try:
            features, labels = load_dataset(path)
            unique = np.unique(labels)
            if len(unique) != 2:
                raise ValueError(f"expected two label classes, found {unique.tolist()}")
            if not set(unique.tolist()).issubset({0, 1}):
                raise ValueError("labels must use 0 for normal and 1 for anomaly")
            scores, components = ODDSAdaptiveAnomalyDetector().fit_score(features)
            run = {
                "dataset": path.stem,
                "rows": len(labels),
                "features": features.shape[1],
                "auc_roc": float(roc_auc_score(labels, scores)),
                "route_code": int(components["route_code"][0]),
            }
            rows.append(run)
            print(f"{run['dataset']}: AUC-ROC={run['auc_roc']:.6f}")
        except Exception as exc:
            rows.append({"dataset": path.stem, "error": str(exc)})
            print(f"{path.name}: ERROR {exc}", file=sys.stderr)

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    return 0 if not result.get("error", pd.Series(dtype=str)).notna().any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
