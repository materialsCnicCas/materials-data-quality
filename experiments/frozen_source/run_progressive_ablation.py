from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = EXP_DIR / "intermediate_progressive"
RESULTS_DIR = EXP_DIR / "results_progressive"
FIGURES_DIR = EXP_DIR / "figures_progressive"
for d in (INTERMEDIATE_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_ROWS = 16000
TARGET_COL = "formation_energy_per_atom"
NOISE_RATIO = 0.5
NOISE_SIGMA = 0.6


def domain_ranges():
    ranges = {
        "band_gap": (0.0, 10.0),
        "e_above_hull": (0.0, 0.5),
        "spacegroup_number": (1.0, 230.0),
        "abc_0": (2.0, 30.0),
        "abc_1": (2.0, 30.0),
        "abc_2": (2.0, 30.0),
        "angle_0": (30.0, 150.0),
        "angle_1": (30.0, 150.0),
        "angle_2": (30.0, 150.0),
    }
    for i in range(20):
        for axis in ("x", "y", "z"):
            ranges[f"coord_{i}_{axis}"] = (0.0, 1.0)
    return ranges


def load_data():
    df = pd.read_csv(ROOT / "data" / "mp20" / "mp_20.csv")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != TARGET_COL and c != "id"]
    keep = [TARGET_COL] + feature_cols
    data = df[keep].copy().dropna(subset=[TARGET_COL])
    missing = data[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if missing[c] <= 0.4]
    data = data[[TARGET_COL] + feature_cols].copy()
    rng = np.random.default_rng(RANDOM_SEED)
    if len(data) > MAX_ROWS:
        idx = rng.choice(data.index.to_numpy(), MAX_ROWS, replace=False)
        data = data.loc[idx].sort_index()
    data = data.reset_index(drop=True)
    for c in feature_cols:
        data[c] = data[c].fillna(data[c].median()).astype(float)
    data[TARGET_COL] = data[TARGET_COL].astype(float)
    return data, feature_cols


def inject_mixed_pollution(clean, feature_cols):
    rng = np.random.default_rng(RANDOM_SEED)
    noisy = clean.copy()
    dirty_y = clean[TARGET_COL].copy()
    n = len(clean)
    k = int(n * NOISE_RATIO)
    polluted = rng.choice(np.arange(n), k, replace=False)
    labels = np.zeros(n, dtype=int)
    labels[polluted] = 1
    types = np.array(["clean"] * n, dtype=object)

    groups = np.array_split(polluted, 4)
    std = clean[feature_cols].std(ddof=0).replace(0, 1.0)
    y_std = clean[TARGET_COL].std(ddof=0)

    # Group 1: statistical outliers in several generic numerical descriptors.
    stat_cols = [c for c in feature_cols if c not in domain_ranges()][:18]
    if len(stat_cols) < 8:
        stat_cols = feature_cols[:18]
    for ridx in groups[0]:
        chosen = rng.choice(stat_cols, size=min(8, len(stat_cols)), replace=False)
        noisy.loc[ridx, chosen] = noisy.loc[ridx, chosen] + rng.normal(0, 4.0, len(chosen)) * std[chosen].to_numpy()
        dirty_y.iloc[ridx] = dirty_y.iloc[ridx] + rng.normal(0, 2.8) * y_std
        types[ridx] = "statistical"

    # Group 2: physically impossible single-field values.
    phys_cols = [c for c in feature_cols if c in domain_ranges()]
    for ridx in groups[1]:
        chosen = rng.choice(phys_cols, size=min(5, len(phys_cols)), replace=False)
        for col in chosen:
            low, high = domain_ranges()[col]
            noisy.at[ridx, col] = low - abs(high - low) * 0.35 if rng.random() < 0.5 else high + abs(high - low) * 0.35
        dirty_y.iloc[ridx] = dirty_y.iloc[ridx] + rng.normal(0, 2.8) * y_std
        types[ridx] = "domain"

    # Group 3: relationship violations that keep most marginal values plausible.
    relation_triplets = [
        ("abc_0", "lattice_mat_0_0", "lattice_mat_0_1", "lattice_mat_0_2"),
        ("abc_1", "lattice_mat_1_0", "lattice_mat_1_1", "lattice_mat_1_2"),
        ("abc_2", "lattice_mat_2_0", "lattice_mat_2_1", "lattice_mat_2_2"),
    ]
    relation_triplets = [t for t in relation_triplets if all(c in feature_cols for c in t)]
    for ridx in groups[2]:
        for abc, m0, m1, m2 in relation_triplets:
            norm = float(np.sqrt(noisy.at[ridx, m0] ** 2 + noisy.at[ridx, m1] ** 2 + noisy.at[ridx, m2] ** 2))
            noisy.at[ridx, abc] = 2.0 if norm > 8.0 else 30.0
        dirty_y.iloc[ridx] = dirty_y.iloc[ridx] + rng.normal(0, 2.8) * y_std
        types[ridx] = "relationship"

    # Group 4: mixed subtle errors.
    for ridx in groups[3]:
        chosen = rng.choice(feature_cols, size=min(12, len(feature_cols)), replace=False)
        noisy.loc[ridx, chosen] = noisy.loc[ridx, chosen] + rng.normal(0, NOISE_SIGMA, len(chosen)) * std[chosen].to_numpy()
        cluster_cols = feature_cols[: min(8, len(feature_cols))]
        noisy.loc[ridx, cluster_cols] = noisy.loc[ridx, cluster_cols] + 1.8 * std[cluster_cols].to_numpy()
        if "band_gap" in feature_cols and rng.random() < 0.5:
            noisy.at[ridx, "band_gap"] = -rng.uniform(0.05, 1.5)
        if relation_triplets:
            abc, *_ = relation_triplets[rng.integers(0, len(relation_triplets))]
            noisy.at[ridx, abc] = np.clip(noisy.at[ridx, abc] * rng.uniform(1.25, 1.6), 2.0, 30.0)
        dirty_y.iloc[ridx] = dirty_y.iloc[ridx] + rng.normal(0, 2.8) * y_std
        types[ridx] = "mixed"

    payload = noisy.copy()
    payload["dirty_target"] = dirty_y
    payload["pollution_label"] = labels
    payload["pollution_type"] = types
    payload.to_csv(INTERMEDIATE_DIR / "mixed_pollution_dataset.csv", index=False, encoding="utf-8-sig")
    return noisy, dirty_y, labels, types


def statistical_score(X):
    arr = X.to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    q1 = np.quantile(arr, 0.25, axis=0)
    q3 = np.quantile(arr, 0.75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    z = np.abs((arr - mean) / std)
    iqr_flag = (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)
    score = (z > 3.0).mean(axis=1) + 0.4 * iqr_flag.mean(axis=1)
    return normalize(score)


def domain_score(X, feature_cols):
    flags = np.zeros((len(X), len(feature_cols)), dtype=float)
    for j, col in enumerate(feature_cols):
        if col not in domain_ranges():
            continue
        low, high = domain_ranges()[col]
        v = X[col].to_numpy(dtype=float)
        flags[:, j] = ((v < low) | (v > high)).astype(float)
    return normalize(flags.mean(axis=1))


def relationship_score(X):
    scores = []
    triplets = [
        ("abc_0", "lattice_mat_0_0", "lattice_mat_0_1", "lattice_mat_0_2"),
        ("abc_1", "lattice_mat_1_0", "lattice_mat_1_1", "lattice_mat_1_2"),
        ("abc_2", "lattice_mat_2_0", "lattice_mat_2_1", "lattice_mat_2_2"),
    ]
    for abc, m0, m1, m2 in triplets:
        if all(c in X.columns for c in (abc, m0, m1, m2)):
            norm = np.sqrt(X[m0] ** 2 + X[m1] ** 2 + X[m2] ** 2)
            resid = np.abs(X[abc] - norm) / (np.abs(norm) + 1e-6)
            scores.append(resid.to_numpy())
    if not scores:
        return np.zeros(len(X))
    arr = np.vstack(scores).T
    return normalize(arr.max(axis=1) + 0.5 * arr.mean(axis=1))


def graph_score(X, base_score, k=12):
    arr = X.to_numpy(dtype=float)
    arr = StandardScaler().fit_transform(arr)
    arr = arr[:, : min(arr.shape[1], 35)]
    # Low-confidence samples whose neighbors are also suspicious receive a boost,
    # while isolated single-rule hits remain mostly unchanged.
    nn_model = NearestNeighbors(n_neighbors=min(k + 1, len(arr)), algorithm="auto")
    nn_model.fit(arr)
    _, nn = nn_model.kneighbors(arr)
    nn = nn[:, 1:] if nn.shape[1] > 1 else nn
    neigh_mean = base_score[nn].mean(axis=1)
    leverage = np.sqrt(np.mean(arr ** 2, axis=1))
    leverage = normalize(leverage)
    residual_room = 1.0 - base_score
    return normalize(base_score + 0.06 * residual_room * neigh_mean + 0.04 * residual_room * leverage)


def normalize(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def metrics(labels, score, ratio=NOISE_RATIO):
    k = int(round(len(labels) * ratio))
    pred = np.zeros(len(labels), dtype=int)
    pred[np.argsort(score)[-k:]] = 1
    return {
        "auc": float(roc_auc_score(labels, score)),
        "ap": float(average_precision_score(labels, score)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
    }


def downstream(clean, noisy, dirty_y, feature_cols, labels, scores):
    idx = np.arange(len(clean))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=RANDOM_SEED)
    X_test = clean.iloc[test_idx][feature_cols]
    y_test = clean.iloc[test_idx][TARGET_COL]
    rows = []

    def fit_eval(name, train_rows, X_train, y_train, det=None, sample_weight=None):
        model = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.055, l2_regularization=0.02, random_state=RANDOM_SEED)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        pred = model.predict(X_test)
        row = {
            "method": name,
            "train_rows": int(len(X_train)),
            "removed_train_rows": int(len(train_rows) - len(X_train)) if isinstance(train_rows, np.ndarray) else 0,
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
        }
        if det:
            row.update(det)
        rows.append(row)

    fit_eval("clean_upper_bound", train_idx, clean.iloc[train_idx][feature_cols], clean.iloc[train_idx][TARGET_COL])
    fit_eval("no_cleaning", train_idx, noisy.iloc[train_idx][feature_cols], dirty_y.iloc[train_idx])
    remove_k = int(labels[train_idx].sum())
    for name, score in scores.items():
        train_score = normalize(score[train_idx])
        remove_local = np.argsort(train_score)[-remove_k:]
        keep_idx = np.setdiff1d(train_idx, train_idx[remove_local])
        det = metrics(labels, score)
        fit_eval(
            name,
            train_idx,
            noisy.iloc[keep_idx][feature_cols],
            dirty_y.iloc[keep_idx],
            det,
        )
    return pd.DataFrame(rows)


def plot(ablation, type_df):
    import matplotlib.pyplot as plt

    method_order = ["statistical", "stat_domain", "stat_domain_relation", "full"]
    det_df = ablation[ablation["method"].isin(method_order)].set_index("method").loc[method_order]
    down_df = ablation[
        ablation["method"].isin(
            ["clean_upper_bound", "no_cleaning", "statistical", "stat_domain", "stat_domain_relation", "full"]
        )
    ].copy()
    det_labels = ["Stat.", "+Domain", "+Relation", "+Graph"]
    down_labels = ["Clean", "No clean", "Stat.", "+Domain", "+Relation", "+Graph"]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=260)
    x = np.arange(len(method_order))
    axes[0].plot(x, det_df["ap"], marker="o", linewidth=2.0, color="#2A6FBB", label="AP")
    axes[0].plot(x, det_df["f1"], marker="s", linewidth=2.0, color="#C73E1D", label="F1")
    axes[0].set_xticks(x, det_labels)
    axes[0].set_ylim(0.84, 1.01)
    axes[0].set_ylabel("Detection score")
    axes[0].set_title("(a) Anomaly-ranking quality")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axes[0].legend(frameon=False, loc="lower right")

    colors = ["#5B5B5B", "#B94E48", "#7BA7C9", "#5FA37A", "#E0A33A", "#2A6FBB"]
    x2 = np.arange(len(down_df))
    axes[1].bar(x2, down_df["r2"], color=colors, width=0.62)
    axes[1].set_xticks(x2, down_labels, rotation=18)
    axes[1].set_ylim(0.42, max(down_df["r2"]) + 0.04)
    axes[1].set_ylabel("Downstream R2")
    axes[1].set_title("(b) Proxy prediction after cleaning")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    for xi, val in zip(x2, down_df["r2"]):
        axes[1].text(xi, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "progressive_ablation_r2_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    type_df.set_index("pollution_type")[["statistical", "stat_domain", "stat_domain_relation", "full"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Recall by pollution type")
    ax.set_ylim(0, 1.05)
    ax.set_title("Module coverage for different pollution mechanisms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pollution_type_recall.png")
    plt.close(fig)


def type_recall(labels, types, scores):
    rows = []
    for t in ["statistical", "domain", "relationship", "mixed"]:
        mask = types == t
        row = {"pollution_type": t, "count": int(mask.sum())}
        for name, score in scores.items():
            k = int(labels.sum())
            pred = np.zeros(len(labels), dtype=int)
            pred[np.argsort(score)[-k:]] = 1
            row[name] = float(((pred == 1) & mask).sum() / max(mask.sum(), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    clean, feature_cols = load_data()
    noisy, dirty_y, labels, types = inject_mixed_pollution(clean, feature_cols)
    X = noisy[feature_cols]

    s_stat = statistical_score(X)
    s_domain = domain_score(X, feature_cols)
    s_relation = relationship_score(X)
    score_stat = s_stat
    score_stat_domain = normalize(score_stat + s_domain)
    # Relationship violations are strong materials-knowledge evidence: they may
    # remain marginally plausible in a single column but break lattice consistency.
    score_stat_domain_relation = normalize(0.2 * score_stat + 0.5 * s_domain + 6.0 * s_relation)
    graph_refined = graph_score(X, score_stat_domain_relation)
    score_full = normalize(0.95 * score_stat_domain_relation + 0.05 * graph_refined)

    scores = {
        "statistical": score_stat,
        "stat_domain": score_stat_domain,
        "stat_domain_relation": score_stat_domain_relation,
        "full": score_full,
    }
    pd.DataFrame({"label": labels, "type": types, **{k: v for k, v in scores.items()}}).to_csv(
        INTERMEDIATE_DIR / "progressive_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    detection_rows = []
    for name, score in scores.items():
        row = {"method": name}
        row.update(metrics(labels, score))
        detection_rows.append(row)
    detection = pd.DataFrame(detection_rows)
    detection.to_csv(RESULTS_DIR / "progressive_detection_metrics.csv", index=False, encoding="utf-8-sig")

    ablation = downstream(clean, noisy, dirty_y, feature_cols, labels, scores)
    ablation.to_csv(RESULTS_DIR / "progressive_downstream_metrics.csv", index=False, encoding="utf-8-sig")

    type_df = type_recall(labels, types, scores)
    type_df.to_csv(RESULTS_DIR / "pollution_type_recall.csv", index=False, encoding="utf-8-sig")
    plot(ablation, type_df)

    summary = {
        "sample_count": int(len(clean)),
        "feature_count": int(len(feature_cols)),
        "noise_ratio": NOISE_RATIO,
        "target": TARGET_COL,
        "detection": detection.to_dict(orient="records"),
        "downstream": ablation.to_dict(orient="records"),
        "type_recall": type_df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "progressive_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
