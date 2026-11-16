"""Central configuration for the data-quality assessment pipeline."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RULE_DIR = PROJECT_DIR / "rule"
MDAD_RULE_DIR = RULE_DIR / "mdad"
SDAD_RULE_DIR = RULE_DIR / "sdad"
SRD_RULE_DIR = RULE_DIR / "srd"

RANDOM_SEED = 42

# Confidence penalties. Every confidence score in the pipeline must stay in [0, 1].
SDAD_VIOLATION_PENALTY_SEVERE = 0.3
SDAD_VIOLATION_PENALTY_MILD = 0.7
MDAD_SAMPLE_VIOLATION_PENALTY = 0.5
SRD_VIOLATION_PENALTY = 0.3

# Output thresholds.
LOW_CONFIDENCE_SAMPLE_THRESHOLD = 0.7
LOW_CONFIDENCE_FEATURE_THRESHOLD = 0.7
READABILITY_HIGH_THRESHOLD = 0.9
READABILITY_MEDIUM_THRESHOLD = 0.8
READABILITY_LOW_THRESHOLD = 0.7

OUTPUT_LOW_CONFIDENCE_SAMPLES = "low_confidence_samples.csv"
OUTPUT_LOW_CONFIDENCE_FEATURES = "low_confidence_features.csv"
OUTPUT_READABILITY_REPORT = "readability_score_report.csv"

MDAD_FEATURE_RELATIONSHIP_RULES = MDAD_RULE_DIR / "feature_relationship_rules.json"
SDAD_UNIFORM_RULES = SDAD_RULE_DIR / "uniform_rules.json"
SDAD_PERSONALIZED_RULES = SDAD_RULE_DIR / "personalized_rules.json"
SRD_CLUSTER_RULES = SRD_RULE_DIR / "cluster_rules.json"

SDAD_Z_SCORE_THRESHOLD = 3.0
SDAD_IQR_MULTIPLIER = 2.5

SRD_CLUSTER_METHOD = "kmeans"
SRD_N_CLUSTERS = "auto"
SRD_AUTO_DETERMINATION_METHOD = "enhanced_elbow"
# Minimum desired membership of a cluster, not a minimum number of clusters.
SRD_MIN_CLUSTER_SIZE = 14
# A sample must be reported by at least this many enabled statistical detectors.
SRD_MIN_DETECTOR_VOTES = 2
# 保存模型复用时样本数变化容忍度：当前样本数与训练时样本数之比超出 [1-threshold, 1+threshold] 范围时重新训练
SRD_MODEL_REUSE_SAMPLE_CHANGE_THRESHOLD = 0.50

SRD_STATISTICAL_METHODS = {
    "isolation_forest": {
        "enabled": True,
        "contamination": "auto",
        "max_samples": "auto",
        "n_estimators": 50,
        "random_state": RANDOM_SEED,
    },
    "local_outlier_factor": {
        "enabled": True,
        "n_neighbors": 15,
        "contamination": 0.1,
    },
    "one_class_svm": {
        "enabled": False,
        "nu": 0.1,
        "kernel": "rbf",
        "gamma": "auto",
    },
    "knn": {
        "enabled": True,
        "n_neighbors": 50,
        "contamination": 0.05,
        "method": "mean",
    },
    "mcd": {
        "enabled": True,
        "contamination": 0.1,
        "random_state": RANDOM_SEED,
    },
}
