import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path(
    "outputs/stage3/features/crack500_features.csv"
)

OUTPUT_DIR = Path(
    "outputs/stage3/features/final_csi"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR / "crack500_csi_results.csv"
)


# --------------------------------------------------
# Load feature dataset
# --------------------------------------------------

df = pd.read_csv(
    INPUT_PATH
)


# --------------------------------------------------
# Final CSI dimensions
# --------------------------------------------------

CSI_FEATURES = [
    "crack_coverage_percent",
    "skeleton_length_pixels",
    "maximum_width_pixels",
    "branch_density"
]


# --------------------------------------------------
# Robust normalization
# 5th percentile -> 0
# 95th percentile -> 1
# --------------------------------------------------

normalized = pd.DataFrame(
    index=df.index
)


for feature in CSI_FEATURES:

    lower = df[feature].quantile(
        0.05
    )

    upper = df[feature].quantile(
        0.95
    )

    if upper == lower:

        normalized[feature] = 0.0

    else:

        normalized[feature] = (
            (
                df[feature] - lower
            )
            / (
                upper - lower
            )
        ).clip(0, 1)


# --------------------------------------------------
# Store normalized dimensions
# --------------------------------------------------

df["normalized_coverage"] = (
    normalized[
        "crack_coverage_percent"
    ]
)

df["normalized_skeleton_length"] = (
    normalized[
        "skeleton_length_pixels"
    ]
)

df["normalized_max_width"] = (
    normalized[
        "maximum_width_pixels"
    ]
)

df["normalized_branch_density"] = (
    normalized[
        "branch_density"
    ]
)


# --------------------------------------------------
# Proposed CSI
#
# Equal weighting:
# 25% each
# --------------------------------------------------

df["CSI"] = (
    0.25
    * df["normalized_coverage"]
    +
    0.25
    * df["normalized_skeleton_length"]
    +
    0.25
    * df["normalized_max_width"]
    +
    0.25
    * df["normalized_branch_density"]
)

df["CSI"] = (
    df["CSI"] * 100
)


# --------------------------------------------------
# Relative severity categories
#
# Based on CSI tertiles.
# These are dataset-relative categories,
# NOT externally validated engineering classes.
# --------------------------------------------------

low_threshold = df["CSI"].quantile(
    1 / 3
)

high_threshold = df["CSI"].quantile(
    2 / 3
)


def classify_severity(csi):

    if csi <= low_threshold:
        return "Low"

    elif csi <= high_threshold:
        return "Moderate"

    else:
        return "High"


df["severity"] = (
    df["CSI"].apply(
        classify_severity
    )
)


# --------------------------------------------------
# Round CSI
# --------------------------------------------------

df["CSI"] = (
    df["CSI"].round(2)
)


# --------------------------------------------------
# Save complete results
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 70)
print("FINAL PROPOSED CSI")
print("=" * 70)

print(
    f"Images analyzed: {len(df)}"
)

print(
    "\nNormalization: "
    "5th–95th percentile robust scaling"
)

print(
    "\nWeights:"
)

print(
    "Coverage          : 25%"
)

print(
    "Skeleton length   : 25%"
)

print(
    "Maximum width     : 25%"
)

print(
    "Branch density    : 25%"
)


print(
    "\nSeverity thresholds:"
)

print(
    f"Low / Moderate boundary: "
    f"{low_threshold:.2f}"
)

print(
    f"Moderate / High boundary: "
    f"{high_threshold:.2f}"
)


print(
    "\nCSI statistics:"
)

print(
    df["CSI"].describe().to_string()
)


print(
    "\nSeverity distribution:"
)

print(
    df["severity"]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    f"\nOutput: {OUTPUT_PATH}"
)


print(
    "\nFinal CSI calculation completed successfully."
)