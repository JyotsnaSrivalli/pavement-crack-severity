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
    "outputs/stage3/features/normalization_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# Candidate CSI dimensions
# --------------------------------------------------

FEATURES = [
    "crack_coverage_percent",
    "skeleton_length_pixels",
    "maximum_width_pixels",
    "branch_density"
]


# --------------------------------------------------
# Min-Max normalization
# --------------------------------------------------

minmax = pd.DataFrame(
    index=df.index
)

for feature in FEATURES:

    minimum = df[feature].min()
    maximum = df[feature].max()

    if maximum == minimum:

        minmax[feature] = 0.0

    else:

        minmax[feature] = (
            (df[feature] - minimum)
            / (maximum - minimum)
        )


# --------------------------------------------------
# Robust percentile normalization
#
# 5th percentile -> 0
# 95th percentile -> 1
#
# Values outside the range are clipped.
# --------------------------------------------------

robust = pd.DataFrame(
    index=df.index
)

for feature in FEATURES:

    lower = df[feature].quantile(
        0.05
    )

    upper = df[feature].quantile(
        0.95
    )

    if upper == lower:

        robust[feature] = 0.0

    else:

        normalized = (
            (df[feature] - lower)
            / (upper - lower)
        )

        robust[feature] = (
            normalized.clip(0, 1)
        )


# --------------------------------------------------
# Compare distributions
# --------------------------------------------------

summary_rows = []

for feature in FEATURES:

    summary_rows.append({

        "feature": feature,

        "original_min":
            df[feature].min(),

        "original_median":
            df[feature].median(),

        "original_max":
            df[feature].max(),

        "minmax_mean":
            minmax[feature].mean(),

        "minmax_median":
            minmax[feature].median(),

        "minmax_std":
            minmax[feature].std(),

        "robust_mean":
            robust[feature].mean(),

        "robust_median":
            robust[feature].median(),

        "robust_std":
            robust[feature].std(),

        "minmax_at_zero":
            (
                minmax[feature] == 0
            ).sum(),

        "minmax_at_one":
            (
                minmax[feature] == 1
            ).sum(),

        "robust_at_zero":
            (
                robust[feature] == 0
            ).sum(),

        "robust_at_one":
            (
                robust[feature] == 1
            ).sum()
    })


summary = pd.DataFrame(
    summary_rows
)


# --------------------------------------------------
# Save summary
# --------------------------------------------------

summary_path = (
    OUTPUT_DIR
    / "normalization_comparison.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# --------------------------------------------------
# Save normalized values
# --------------------------------------------------

minmax_output = (
    OUTPUT_DIR
    / "minmax_features.csv"
)

robust_output = (
    OUTPUT_DIR
    / "robust_features.csv"
)


minmax.to_csv(
    minmax_output,
    index=False
)

robust.to_csv(
    robust_output,
    index=False
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("=" * 70)
print("NORMALIZATION COMPARISON")
print("=" * 70)

print(
    f"Images analyzed: {len(df)}"
)

print(
    "\nCandidate CSI dimensions:"
)

for feature in FEATURES:
    print(f" - {feature}")


print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(summary_path)
print(minmax_output)
print(robust_output)

print(
    "\nNormalization analysis completed successfully."
)