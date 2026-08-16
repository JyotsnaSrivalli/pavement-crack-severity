import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path(
    "outputs/stage3/features/final_csi/crack500_csi_results.csv"
)

OUTPUT_DIR = Path(
    "outputs/stage3/features/final_csi/validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load CSI results
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("PROPOSED CSI VALIDATION")
print("=" * 70)

print(
    f"Images analyzed: {len(df)}"
)


# --------------------------------------------------
# Features used by CSI
# --------------------------------------------------

NORMALIZED_FEATURES = [
    "normalized_coverage",
    "normalized_skeleton_length",
    "normalized_max_width",
    "normalized_branch_density"
]

ORIGINAL_FEATURES = [
    "crack_coverage_percent",
    "skeleton_length_pixels",
    "maximum_width_pixels",
    "branch_density"
]


# --------------------------------------------------
# 1. CSI correlation analysis
# --------------------------------------------------

print("\n" + "=" * 70)
print("CSI CORRELATION WITH COMPONENTS")
print("=" * 70)

correlations = []

for normalized, original in zip(
    NORMALIZED_FEATURES,
    ORIGINAL_FEATURES
):

    correlation = df[
        ["CSI", normalized]
    ].corr().iloc[0, 1]

    correlations.append({
        "feature": original,
        "normalized_feature": normalized,
        "CSI_correlation": correlation
    })

    print(
        f"{original:<30} "
        f"{correlation:.4f}"
    )


correlation_df = pd.DataFrame(
    correlations
)

correlation_df.to_csv(
    OUTPUT_DIR
    / "csi_component_correlations.csv",
    index=False
)


# --------------------------------------------------
# 2. Calculate individual contributions
# --------------------------------------------------

df["coverage_contribution"] = (
    df["normalized_coverage"] * 25
)

df["length_contribution"] = (
    df["normalized_skeleton_length"] * 25
)

df["width_contribution"] = (
    df["normalized_max_width"] * 25
)

df["complexity_contribution"] = (
    df["normalized_branch_density"] * 25
)


# --------------------------------------------------
# Contribution statistics
# --------------------------------------------------

contribution_columns = [
    "coverage_contribution",
    "length_contribution",
    "width_contribution",
    "complexity_contribution"
]

contribution_statistics = (
    df[contribution_columns]
    .describe()
    .T
)

contribution_statistics.to_csv(
    OUTPUT_DIR
    / "csi_contribution_statistics.csv"
)


print("\n" + "=" * 70)
print("CSI COMPONENT CONTRIBUTIONS")
print("=" * 70)

print(
    contribution_statistics[
        [
            "mean",
            "std",
            "min",
            "50%",
            "max"
        ]
    ].to_string()
)


# --------------------------------------------------
# 3. Severity-group analysis
# --------------------------------------------------

group_features = [
    "CSI",
    "crack_coverage_percent",
    "skeleton_length_pixels",
    "maximum_width_pixels",
    "branch_density"
]

group_statistics = (
    df.groupby("severity")[
        group_features
    ]
    .agg(
        [
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max"
        ]
    )
)


group_statistics.to_csv(
    OUTPUT_DIR
    / "severity_group_statistics.csv"
)


print("\n" + "=" * 70)
print("SEVERITY GROUP STATISTICS")
print("=" * 70)

print(
    group_statistics.to_string()
)


# --------------------------------------------------
# 4. Simpler mean table for interpretation
# --------------------------------------------------

mean_table = (
    df.groupby("severity")[
        group_features
    ]
    .mean()
    .reindex(
        ["Low", "Moderate", "High"]
    )
)


mean_table.to_csv(
    OUTPUT_DIR
    / "severity_group_means.csv"
)


print("\n" + "=" * 70)
print("MEAN FEATURE VALUES BY SEVERITY")
print("=" * 70)

print(
    mean_table.to_string()
)


# --------------------------------------------------
# 5. Check monotonic progression
# --------------------------------------------------

print("\n" + "=" * 70)
print("MONOTONIC PROGRESSION CHECK")
print("=" * 70)


ordered_means = mean_table


for feature in group_features:

    values = (
        ordered_means[feature]
        .values
    )

    increasing = (
        values[0]
        <= values[1]
        <= values[2]
    )

    decreasing = (
        values[0]
        >= values[1]
        >= values[2]
    )

    if increasing:

        result = "PASS - increasing"

    elif decreasing:

        result = "PASS - decreasing"

    else:

        result = "CHECK - non-monotonic"

    print(
        f"{feature:<30} {result}"
    )


# --------------------------------------------------
# 6. CSI distribution by severity
# --------------------------------------------------

severity_distribution = (
    df.groupby("severity")[
        "CSI"
    ]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max"
        ]
    )
    .reindex(
        ["Low", "Moderate", "High"]
    )
)


severity_distribution.to_csv(
    OUTPUT_DIR
    / "csi_by_severity.csv"
)


print("\n" + "=" * 70)
print("CSI DISTRIBUTION BY SEVERITY")
print("=" * 70)

print(
    severity_distribution.to_string()
)


# --------------------------------------------------
# 7. Overall CSI statistics
# --------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL CSI STATISTICS")
print("=" * 70)

print(
    df["CSI"].describe().to_string()
)


# --------------------------------------------------
# 8. Save complete validation dataset