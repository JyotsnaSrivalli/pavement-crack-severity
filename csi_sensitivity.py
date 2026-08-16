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
    "outputs/stage3/features/final_csi/sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    INPUT_PATH
)


# --------------------------------------------------
# Normalized CSI dimensions
# --------------------------------------------------

FEATURES = [
    "normalized_coverage",
    "normalized_skeleton_length",
    "normalized_max_width",
    "normalized_branch_density"
]


# --------------------------------------------------
# Predefined weighting schemes
# --------------------------------------------------

WEIGHTS = {

    "A_equal":
        [0.25, 0.25, 0.25, 0.25],

    "B_extent":
        [0.30, 0.25, 0.25, 0.20],

    "C_width":
        [0.25, 0.25, 0.30, 0.20],

    "D_extent_width":
        [0.30, 0.20, 0.30, 0.20]
}


# --------------------------------------------------
# Calculate CSI for each scheme
# --------------------------------------------------

results = pd.DataFrame(
    index=df.index
)

results["image"] = df["image"]


for scheme, weights in WEIGHTS.items():

    csi = np.zeros(
        len(df)
    )

    for feature, weight in zip(
        FEATURES,
        weights
    ):

        csi += (
            df[feature]
            * weight
        )

    results[
        f"CSI_{scheme}"
    ] = csi * 100


# --------------------------------------------------
# Summary statistics
# --------------------------------------------------

print("=" * 70)
print("CSI SENSITIVITY ANALYSIS")
print("=" * 70)

print(
    f"Images analyzed: {len(df)}"
)


print("\n" + "=" * 70)
print("WEIGHTING SCHEMES")
print("=" * 70)

for scheme, weights in WEIGHTS.items():

    print(
        f"{scheme:<20}"
        f"Coverage={weights[0]:.2f}, "
        f"Length={weights[1]:.2f}, "
        f"Width={weights[2]:.2f}, "
        f"Branch={weights[3]:.2f}"
    )


# --------------------------------------------------
# Statistics
# --------------------------------------------------

statistics = (
    results.drop(
        columns=["image"]
    )
    .describe()
    .T
)

statistics.to_csv(
    OUTPUT_DIR
    / "sensitivity_statistics.csv"
)


print("\n" + "=" * 70)
print("CSI STATISTICS")
print("=" * 70)

print(
    statistics[
        [
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max"
        ]
    ].to_string()
)


# --------------------------------------------------
# Correlation between CSI schemes
# --------------------------------------------------

scheme_columns = [
    column
    for column in results.columns
    if column.startswith("CSI_")
]

correlation = (
    results[
        scheme_columns
    ]
    .corr(
        method="spearman"
    )
)

correlation.to_csv(
    OUTPUT_DIR
    / "scheme_rank_correlation.csv"
)


print("\n" + "=" * 70)
print("SPEARMAN RANK CORRELATION")
print("=" * 70)

print(
    correlation.round(4).to_string()
)


# --------------------------------------------------
# Rank stability
# --------------------------------------------------

ranks = pd.DataFrame(
    index=results.index
)

for column in scheme_columns:

    ranks[column] = (
        results[column]
        .rank(
            method="average"
        )
    )


rank_correlation = (
    ranks.corr(
        method="spearman"
    )
)


rank_correlation.to_csv(
    OUTPUT_DIR
    / "rank_stability.csv"
)


print("\n" + "=" * 70)
print("RANK STABILITY")
print("=" * 70)

print(
    rank_correlation.round(4).to_string()
)


# --------------------------------------------------
# Difference from baseline A
# --------------------------------------------------

baseline = results[
    "CSI_A_equal"
]

difference_rows = []

for scheme in WEIGHTS:

    if scheme == "A_equal":
        continue

    current = results[
        f"CSI_{scheme}"
    ]

    difference = (
        current - baseline
    )

    difference_rows.append({

        "scheme": scheme,

        "mean_absolute_difference":
            difference.abs().mean(),

        "median_absolute_difference":
            difference.abs().median(),

        "maximum_absolute_difference":
            difference.abs().max()
    })


difference_df = pd.DataFrame(
    difference_rows
)

difference_df.to_csv(
    OUTPUT_DIR
    / "difference_from_baseline.csv",
    index=False
)


print("\n" + "=" * 70)
print("DIFFERENCE FROM EQUAL-WEIGHT BASELINE")
print("=" * 70)

print(
    difference_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# Save complete results
# --------------------------------------------------

results.to_csv(
    OUTPUT_DIR
    / "csi_sensitivity_results.csv",
    index=False
)


# --------------------------------------------------
# Save ranks
# --------------------------------------------------

ranks_output = results[
    ["image"]
].copy()

for column in scheme_columns:

    ranks_output[
        column.replace(
            "CSI_",
            "Rank_"
        )
    ] = ranks[column]


ranks_output.to_csv(
    OUTPUT_DIR
    / "csi_sensitivity_ranks.csv",
    index=False
)


# --------------------------------------------------
# Final message
# --------------------------------------------------

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    OUTPUT_DIR
    / "sensitivity_statistics.csv"
)

print(
    OUTPUT_DIR
    / "scheme_rank_correlation.csv"
)

print(
    OUTPUT_DIR
    / "rank_stability.csv"
)

print(
    OUTPUT_DIR
    / "difference_from_baseline.csv"
)

print(
    OUTPUT_DIR
    / "csi_sensitivity_results.csv"
)

print(
    OUTPUT_DIR
    / "csi_sensitivity_ranks.csv"
)

print(
    "\nCSI sensitivity analysis completed successfully."
)