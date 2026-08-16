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
    "outputs/stage3/features/analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("CRACK500 FEATURE ANALYSIS")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# Feature columns
# --------------------------------------------------

FEATURES = [
    "endpoint_count",
    "branch_point_count",
    "connected_components",
    "graph_nodes",
    "graph_edges",
    "skeleton_length_pixels",
    "branch_density",
    "endpoint_density",
    "average_width_pixels",
    "maximum_width_pixels",
    "crack_coverage_percent",
    "crack_area_pixels",
    "dominant_orientation_degrees"
]


# --------------------------------------------------
# Check missing values
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df[FEATURES].isnull().sum()

print(missing.to_string())


# --------------------------------------------------
# Check constant features
# --------------------------------------------------

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for feature in FEATURES:

    if df[feature].nunique() <= 1:
        constant_features.append(feature)

if constant_features:
    for feature in constant_features:
        print(feature)
else:
    print("No constant features found.")


# --------------------------------------------------
# Descriptive statistics
# --------------------------------------------------

statistics = df[FEATURES].describe().T

statistics["median"] = (
    df[FEATURES].median()
)

statistics["variance"] = (
    df[FEATURES].var()
)

statistics["coefficient_of_variation"] = (
    statistics["std"]
    / statistics["mean"].replace(0, np.nan)
)

statistics = statistics[
    [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "median",
        "50%",
        "75%",
        "max",
        "variance",
        "coefficient_of_variation"
    ]
]

statistics.to_csv(
    OUTPUT_DIR
    / "feature_statistics.csv"
)

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    statistics.to_string()
)


# --------------------------------------------------
# Correlation matrix
# --------------------------------------------------

correlation = (
    df[FEATURES]
    .corr(method="pearson")
)

correlation.to_csv(
    OUTPUT_DIR
    / "feature_correlation.csv"
)

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(
    correlation.round(3).to_string()
)


# --------------------------------------------------
# Highly correlated feature pairs
# --------------------------------------------------

print("\n" + "=" * 70)
print("HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 70)

pairs = []

for i in range(len(FEATURES)):

    for j in range(i + 1, len(FEATURES)):

        feature_a = FEATURES[i]
        feature_b = FEATURES[j]

        value = correlation.loc[
            feature_a,
            feature_b
        ]

        if abs(value) >= 0.80:

            pairs.append({
                "feature_1": feature_a,
                "feature_2": feature_b,
                "correlation": value
            })

if pairs:

    pairs_df = pd.DataFrame(
        pairs
    )

    pairs_df = pairs_df.sort_values(
        "correlation",
        key=lambda x: abs(x),
        ascending=False
    )

    print(
        pairs_df.to_string(
            index=False
        )
    )

    pairs_df.to_csv(
        OUTPUT_DIR
        / "highly_correlated_pairs.csv",
        index=False
    )

else:

    print(
        "No feature pairs with "
        "|correlation| >= 0.80."
    )


# --------------------------------------------------
# Feature ranking by variability
# --------------------------------------------------

variability = statistics[
    [
        "mean",
        "std",
        "coefficient_of_variation"
    ]
].copy()

variability = variability.sort_values(
    "coefficient_of_variation",
    ascending=False
)

variability.to_csv(
    OUTPUT_DIR
    / "feature_variability.csv"
)

print("\n" + "=" * 70)
print("FEATURE VARIABILITY")
print("=" * 70)

print(
    variability.to_string()
)


# --------------------------------------------------
# Final output summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("ANALYSIS FILES CREATED")
print("=" * 70)

print(
    OUTPUT_DIR
    / "feature_statistics.csv"
)

print(
    OUTPUT_DIR
    / "feature_correlation.csv"
)

print(
    OUTPUT_DIR
    / "feature_variability.csv"
)

if pairs:
    print(
        OUTPUT_DIR
        / "highly_correlated_pairs.csv"
    )

print("\nFeature analysis completed successfully.")