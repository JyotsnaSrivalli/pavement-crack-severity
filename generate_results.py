import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

CSI_PATH = Path(
    "outputs/stage3/features/final_csi/"
    "crack500_csi_results.csv"
)

FEATURE_PATH = Path(
    "outputs/stage3/features/crack500_features.csv"
)

OUTPUT_DIR = Path(
    "outputs/stage3/features/final_csi/results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(CSI_PATH)

features = pd.read_csv(
    FEATURE_PATH
)


print("=" * 70)
print("GENERATING RESEARCH RESULTS")
print("=" * 70)

print(
    f"Images analyzed: {len(df)}"
)


# --------------------------------------------------
# 1. CSI distribution
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["CSI"],
    bins=30
)

plt.xlabel(
    "Proposed CSI"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Distribution of Proposed Crack Severity Index"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "csi_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. CSI by severity group
# --------------------------------------------------

groups = [
    "Low",
    "Moderate",
    "High"
]

group_values = [
    df.loc[
        df["severity"] == group,
        "CSI"
    ]
    for group in groups
]

plt.figure(figsize=(8, 6))

plt.boxplot(
    group_values,
    tick_labels=groups
)

plt.xlabel(
    "Relative Severity Group"
)

plt.ylabel(
    "Proposed CSI"
)

plt.title(
    "CSI Distribution Across Relative Severity Groups"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "csi_by_severity.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 3. Feature comparison by severity
# --------------------------------------------------

comparison_features = [
    (
        "crack_coverage_percent",
        "Crack Coverage (%)",
        "coverage_by_severity.png"
    ),
    (
        "skeleton_length_pixels",
        "Skeleton Length (pixels)",
        "length_by_severity.png"
    ),
    (
        "maximum_width_pixels",
        "Maximum Crack Width (pixels)",
        "width_by_severity.png"
    ),
    (
        "branch_density",
        "Branch Density",
        "branch_density_by_severity.png"
    )
]


for feature, label, filename in comparison_features:

    means = (
        df.groupby("severity")[feature]
        .mean()
        .reindex(groups)
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        groups,
        means.values
    )

    plt.xlabel(
        "Relative Severity Group"
    )

    plt.ylabel(
        label
    )

    plt.title(
        f"{label} Across Relative Severity Groups"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 4. Correlation heatmap
# --------------------------------------------------

feature_columns = [
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

correlation = (
    features[feature_columns]
    .corr()
)


plt.figure(
    figsize=(13, 11)
)

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(
    label="Pearson Correlation"
)

plt.xticks(
    range(len(feature_columns)),
    feature_columns,
    rotation=90
)

plt.yticks(
    range(len(feature_columns)),
    feature_columns
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_correlation_heatmap.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 5. CSI component contribution
# --------------------------------------------------

# Recalculate component contributions because
# the main CSI CSV stores the normalized features,
# but not the individual contribution columns.

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


contribution_columns = [
    "coverage_contribution",
    "length_contribution",
    "width_contribution",
    "complexity_contribution"
]

contribution_labels = [
    "Coverage",
    "Skeleton Length",
    "Maximum Width",
    "Branch Density"
]

mean_contributions = (
    df[contribution_columns]
    .mean()
)


plt.figure(figsize=(8, 6))

plt.bar(
    contribution_labels,
    mean_contributions.values
)

plt.xlabel(
    "CSI Component"
)

plt.ylabel(
    "Mean Contribution to CSI"
)

plt.title(
    "Average Contribution of CSI Components"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "csi_component_contributions.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 6. Save research summary
# --------------------------------------------------

summary = pd.DataFrame({

    "metric": [
        "Dataset images",
        "CSI mean",
        "CSI median",
        "CSI standard deviation",
        "CSI minimum",
        "CSI maximum",
        "Low images",
        "Moderate images",
        "High images"
    ],

    "value": [
        len(df),
        df["CSI"].mean(),
        df["CSI"].median(),
        df["CSI"].std(),
        df["CSI"].min(),
        df["CSI"].max(),
        (df["severity"] == "Low").sum(),
        (df["severity"] == "Moderate").sum(),
        (df["severity"] == "High").sum()
    ]
})


summary.to_csv(
    OUTPUT_DIR / "research_summary.csv",
    index=False
)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n" + "=" * 70)
print("RESULT FIGURES CREATED")
print("=" * 70)

print(
    "csi_distribution.png"
)

print(
    "csi_by_severity.png"
)

print(
    "coverage_by_severity.png"
)

print(
    "length_by_severity.png"
)

print(
    "width_by_severity.png"
)

print(
    "branch_density_by_severity.png"
)

print(
    "feature_correlation_heatmap.png"
)

print(
    "csi_component_contributions.png"
)

print(
    "research_summary.csv"
)

print(
    f"\nOutput folder: {OUTPUT_DIR}"
)

print(
    "\nResearch results generated successfully."
)