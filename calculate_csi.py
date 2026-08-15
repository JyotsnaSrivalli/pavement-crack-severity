import pandas as pd


INPUT_PATH = (
    "outputs/features/combined_features.csv"
)

OUTPUT_PATH = (
    "outputs/features/csi_results.csv"
)


# Load combined feature table
df = pd.read_csv(INPUT_PATH)


# Features selected for the proposed CSI
CSI_FEATURES = [
    "skeleton_length_pixels",
    "average_width_pixels",
    "maximum_width_pixels",
    "crack_coverage_percent",
    "branch_density",
    "connected_components"
]


# Proposed weights
WEIGHTS = {
    "skeleton_length_pixels": 0.25,
    "average_width_pixels": 0.20,
    "maximum_width_pixels": 0.20,
    "crack_coverage_percent": 0.20,
    "branch_density": 0.10,
    "connected_components": 0.05
}


# Normalize each CSI feature
# using the complete dataset
normalized = pd.DataFrame()

for feature in CSI_FEATURES:

    minimum = df[feature].min()
    maximum = df[feature].max()

    if maximum == minimum:
        normalized[feature] = 0.0

    else:
        normalized[feature] = (
            (df[feature] - minimum)
            / (maximum - minimum)
        )


# Calculate weighted CSI
df["CSI"] = 0.0

for feature in CSI_FEATURES:

    df["CSI"] += (
        normalized[feature]
        * WEIGHTS[feature]
    )


# Convert from 0–1 to 0–100
df["CSI"] = (
    df["CSI"] * 100
)


# Proposed severity categories
def classify_severity(csi):

    if csi < 33.33:
        return "Low"

    elif csi < 66.67:
        return "Moderate"

    else:
        return "Severe"


df["severity"] = df["CSI"].apply(
    classify_severity
)


# Round CSI
df["CSI"] = df["CSI"].round(2)


# Save results
df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "CSI calculation completed."
)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    "\nCSI Results:"
)

print(
    df[
        [
            "image",
            "CSI",
            "severity"
        ]
    ].to_string(index=False)
)