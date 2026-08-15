import pandas as pd


INPUT_PATH = (
    "outputs/features/combined_features.csv"
)

OUTPUT_PATH = (
    "outputs/features/normalized_features.csv"
)


# Load combined features
df = pd.read_csv(INPUT_PATH)


# Keep image names separately
image_names = df["image"]


# Select numerical features
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
    "crack_area_pixels"
]


# Min-Max normalization
for column in feature_columns:

    minimum = df[column].min()
    maximum = df[column].max()

    if maximum == minimum:
        df[column] = 0.0

    else:
        df[column] = (
            (df[column] - minimum)
            / (maximum - minimum)
        )


# Save normalized features
df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "Feature normalization completed."
)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    "\nNormalized feature table:"
)

print(
    df.to_string(index=False)
)