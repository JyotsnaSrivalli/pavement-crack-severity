import pandas as pd


GRAPH_PATH = (
    "outputs/features/graph_features.csv"
)

GEOMETRIC_PATH = (
    "outputs/features/geometric_features.csv"
)

OUTPUT_PATH = (
    "outputs/features/combined_features.csv"
)


# Load Jassi's graph features
graph_features = pd.read_csv(
    GRAPH_PATH
)

# Load Valli's geometric features
geometric_features = pd.read_csv(
    GEOMETRIC_PATH
)


# Combine using the image name
combined_features = pd.merge(
    graph_features,
    geometric_features,
    on="image",
    how="inner"
)


# Save combined features
combined_features.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "Combined features saved successfully."
)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    "\nCombined feature table:"
)

print(
    combined_features.to_string(
        index=False
    )
)