import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "outputs/stage3/features/crack500_features.csv"
)


FEATURES = [
    "crack_coverage_percent",
    "skeleton_length_pixels",
    "maximum_width_pixels",
    "branch_density"
]


df = pd.read_csv(INPUT_PATH)


print("=" * 70)
print("CSI FEATURE DISTRIBUTIONS")
print("=" * 70)

for feature in FEATURES:

    print("\n" + "-" * 70)
    print(feature)

    values = df[feature]

    percentiles = values.quantile(
        [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]
    )

    print(
        percentiles.to_string()
    )


print("\n" + "=" * 70)
print("CSI FEATURE DISTRIBUTION ANALYSIS COMPLETED")
print("=" * 70)