import csv

from geometric_features import (
    load_crack_mask,
    calculate_width_features,
    calculate_crack_coverage,
    calculate_crack_area,
    calculate_crack_orientation
)


MASKS = [
    "crack1_shape_filtered.png",
    "crack2_shape_filtered.png"
]


OUTPUT_PATH = (
    "outputs/features/geometric_features.csv"
)


rows = []


for filename in MASKS:

    path = (
        f"outputs/segmentation/{filename}"
    )

    mask = load_crack_mask(path)

    width_features = (
        calculate_width_features(mask)
    )

    coverage = (
        calculate_crack_coverage(mask)
    )

    area = (
        calculate_crack_area(mask)
    )

    orientation = (
        calculate_crack_orientation(mask)
    )

    rows.append({
        "image": filename.replace(
            "_shape_filtered.png",
            ""
        ),

        "average_width_pixels":
            round(
                width_features[
                    "average_width_pixels"
                ],
                2
            ),

        "maximum_width_pixels":
            round(
                width_features[
                    "maximum_width_pixels"
                ],
                2
            ),

        "crack_coverage_percent":
            round(
                coverage,
                4
            ),

        "crack_area_pixels":
            area,

        "dominant_orientation_degrees":
            round(
                orientation,
                2
            )
    })


fieldnames = [
    "image",
    "average_width_pixels",
    "maximum_width_pixels",
    "crack_coverage_percent",
    "crack_area_pixels",
    "dominant_orientation_degrees"
]


with open(
    OUTPUT_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)


print(
    "Geometric features saved successfully."
)

print(
    f"Output: {OUTPUT_PATH}"
)