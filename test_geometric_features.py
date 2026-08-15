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


for filename in MASKS:

    path = (
        f"outputs/segmentation/{filename}"
    )

    print("\n" + "=" * 50)
    print(f"Processing: {filename}")
    print("=" * 50)

    mask = load_crack_mask(path)

    # Width features
    width_features = calculate_width_features(
        mask
    )

    # Coverage
    coverage = calculate_crack_coverage(
        mask
    )

    # Crack area
    area = calculate_crack_area(
        mask
    )
    orientation = calculate_crack_orientation(
        mask
    )

    print(
        f"Average width: "
        f"{width_features['average_width_pixels']:.2f} pixels"
    )

    print(
        f"Maximum width: "
        f"{width_features['maximum_width_pixels']:.2f} pixels"
    )

    print(
        f"Crack coverage: "
        f"{coverage:.4f}%"
    )

    print(
        f"Crack area: "
        f"{area} pixels"
    )
    print(
        f"Dominant orientation: "
        f"{orientation:.2f} degrees"
    )



print(
    "\nGeometric feature test completed."
)