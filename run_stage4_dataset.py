import csv
import time
from pathlib import Path

import networkx as nx

from config import DATASET_DIR, OUTPUT_DIR

from skeletonization import skeletonize_crack

from feature_extraction import (
    get_clustered_endpoints,
    get_clustered_branch_points,
    build_skeleton_graph,
    calculate_skeleton_length,
    calculate_branch_density,
    calculate_endpoint_density
)

from geometric_features import (
    load_crack_mask,
    calculate_width_features,
    calculate_crack_coverage,
    calculate_crack_area,
    calculate_crack_orientation
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_PATH = (
    OUTPUT_DIR
    / "stage3"
    / "features"
    / "crack500_features.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Find all ground-truth PNG masks
# --------------------------------------------------

mask_files = sorted(
    DATASET_DIR.glob("*.png")
)

if not mask_files:
    raise FileNotFoundError(
        f"No PNG masks found in: {DATASET_DIR}"
    )


print("=" * 70)
print("CRACK500 FULL GROUND-TRUTH FEATURE EXTRACTION")
print("=" * 70)

print(
    f"Dataset folder: {DATASET_DIR}"
)

print(
    f"PNG masks found: {len(mask_files)}"
)

print(
    "\nProcessing masks..."
)


# --------------------------------------------------
# Feature columns
# --------------------------------------------------

fieldnames = [
    "image",

    # Graph / structural
    "endpoint_count",
    "branch_point_count",
    "connected_components",
    "graph_nodes",
    "graph_edges",
    "skeleton_length_pixels",
    "branch_density",
    "endpoint_density",

    # Geometric
    "average_width_pixels",
    "maximum_width_pixels",
    "crack_coverage_percent",
    "crack_area_pixels",
    "dominant_orientation_degrees"
]


# --------------------------------------------------
# Process dataset
# --------------------------------------------------

rows = []

start_time = time.time()

failed = []


for index, mask_path in enumerate(
    mask_files,
    start=1
):

    image_name = mask_path.stem

    try:

        # ------------------------------------------
        # Load ground-truth mask
        # ------------------------------------------

        mask = load_crack_mask(
            str(mask_path)
        )

        # ------------------------------------------
        # Skeletonization
        # ------------------------------------------

        skeleton_image = (
            skeletonize_crack(mask)
        )

        skeleton = (
            skeleton_image > 0
        )

        # ------------------------------------------
        # Graph features
        # ------------------------------------------

        endpoints = (
            get_clustered_endpoints(
                skeleton,
                radius=3
            )
        )

        branch_points = (
            get_clustered_branch_points(
                skeleton,
                radius=3
            )
        )

        graph = build_skeleton_graph(
            skeleton
        )

        connected_components = (
            nx.number_connected_components(
                graph
            )
        )

        skeleton_length = (
            calculate_skeleton_length(
                graph
            )
        )

        branch_density = (
            calculate_branch_density(
                len(branch_points),
                skeleton_length
            )
        )

        endpoint_density = (
            calculate_endpoint_density(
                len(endpoints),
                skeleton_length
            )
        )

        # ------------------------------------------
        # Geometric features
        # ------------------------------------------

        width_features = (
            calculate_width_features(
                mask
            )
        )

        coverage = (
            calculate_crack_coverage(
                mask
            )
        )

        area = (
            calculate_crack_area(
                mask
            )
        )

        orientation = (
            calculate_crack_orientation(
                mask
            )
        )

        # ------------------------------------------
        # Store row
        # ------------------------------------------

        rows.append({

            "image":
                image_name,

            "endpoint_count":
                len(endpoints),

            "branch_point_count":
                len(branch_points),

            "connected_components":
                connected_components,

            "graph_nodes":
                graph.number_of_nodes(),

            "graph_edges":
                graph.number_of_edges(),

            "skeleton_length_pixels":
                round(
                    skeleton_length,
                    2
                ),

            "branch_density":
                round(
                    branch_density,
                    6
                ),

            "endpoint_density":
                round(
                    endpoint_density,
                    6
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


        # ------------------------------------------
        # Progress
        # ------------------------------------------

        if (
            index == 1
            or index % 10 == 0
            or index == len(mask_files)
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"[{index}/{len(mask_files)}] "
                f"{image_name} "
                f"| elapsed: "
                f"{elapsed:.1f}s"
            )


    except Exception as error:

        failed.append({
            "image": image_name,
            "error": str(error)
        })

        print(
            f"\nERROR processing "
            f"{image_name}: {error}"
        )


# --------------------------------------------------
# Save feature table
# --------------------------------------------------

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


# --------------------------------------------------
# Summary
# --------------------------------------------------

elapsed = (
    time.time()
    - start_time
)


print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print(
    f"Total masks found: {len(mask_files)}"
)

print(
    f"Successfully processed: {len(rows)}"
)

print(
    f"Failed: {len(failed)}"
)

print(
    f"Time taken: {elapsed:.2f} seconds"
)

print(
    f"Output: {OUTPUT_PATH}"
)


# --------------------------------------------------
# Failed image report
# --------------------------------------------------

if failed:

    failed_path = (
        OUTPUT_PATH.parent
        / "failed_images.csv"
    )

    with open(
        failed_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "error"
            ]
        )

        writer.writeheader()

        writer.writerows(
            failed
        )

    print(
        f"\nFailed-image report: "
        f"{failed_path}"
    )

else:

    print(
        "\nAll images processed successfully."
    )