import csv
import networkx as nx

from feature_extraction import (
    load_skeleton,
    get_clustered_endpoints,
    get_clustered_branch_points,
    build_skeleton_graph,
    calculate_skeleton_length,
    calculate_branch_density,
    calculate_endpoint_density
)


SKELETONS = [
    "crack1_shape_skeleton.png",
    "crack2_shape_skeleton.png"
]


OUTPUT_PATH = (
    "outputs/features/graph_features.csv"
)


rows = []


for filename in SKELETONS:

    path = (
        f"outputs/segmentation/{filename}"
    )

    skeleton = load_skeleton(path)

    endpoints = get_clustered_endpoints(
        skeleton,
        radius=3
    )

    branch_points = get_clustered_branch_points(
        skeleton,
        radius=3
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

    rows.append({
        "image": filename.replace(
            "_shape_skeleton.png",
            ""
        ),
        "endpoint_count": len(
            endpoints
        ),
        "branch_point_count": len(
            branch_points
        ),
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
            )
    })


fieldnames = [
    "image",
    "endpoint_count",
    "branch_point_count",
    "connected_components",
    "graph_nodes",
    "graph_edges",
    "skeleton_length_pixels",
    "branch_density",
    "endpoint_density"
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
    "Graph features saved successfully."
)

print(
    f"Output: {OUTPUT_PATH}"
)