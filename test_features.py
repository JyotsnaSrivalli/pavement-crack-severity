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


for filename in SKELETONS:

    path = (
        f"outputs/segmentation/{filename}"
    )

    print("\n" + "=" * 50)
    print(f"Processing: {filename}")
    print("=" * 50)

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

    print(
        f"Endpoints: {len(endpoints)}"
    )

    print(
        f"Branch points: {len(branch_points)}"
    )

    print(
        f"Graph nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Graph edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Connected components: "
        f"{connected_components}"
    )

    print(
        f"Skeleton length: "
        f"{skeleton_length:.2f} pixels"
    )

    print(
        f"Branch density: "
        f"{branch_density:.6f}"
    )

    print(
        f"Endpoint density: "
        f"{endpoint_density:.6f}"
    )


print(
    "\nGraph feature extraction completed."
)