import cv2
import numpy as np
import networkx as nx


def load_skeleton(path):
    """
    Jassi — Load a binary crack skeleton.
    """

    skeleton_image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if skeleton_image is None:
        raise FileNotFoundError(
            f"Could not load skeleton: {path}"
        )

    return skeleton_image > 0


def get_neighbor_count(skeleton, x, y):
    """
    Count 8-connected skeleton neighbors.
    """

    neighborhood = skeleton[
        y - 1:y + 2,
        x - 1:x + 2
    ]

    return int(
        np.sum(neighborhood) - 1
    )


def detect_endpoints(skeleton):
    """
    Detect raw endpoint pixels.
    """

    endpoints = []

    height, width = skeleton.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            if not skeleton[y, x]:
                continue

            if get_neighbor_count(
                skeleton,
                x,
                y
            ) == 1:

                endpoints.append(
                    (x, y)
                )

    return endpoints


def detect_branch_points(skeleton):
    """
    Detect raw branch-point pixels.
    """

    branch_points = []

    height, width = skeleton.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            if not skeleton[y, x]:
                continue

            if get_neighbor_count(
                skeleton,
                x,
                y
            ) >= 3:

                branch_points.append(
                    (x, y)
                )

    return branch_points


def cluster_points(points, radius=3):
    """
    Group nearby detection pixels into
    physical endpoints or branch points.
    """

    if not points:
        return []

    points = np.array(
        points,
        dtype=np.float32
    )

    used = np.zeros(
        len(points),
        dtype=bool
    )

    clusters = []

    for i in range(len(points)):

        if used[i]:
            continue

        cluster = [i]
        used[i] = True

        changed = True

        while changed:

            changed = False

            for j in range(len(points)):

                if used[j]:
                    continue

                for index in cluster:

                    distance = np.linalg.norm(
                        points[j]
                        - points[index]
                    )

                    if distance <= radius:

                        cluster.append(j)
                        used[j] = True
                        changed = True
                        break

        center = np.mean(
            points[cluster],
            axis=0
        )

        clusters.append(
            (
                int(round(center[0])),
                int(round(center[1]))
            )
        )

    return clusters


def get_clustered_endpoints(
    skeleton,
    radius=3
):
    """
    Return clustered endpoints.
    """

    return cluster_points(
        detect_endpoints(skeleton),
        radius
    )


def get_clustered_branch_points(
    skeleton,
    radius=3
):
    """
    Return clustered branch points.
    """

    return cluster_points(
        detect_branch_points(skeleton),
        radius
    )


def build_skeleton_graph(skeleton):
    """
    Build a weighted NetworkX graph.

    Nodes = skeleton pixels.
    Edges = neighboring pixels.
    Weight = Euclidean pixel distance.
    """

    graph = nx.Graph()

    height, width = skeleton.shape

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]

    # Add nodes
    for y in range(height):
        for x in range(width):

            if skeleton[y, x]:

                graph.add_node(
                    (x, y),
                    x=x,
                    y=y
                )

    # Add edges
    for y in range(height):
        for x in range(width):

            if not skeleton[y, x]:
                continue

            current = (x, y)

            for dy, dx in neighbors:

                nx_pos = x + dx
                ny_pos = y + dy

                if (
                    0 <= nx_pos < width
                    and
                    0 <= ny_pos < height
                    and
                    skeleton[ny_pos, nx_pos]
                ):

                    neighbor = (
                        nx_pos,
                        ny_pos
                    )

                    if current < neighbor:

                        distance = np.sqrt(
                            dx * dx +
                            dy * dy
                        )

                        graph.add_edge(
                            current,
                            neighbor,
                            weight=distance
                        )

    return graph


def calculate_skeleton_length(graph):
    """
    Calculate total weighted skeleton length.
    """

    return sum(
        data["weight"]
        for _, _, data
        in graph.edges(
            data=True
        )
    )


def calculate_branch_density(
    branch_point_count,
    skeleton_length
):
    """
    Branch points per pixel of skeleton length.
    """

    if skeleton_length == 0:
        return 0.0

    return (
        branch_point_count
        / skeleton_length
    )


def calculate_endpoint_density(
    endpoint_count,
    skeleton_length
):
    """
    Endpoints per pixel of skeleton length.
    """

    if skeleton_length == 0:
        return 0.0

    return (
        endpoint_count
        / skeleton_length
    )