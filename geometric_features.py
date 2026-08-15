import cv2
import numpy as np


def load_crack_mask(path):
    """
    Load a binary crack segmentation mask.

    Crack pixels should be white (255).
    Background should be black (0).
    """

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not load crack mask: {path}"
        )

    # Ensure binary mask
    binary_mask = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    return binary_mask


def calculate_width_features(mask):
    """
    Estimate crack width using a distance transform.

    For a crack pixel, the distance transform gives
    the distance to the nearest background pixel.

    Local crack width ≈ 2 × distance.
    """

    # Distance transform requires foreground
    # pixels to be non-zero.
    distance = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    crack_distances = distance[
        mask > 0
    ]

    if len(crack_distances) == 0:
        return {
            "average_width_pixels": 0.0,
            "maximum_width_pixels": 0.0
        }

    widths = 2 * crack_distances

    average_width = np.mean(
        widths
    )

    maximum_width = np.max(
        widths
    )

    return {
        "average_width_pixels":
            float(average_width),

        "maximum_width_pixels":
            float(maximum_width)
    }
def calculate_crack_coverage(mask):
    """
    Calculate the percentage of the image
    occupied by the detected crack mask.
    """

    total_pixels = mask.size

    crack_pixels = np.count_nonzero(
        mask
    )

    if total_pixels == 0:
        return 0.0

    coverage = (
        crack_pixels
        / total_pixels
    ) * 100

    return float(coverage)
def calculate_crack_area(mask):
    """
    Calculate the number of pixels belonging
    to the detected crack region.
    """

    crack_pixels = np.count_nonzero(
        mask
    )

    return int(crack_pixels)
def calculate_crack_orientation(mask):
    """
    Calculate the dominant crack orientation using PCA.

    The orientation is measured in degrees from 0 to 180.
    """

    # Get coordinates of crack pixels
    y, x = np.where(mask > 0)

    # Need at least two points for PCA
    if len(x) < 2:
        return 0.0

    # Create coordinate matrix
    coordinates = np.column_stack(
        (x, y)
    ).astype(np.float64)

    # Center the coordinates
    centered = (
        coordinates
        - np.mean(coordinates, axis=0)
    )

    # Calculate covariance matrix
    covariance = np.cov(
        centered,
        rowvar=False
    )

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Direction of maximum variance
    principal_vector = eigenvectors[
        :, np.argmax(eigenvalues)
    ]

    # Calculate angle
    angle = np.degrees(
        np.arctan2(
            principal_vector[1],
            principal_vector[0]
        )
    )

    # Convert to 0–180 degrees
    angle = angle % 180

    return float(angle)