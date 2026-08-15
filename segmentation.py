import cv2
import numpy as np


def apply_strong_response_threshold(
    gabor_image,
    percentile=85
):
    """
    Jassi — Gabor response thresholding.
    """

    blurred = cv2.GaussianBlur(
        gabor_image,
        (5, 5),
        0
    )

    threshold_value = np.percentile(
        blurred,
        percentile
    )

    binary_mask = np.where(
        blurred >= threshold_value,
        255,
        0
    ).astype(np.uint8)

    return binary_mask


def clean_binary_mask(binary_mask):
    """
    Jassi — Morphological cleaning.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    opened = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return closed


def remove_small_components(
    binary_mask,
    min_area=200
):
    """
    Remove very small disconnected regions.
    """

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    filtered_mask = np.zeros_like(
        binary_mask
    )

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            filtered_mask[
                labels == label
            ] = 255

    return filtered_mask


def filter_crack_like_components(
    binary_mask,
    min_area=200,
    min_elongation=2.0
):
    """
    Jassi — Shape-based crack filtering.

    Retains components that are sufficiently elongated,
    which helps suppress compact pavement-texture regions.
    """

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    filtered_mask = np.zeros_like(
        binary_mask
    )

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < min_area:
            continue

        component = np.uint8(
            labels == label
        ) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        # Compactness:
        # circle-like objects have higher compactness.
        compactness = (
            4 * np.pi * area
        ) / (
            perimeter * perimeter
        )

        # Crack-like structures are usually less compact.
        if compactness <= 0.65:
            filtered_mask[
                labels == label
            ] = 255

    return filtered_mask