import cv2
import numpy as np

from config import DATASET_DIR
from segmentation import clean_binary_mask


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_COUNT = 10

PERCENTILES = [90, 92, 95]

MIN_AREAS = [200, 500, 1000]

MIN_ELONGATIONS = [2.0, 3.0]

MAX_COMPACTNESS = [0.65, 0.50]


OUTPUT_DIR = (
    "outputs/stage3/evaluation/tuning_10"
)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

import os

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Load first 10 CRACK500 images
# --------------------------------------------------

image_files = sorted(
    DATASET_DIR.glob("*.jpg")
)[:TEST_COUNT]


if not image_files:
    raise FileNotFoundError(
        "No JPG images found."
    )


print("=" * 60)
print("10-IMAGE SEGMENTATION TUNING")
print("=" * 60)

print(
    f"Images used: {len(image_files)}"
)


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

def calculate_metrics(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    ).astype(np.uint8)

    ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)

    tp = np.sum(
        (prediction == 1)
        & (ground_truth == 1)
    )

    fp = np.sum(
        (prediction == 1)
        & (ground_truth == 0)
    )

    fn = np.sum(
        (prediction == 0)
        & (ground_truth == 1)
    )

    union = tp + fp + fn

    if union == 0:
        iou = 1.0
    else:
        iou = tp / union

    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * tp
            / dice_denominator
        )

    precision_denominator = (
        tp + fp
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            tp
            / precision_denominator
        )

    recall_denominator = (
        tp + fn
    )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            tp
            / recall_denominator
        )

    return (
        iou,
        dice,
        precision,
        recall
    )


# --------------------------------------------------
# Improved crack-shape filtering
# --------------------------------------------------

def filter_crack_components(
    binary_mask,
    min_area,
    min_elongation,
    max_compactness
):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    filtered_mask = np.zeros_like(
        binary_mask
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < min_area:
            continue

        component = (
            labels == label
        ).astype(np.uint8) * 255

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

        # ------------------------------------------
        # Compactness
        # ------------------------------------------

        compactness = (
            4 * np.pi * area
        ) / (
            perimeter * perimeter
        )

        if compactness > max_compactness:
            continue

        # ------------------------------------------
        # Elongation
        # ------------------------------------------

        if len(contour) < 5:
            # Very small/degenerate contour.
            # Let compactness decide.
            elongation = 1.0
        else:

            ellipse = cv2.fitEllipse(
                contour
            )

            (_, _), (axis1, axis2), _ = (
                ellipse
            )

            major_axis = max(
                axis1,
                axis2
            )

            minor_axis = min(
                axis1,
                axis2
            )

            if minor_axis <= 0:
                elongation = 999.0
            else:
                elongation = (
                    major_axis
                    / minor_axis
                )

        if elongation < min_elongation:
            continue

        filtered_mask[
            labels == label
        ] = 255

    return filtered_mask


# --------------------------------------------------
# Store results
# --------------------------------------------------

results = []


# --------------------------------------------------
# Test parameter combinations
# --------------------------------------------------

for percentile in PERCENTILES:

    for min_area in MIN_AREAS:

        for min_elongation in (
            MIN_ELONGATIONS
        ):

            for max_compactness in (
                MAX_COMPACTNESS
            ):

                print(
                    "\n"
                    + "-" * 60
                )

                print(
                    f"Percentile: {percentile}"
                )

                print(
                    f"Min area: {min_area}"
                )

                print(
                    f"Min elongation: "
                    f"{min_elongation}"
                )

                print(
                    f"Max compactness: "
                    f"{max_compactness}"
                )

                image_metrics = []

                # ----------------------------------
                # Process all 10 images
                # ----------------------------------

                for image_path in (
                    image_files
                ):

                    gabor_path = (
                        DATASET_DIR
                        / f"{image_path.stem}.jpg"
                    )

                    # ----------------------------------
                    # We need Gabor from the image.
                    # Import preprocessing here.
                    # ----------------------------------

                    from preprocessing import (
                        load_image,
                        to_grayscale,
                        apply_clahe,
                        apply_gabor_filter_bank
                    )

                    image = load_image(
                        str(gabor_path)
                    )

                    gray = to_grayscale(
                        image
                    )

                    clahe = apply_clahe(
                        gray
                    )

                    gabor = (
                        apply_gabor_filter_bank(
                            clahe
                        )
                    )

                    # ----------------------------------
                    # Threshold
                    # ----------------------------------

                    blurred = cv2.GaussianBlur(
                        gabor,
                        (5, 5),
                        0
                    )

                    threshold_value = (
                        np.percentile(
                            blurred,
                            percentile
                        )
                    )

                    binary_mask = np.where(
                        blurred
                        >= threshold_value,
                        255,
                        0
                    ).astype(np.uint8)

                    # ----------------------------------
                    # Existing morphological cleaning
                    # ----------------------------------

                    cleaned = (
                        clean_binary_mask(
                            binary_mask
                        )
                    )

                    # ----------------------------------
                    # Improved shape filtering
                    # ----------------------------------

                    final_mask = (
                        filter_crack_components(
                            cleaned,
                            min_area,
                            min_elongation,
                            max_compactness
                        )
                    )

                    # ----------------------------------
                    # Ground truth
                    # ----------------------------------

                    ground_truth_path = (
                        DATASET_DIR
                        / f"{image_path.stem}.png"
                    )

                    ground_truth = cv2.imread(
                        str(ground_truth_path),
                        cv2.IMREAD_GRAYSCALE
                    )

                    if ground_truth is None:
                        raise FileNotFoundError(
                            f"Could not load mask: "
                            f"{ground_truth_path}"
                        )

                    # ----------------------------------
                    # Metrics
                    # ----------------------------------

                    metrics = (
                        calculate_metrics(
                            final_mask,
                            ground_truth
                        )
                    )

                    image_metrics.append(
                        metrics
                    )

                # ----------------------------------
                # Average over 10 images
                # ----------------------------------

                average_metrics = (
                    np.mean(
                        image_metrics,
                        axis=0
                    )
                )

                avg_iou = (
                    average_metrics[0]
                )

                avg_dice = (
                    average_metrics[1]
                )

                avg_precision = (
                    average_metrics[2]
                )

                avg_recall = (
                    average_metrics[3]
                )

                print(
                    f"Average IoU: "
                    f"{avg_iou:.4f}"
                )

                print(
                    f"Average Dice: "
                    f"{avg_dice:.4f}"
                )

                print(
                    f"Average Precision: "
                    f"{avg_precision:.4f}"
                )

                print(
                    f"Average Recall: "
                    f"{avg_recall:.4f}"
                )

                results.append({
                    "percentile": percentile,
                    "min_area": min_area,
                    "min_elongation":
                        min_elongation,
                    "max_compactness":
                        max_compactness,
                    "iou": avg_iou,
                    "dice": avg_dice,
                    "precision":
                        avg_precision,
                    "recall":
                        avg_recall
                })


# --------------------------------------------------
# Find best configuration
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL 10-IMAGE TUNING RESULTS")
print("=" * 60)


# Sort by IoU
results.sort(
    key=lambda x: x["iou"],
    reverse=True
)


print(
    "\nTop configurations:"
)


for result in results[:10]:

    print(
        f"\nPercentile: "
        f"{result['percentile']}"
    )

    print(
        f"Min area: "
        f"{result['min_area']}"
    )

    print(
        f"Min elongation: "
        f"{result['min_elongation']}"
    )

    print(
        f"Max compactness: "
        f"{result['max_compactness']}"
    )

    print(
        f"IoU: "
        f"{result['iou']:.4f}"
    )

    print(
        f"Dice: "
        f"{result['dice']:.4f}"
    )

    print(
        f"Precision: "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{result['recall']:.4f}"
    )


best = results[0]


print("\n" + "=" * 60)
print("BEST CONFIGURATION")
print("=" * 60)

print(
    f"Percentile: "
    f"{best['percentile']}"
)

print(
    f"Minimum area: "
    f"{best['min_area']}"
)

print(
    f"Minimum elongation: "
    f"{best['min_elongation']}"
)

print(
    f"Maximum compactness: "
    f"{best['max_compactness']}"
)

print(
    f"Average IoU: "
    f"{best['iou']:.4f}"
)

print(
    f"Average Dice: "
    f"{best['dice']:.4f}"
)

print(
    f"Average Precision: "
    f"{best['precision']:.4f}"
)

print(
    f"Average Recall: "
    f"{best['recall']:.4f}"
)