import cv2

from segmentation import (
    apply_strong_response_threshold,
    clean_binary_mask,
    remove_small_components,
    filter_crack_like_components
)

from skeletonization import skeletonize_crack


IMAGES = [
    "crack1",
    "crack2"
]


for name in IMAGES:

    print(f"\nProcessing {name}...")

    input_path = (
        f"outputs/gabor/{name}_gabor.png"
    )

    gabor_image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if gabor_image is None:
        raise FileNotFoundError(
            f"Could not load: {input_path}"
        )

    # 1. Strong Gabor response
    strong_mask = (
        apply_strong_response_threshold(
            gabor_image,
            percentile=85
        )
    )

    # 2. Morphological cleaning
    cleaned_mask = clean_binary_mask(
        strong_mask
    )

    # 3. Remove tiny components
    component_mask = remove_small_components(
        cleaned_mask,
        min_area=200
    )

    # 4. Keep crack-like shapes
    final_mask = filter_crack_like_components(
        component_mask,
        min_area=200
    )

    # Save final mask
    mask_path = (
        f"outputs/segmentation/"
        f"{name}_shape_filtered.png"
    )

    cv2.imwrite(
        mask_path,
        final_mask
    )

    # 5. Skeletonization
    skeleton = skeletonize_crack(
        final_mask
    )

    skeleton_path = (
        f"outputs/segmentation/"
        f"{name}_shape_skeleton.png"
    )

    cv2.imwrite(
        skeleton_path,
        skeleton
    )

    print(f"{name} completed.")
    print(f"Mask: {mask_path}")
    print(f"Skeleton: {skeleton_path}")


print(
    "\nJASSI FINAL PIPELINE COMPLETED."
)