import cv2

from config import DATASET_DIR, OUTPUT_DIR

from preprocessing import (
    load_image,
    to_grayscale,
    apply_clahe,
    apply_gabor_filter_bank
)

from segmentation import (
    apply_strong_response_threshold,
    clean_binary_mask,
    remove_small_components,
    filter_crack_like_components
)

from skeletonization import skeletonize_crack


# --------------------------------------------------
# Select one real CRACK500 image
# --------------------------------------------------

image_files = sorted(
    DATASET_DIR.glob("*.jpg")
)

if not image_files:
    raise FileNotFoundError(
        "No JPG images found in CRACK500 dataset."
    )

image_path = image_files[0]

name = image_path.stem

print("=" * 60)
print("STAGE 3 SINGLE IMAGE TEST")
print("=" * 60)

print(f"Input image: {image_path}")
print(f"Image name: {name}")


# --------------------------------------------------
# Create output directories
# --------------------------------------------------

stage3_dir = OUTPUT_DIR / "stage3"

preprocessing_dir = (
    stage3_dir / "preprocessing"
)

segmentation_dir = (
    stage3_dir / "segmentation"
)

skeleton_dir = (
    stage3_dir / "skeletons"
)

preprocessing_dir.mkdir(
    parents=True,
    exist_ok=True
)

segmentation_dir.mkdir(
    parents=True,
    exist_ok=True
)

skeleton_dir.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 1. Load image
# --------------------------------------------------

image = load_image(
    str(image_path)
)

print(
    f"Original image shape: "
    f"{image.shape}"
)


# --------------------------------------------------
# 2. Grayscale
# --------------------------------------------------

gray = to_grayscale(
    image
)

cv2.imwrite(
    str(
        preprocessing_dir
        / f"{name}_gray.png"
    ),
    gray
)


# --------------------------------------------------
# 3. CLAHE
# --------------------------------------------------

clahe = apply_clahe(
    gray
)

cv2.imwrite(
    str(
        preprocessing_dir
        / f"{name}_clahe.png"
    ),
    clahe
)


# --------------------------------------------------
# 4. Gabor filtering
# --------------------------------------------------

gabor = apply_gabor_filter_bank(
    clahe
)

cv2.imwrite(
    str(
        preprocessing_dir
        / f"{name}_gabor.png"
    ),
    gabor
)


# --------------------------------------------------
# 5. Strong-response threshold
# --------------------------------------------------

strong_mask = (
    apply_strong_response_threshold(
        gabor,
        percentile=85
    )
)


# --------------------------------------------------
# 6. Morphological cleaning
# --------------------------------------------------

cleaned_mask = (
    clean_binary_mask(
        strong_mask
    )
)


# --------------------------------------------------
# 7. Remove small components
# --------------------------------------------------

component_mask = (
    remove_small_components(
        cleaned_mask,
        min_area=200
    )
)


# --------------------------------------------------
# 8. Crack-shape filtering
# --------------------------------------------------

final_mask = (
    filter_crack_like_components(
        component_mask,
        min_area=200
    )
)


mask_path = (
    segmentation_dir
    / f"{name}_shape_filtered.png"
)

cv2.imwrite(
    str(mask_path),
    final_mask
)


# --------------------------------------------------
# 9. Skeletonization
# --------------------------------------------------

skeleton = skeletonize_crack(
    final_mask
)


skeleton_path = (
    skeleton_dir
    / f"{name}_shape_skeleton.png"
)

cv2.imwrite(
    str(skeleton_path),
    skeleton
)


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\nProcessing completed successfully.")

print(
    f"Segmentation: {mask_path}"
)

print(
    f"Skeleton: {skeleton_path}"
)

print(
    "\nStage 3 single-image test PASSED."
)