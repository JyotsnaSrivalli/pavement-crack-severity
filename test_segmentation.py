import cv2

from segmentation import (
    apply_otsu_threshold,
    apply_strong_response_threshold,
    clean_binary_mask,
    remove_small_components
)


INPUT_PATH = "outputs/gabor/crack2_gabor.png"

OTSU_OUTPUT = "outputs/segmentation/crack2_otsu.png"
STRONG_OUTPUT = "outputs/segmentation/crack2_strong.png"
CLEAN_OUTPUT = "outputs/segmentation/crack2_clean.png"
FINAL_OUTPUT = "outputs/segmentation/crack2_filtered.png"


gabor_image = cv2.imread(
    INPUT_PATH,
    cv2.IMREAD_GRAYSCALE
)

if gabor_image is None:
    raise FileNotFoundError(
        f"Could not load: {INPUT_PATH}"
    )


# Otsu baseline
otsu_mask = apply_otsu_threshold(
    gabor_image
)

cv2.imwrite(
    OTSU_OUTPUT,
    otsu_mask
)


# Strong-response threshold
strong_mask = apply_strong_response_threshold(
    gabor_image,
    percentile=85
)

cv2.imwrite(
    STRONG_OUTPUT,
    strong_mask
)


# Morphological cleaning
cleaned_mask = clean_binary_mask(
    strong_mask
)

cv2.imwrite(
    CLEAN_OUTPUT,
    cleaned_mask
)


# Remove small components
filtered_mask = remove_small_components(
    cleaned_mask,
    min_area=300
)

cv2.imwrite(
    FINAL_OUTPUT,
    filtered_mask
)


print("Segmentation completed successfully.")
print("Final output:", FINAL_OUTPUT)