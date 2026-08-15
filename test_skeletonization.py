import cv2

from skeletonization import skeletonize_crack


# Jassi's filtered segmentation output
INPUT_PATH = "outputs/segmentation/crack2_filtered.png"

# Skeleton output
OUTPUT_PATH = "outputs/segmentation/crack2_skeleton.png"


# Load binary crack mask
binary_mask = cv2.imread(
    INPUT_PATH,
    cv2.IMREAD_GRAYSCALE
)

if binary_mask is None:
    raise FileNotFoundError(
        f"Could not load: {INPUT_PATH}"
    )


# Skeletonize the crack mask
skeleton = skeletonize_crack(
    binary_mask
)


# Save skeleton
cv2.imwrite(
    OUTPUT_PATH,
    skeleton
)


print("Skeletonization completed successfully.")
print("Skeleton output:", OUTPUT_PATH)