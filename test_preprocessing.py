import cv2

from preprocessing import (
    load_image,
    to_grayscale,
    apply_clahe,
    apply_blackhat,
    apply_gabor_filter_bank
)


IMAGE_PATH = "data/samples/crack2.jpg"


# Load original image
image = load_image(IMAGE_PATH)

# Convert to grayscale
gray = to_grayscale(image)

# Apply CLAHE
clahe = apply_clahe(gray)

blackhat = apply_blackhat(clahe)

#Apply multi-orientation Gabor filtering
gabor = apply_gabor_filter_bank(clahe)

# Save outputs
cv2.imwrite("outputs/clahe/crack2_gray.png", gray)
cv2.imwrite("outputs/clahe/crack2_clahe.png", clahe)
cv2.imwrite("outputs/gabor/crack2_blackhat.png", blackhat)
cv2.imwrite("outputs/gabor/crack2_gabor.png",gabor)


# Display basic information
print("Original image shape:", image.shape)
print("Grayscale image shape:", gray.shape)
print("CLAHE completed successfully.")