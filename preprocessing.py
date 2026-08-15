import cv2


def load_image(path):
    """Load an image from the given path."""
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image


def to_grayscale(image):
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhance local contrast using CLAHE."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(gray_image)
import numpy as np


def apply_gabor_filter_bank(
    gray_image,
    orientations=None,
    wavelength=20.0,
    sigma=5.0,
    gamma=0.5
):
    """
    Valli — Multi-orientation Gabor Filter Bank.

    Enhances elongated crack-like structures while
    suppressing some pavement texture.

    The filter is applied at multiple orientations because
    pavement cracks can occur in different directions.
    """

    if orientations is None:
        orientations = [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ]

    enhanced_image = gray_image
    responses = []

    for theta in orientations:

        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=sigma,
            theta=theta,
            lambd=wavelength,
            gamma=gamma,
            psi=np.pi / 2,
            ktype=cv2.CV_32F
        )

        filtered = cv2.filter2D(
            enhanced_image,
            cv2.CV_32F,
            kernel
        )

        # We are interested in strong dark-line responses.
        dark_response = np.maximum(-filtered, 0)

        responses.append(dark_response)

    # Combine the strongest response from all orientations.
    gabor_response = np.max(
        np.array(responses),
        axis=0
    )

    # Normalize result to 0–255.
    gabor_response = cv2.normalize(
        gabor_response,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gabor_response.astype(np.uint8)
def apply_blackhat(gray_image, kernel_size=15):
    """
    Valli — Enhance dark crack-like structures.

    Black-hat morphology is useful for detecting dark
    structures against a brighter pavement background.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        gray_image,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    blackhat = cv2.normalize(
        blackhat,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return blackhat.astype(np.uint8)