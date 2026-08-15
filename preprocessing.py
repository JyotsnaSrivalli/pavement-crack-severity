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