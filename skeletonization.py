import cv2
import numpy as np
from skimage.morphology import skeletonize


def skeletonize_crack(binary_mask):
    """
    Jassi — Convert the binary crack mask into a
    one-pixel-wide skeleton.

    The skeleton preserves the shape and connectivity
    of the detected crack structure.
    """

    # Convert 0/255 image to boolean
    binary = binary_mask > 0

    # Skeletonization
    skeleton = skeletonize(binary)

    # Convert back to 0/255
    skeleton_image = (
        skeleton.astype(np.uint8) * 255
    )

    return skeleton_image