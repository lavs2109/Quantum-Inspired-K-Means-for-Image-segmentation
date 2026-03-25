import cv2
import numpy as np


def load_image(image_path):
    """
    Load image from disk.
    Supports grayscale and color images.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # If grayscale convert to 3D format
    if img.ndim == 2:
        img = img[..., None]

    return img


def normalize_image(img):
    """
    Normalize image using 2–98 percentile normalization
    to remove extreme outliers.
    """

    img_norm = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[2]):
        p2, p98 = np.percentile(img[..., c], (2, 98))

        # Clip extreme values
        clipped = np.clip(img[..., c], p2, p98)

        # Normalize
        img_norm[..., c] = (clipped - p2) / (p98 - p2 + 1e-6)

    return img_norm


def reshape_features(img_norm):
    """
    Convert image into feature matrix for clustering.
    """
    H, W, C = img_norm.shape

    X = img_norm.reshape(-1, C)

    return X, H, W, C