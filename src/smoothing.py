import cv2
import numpy as np


def smooth_labels(label_img, kernel_size=5):
    """
    Apply spatial smoothing to segmentation labels.

    Parameters:
        label_img : 2D numpy array of cluster labels
        kernel_size : size of median filter

    Returns:
        smoothed label image
    """

    # Ensure correct datatype
    label_img = label_img.astype(np.uint8)

    # Median filtering removes noisy labels
    smoothed = cv2.medianBlur(label_img, kernel_size)

    return smoothed