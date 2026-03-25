import os
import cv2
import numpy as np

def qikm_segment(input_path, output_path, k=5):

    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    # Convert grayscale → 3D
    if img.ndim == 2:
        img = img[..., None]

    H, W, C = img.shape

    # Normalize image
    data = img.astype(np.float32) / 255.0

    # Convert image to feature matrix
    X = data.reshape((-1, C))

    # KMeans parameters
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Apply KMeans
    _, labels, centers = cv2.kmeans(
        X,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    # Convert centers back to image colors
    centers = np.clip(centers * 255, 0, 255).astype(np.uint8)

    segmented = centers[labels.flatten()]
    segmented = segmented.reshape((H, W, C))

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, segmented)

    label_path = os.path.splitext(output_path)[0] + "_labels.png"
    cv2.imwrite(label_path, labels.reshape(H, W).astype(np.uint8))

    print(f"[QIKM] Saved: {output_path}")



