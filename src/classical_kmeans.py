import os
import cv2
import numpy as np

def classical_kmeans_segment(input_path, output_path, k=5, beta=7.0, sigma_factor=1.2):
        # Read image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(input_path)

    if img.ndim == 2:
        img = img[..., None]

    H, W, C = img.shape

    # Normalize image
    img_norm = img.astype(np.float32) / 255.0

    # Convert to feature matrix
    X = img_norm.reshape(-1, C)

    N = X.shape[0]

    # KMeans++ initialization
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.1
    )

    _, _, centers = cv2.kmeans(
        X.astype(np.float32),
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    centroids = centers.astype(np.float32)

    sigma = None

    # QIKM refinement loop
    for it in range(50):

        diff = X[:, None, :] - centroids[None, :, :]
        dists_sq = np.sum(diff**2, axis=-1)

        if sigma is None:
            sigma = sigma_factor * np.sqrt(dists_sq.mean())

        amplitudes = np.exp(-dists_sq / (2 * sigma**2))
        interference = np.exp(-beta * np.sqrt(dists_sq + 1e-8))

        P = amplitudes * interference
        P /= P.sum(axis=1, keepdims=True) + 1e-10

        new_centroids = (P.T @ X) / (P.sum(axis=0)[:, None] + 1e-10)

        if np.max(np.abs(new_centroids - centroids)) < 1e-4:
            break

        centroids = new_centroids

    # Recompute probabilities using FINAL centroids
    diff = X[:, None, :] - centroids[None, :, :]
    dists_sq = np.sum(diff**2, axis=-1)

    amplitudes = np.exp(-dists_sq / (2 * sigma**2))
    interference = np.exp(-beta * np.sqrt(dists_sq + 1e-8))

    P = amplitudes * interference
    P **= 8
    P /= P.sum(axis=1, keepdims=True) + 1e-10

    labels = np.argmax(P, axis=1)

    label_img = labels.reshape(H, W).astype(np.uint8)

    # Spatial smoothing using median filter
    label_img = cv2.medianBlur(label_img, 5)

    # Reconstruct segmented image
    centroids_u8 = np.clip(centroids * 255, 0, 255).astype(np.uint8)

    result = centroids_u8[label_img.ravel()]
    result = result.reshape(H, W, C)

    # Save outputs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, result)

    label_path = os.path.splitext(output_path)[0] + "_labels.png"
    cv2.imwrite(label_path, label_img)

    print(f"[QIKM] Saved: {output_path}")

