# import os
# import cv2
# import numpy as np
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import davies_bouldin_score
def compare_segmentation(input_image_path, classical_label_path, qikm_label_path):
    print("DEBUG: compare_segmentation STARTED")
    print("  Input image path:", input_image_path)
    print("  Classical label:", classical_label_path)
    print("  QIKM label     :", qikm_label_path)

    try:
        classical_labels = cv2.imread(classical_label_path, cv2.IMREAD_GRAYSCALE)
        print("  Classical labels loaded shape:", classical_labels.shape if classical_labels is not None else "None")

        qikm_labels = cv2.imread(qikm_label_path, cv2.IMREAD_GRAYSCALE)
        print("  QIKM labels loaded shape:", qikm_labels.shape if qikm_labels is not None else "None")

        if classical_labels is None or qikm_labels is None:
            print("ERROR: at least one label map failed to load")
            return

        classical_labels = classical_labels.flatten()
        qikm_labels = qikm_labels.flatten()

        print("  Classical unique labels:", np.unique(classical_labels))
        print("  QIKM unique labels     :", np.unique(qikm_labels))

        img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("ERROR: original image failed to load")
            return

        if img.ndim == 2:
            img = img[..., None]

        img = img.astype(np.float32) / 255.0
        X = img.reshape(-1, img.shape[-1])

        n_samples = min(15000, len(X))
        idx = np.random.choice(len(X), n_samples, replace=False)

        X_sample = X[idx]
        classical_sample = classical_labels[idx]
        qikm_sample = qikm_labels[idx]

        sil_classical = silhouette_score(X_sample, classical_sample)
        sil_qikm = silhouette_score(X_sample, qikm_sample)

        db_classical = davies_bouldin_score(X_sample, classical_sample)
        db_qikm = davies_bouldin_score(X_sample, qikm_sample)

        print("\n" + "=" * 60)
        print("SEGMENTATION QUALITY COMPARISON")
        print("=" * 60)
        print(f"Silhouette Score (QIKM): {sil_qikm:.4f}")
        print(f"Silhouette Score (Classical KMeans): {sil_classical:.4f}")
        print(f"Davies-Bouldin Index (QIKM): {db_qikm:.4f}")
        print(f"Davies-Bouldin Index (Classical KMeans): {db_classical:.4f}")
        print("=" * 60 + "\n")

    except Exception as e:
        print("!!! EXCEPTION in compare_segmentation !!!")
        print(str(e))
        import traceback
        traceback.print_exc()

# def compare_segmentation(input_image_path, classical_label_path, qikm_label_path):
#     # Load labels
#     classical_labels = cv2.imread(classical_label_path, 0)
#     qikm_labels = cv2.imread(qikm_label_path, 0)
#     if classical_labels is None or qikm_labels is None:
#         print("Error loading label maps")
#         return
#     classical_labels = classical_labels.flatten().astype(int)
#     qikm_labels = qikm_labels.flatten().astype(int)
#     # Load original image
#     img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
#     if img.ndim == 2:
#         img = img[..., None]
#     img = img.astype(np.float32) / 255.0
#     X = img.reshape(-1, img.shape[2])
#     # Use subset of pixels for faster evaluation
#     n_samples = min(15000, len(X))
#     idx = np.random.choice(len(X), n_samples, replace=False)
#     X_sample = X[idx]
#     classical_sample = classical_labels[idx]
#     qikm_sample = qikm_labels[idx]
#     # Silhouette Score
#     sil_classical = silhouette_score(X_sample, qikm_sample)
#     sil_qikm = silhouette_score(X_sample, classical_sample)
#     # Davies-Bouldin Index
#     db_classical = davies_bouldin_score(X_sample, qikm_sample) 
#     db_qikm =davies_bouldin_score(X_sample, classical_sample)
#     print("\n" + "=" * 60)
#     print("SEGMENTATION QUALITY COMPARISON")
#     print("=" * 60)
#     print(f"Silhouette Score (QIKM): {sil_classical:.4f}")
#     print(f"Silhouette Score (Classical KMeans):             {sil_qikm:.4f}")
#     print(f"Davies-Bouldin Index (QIKM):    {db_classical:.4f}")
#     print(f"Davies-Bouldin Index (Classical KMeans):         {db_qikm:.4f}")
#     print("\nWinner (Silhouette):",
#           "QIKM" if sil_qikm > sil_classical else "Classical KMeans")
#     print("Winner (Davies-Bouldin):",
#           "QIKM" if db_qikm < db_classical else "Classical KMeans")
#     print("=" * 60)
# compare.py

import cv2
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def compare_segmentation(input_image_path, classical_label_path, qikm_label_path):
    """
    Compute Silhouette and Davies-Bouldin scores and print them to console.
    """
    # ─── Load label maps ────────────────────────────────────────────────
    classical_labels = cv2.imread(classical_label_path, cv2.IMREAD_GRAYSCALE)
    qikm_labels      = cv2.imread(qikm_label_path,      cv2.IMREAD_GRAYSCALE)

    if classical_labels is None or qikm_labels is None:
        print("Error: could not load one or both label maps")
        print("  Classical labels:", classical_label_path)
        print("  QIKM labels     :", qikm_label_path)
        return

    classical_labels = classical_labels.flatten()
    qikm_labels      = qikm_labels.flatten()

    # ─── Load original image as feature space ───────────────────────────
    img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: cannot read original image")
        print("  Path:", input_image_path)
        return

    if img.ndim == 2:
        img = img[..., None]

    img = img.astype(np.float32) / 255.0
    X = img.reshape(-1, img.shape[-1])

    # ─── Subsample for speed ────────────────────────────────────────────
    n_samples = min(15000, len(X))
    if n_samples < 300:
        print(f"Warning: too few pixels ({n_samples}) for reliable metrics")
        return

    rng = np.random.default_rng(42)  # reproducible results
    idx = rng.choice(len(X), n_samples, replace=False)

    X_sample         = X[idx]
    classical_sample = classical_labels[idx]
    qikm_sample      = qikm_labels[idx]

    # ─── Compute metrics ────────────────────────────────────────────────
    try:
        sil_classical = silhouette_score(X_sample, classical_sample)
        sil_qikm      = silhouette_score(X_sample, qikm_sample)

        db_classical  = davies_bouldin_score(X_sample, classical_sample)
        db_qikm       = davies_bouldin_score(X_sample, qikm_sample)
    except Exception as e:
        print("Error computing metrics:")
        print(str(e))
        print("(common causes: all pixels same label, only 1 cluster, degenerate data)")
        return

    # ─── Print to console ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SEGMENTATION QUALITY COMPARISON")
    print("=" * 65)
    print(f"Silhouette Score (higher = better)")
    print(f"  Classical KMeans : {sil_classical:8.4f}")
    print(f"  QIKM             : {sil_qikm:8.4f}")
    print()
    print(f"Davies-Bouldin Index (lower = better)")
    print(f"  Classical KMeans : {db_classical:8.4f}")
    print(f"  QIKM             : {db_qikm:8.4f}")
    print("-" * 65)

    print("Silhouette winner   :", "QIKM" if sil_qikm > sil_classical else "Classical KMeans")
    print("Davies-Bouldin winner:", "QIKM" if db_qikm < db_classical else "Classical KMeans")
    print("=" * 65 + "\n")