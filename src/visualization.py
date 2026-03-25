# 


import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os


def show_segmentation_results(original_path, classical_path, qikm_path):
    """
    Show original + two segmented images side-by-side,
    and display Silhouette & Davies-Bouldin scores below them.
    """
    # ─── Load the result images ────────────────────────────────────────
    orig_img     = cv2.imread(original_path)
    classical_img = cv2.imread(classical_path)
    qikm_img     = cv2.imread(qikm_path)

    if orig_img is None or classical_img is None or qikm_img is None:
        print("Error: could not load one or more output images for visualization")
        return

    # BGR → RGB for correct colors in matplotlib
    orig     = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    classical = cv2.cvtColor(classical_img, cv2.COLOR_BGR2RGB)
    qikm     = cv2.cvtColor(qikm_img, cv2.COLOR_BGR2RGB)

    # ─── Try to load label maps and compute metrics ─────────────────────
    metrics_text = "Metrics could not be calculated"

    try:
        # Label paths (must match exactly what your segmentation functions save)
        classical_labels_path = os.path.splitext(classical_path)[0] + "_labels.png"
        qikm_labels_path      = os.path.splitext(qikm_path)[0] + "_labels.png"

        cla_labels = cv2.imread(classical_labels_path, cv2.IMREAD_GRAYSCALE)
        qik_labels = cv2.imread(qikm_labels_path,      cv2.IMREAD_GRAYSCALE)

        if cla_labels is None or qik_labels is None:
            metrics_text = "Label maps not found → no metrics"
        else:
            # Flatten labels
            cla_flat = cla_labels.flatten()
            qik_flat = qik_labels.flatten()

            # Use pixel colors from original image as features
            h, w, _ = orig_img.shape
            X = orig_img.reshape(-1, 3).astype(np.float32) / 255.0

            # Subsample to avoid memory/time issues
            n_samples = min(15000, len(X))
            if n_samples < 500:
                metrics_text = "Image too small for reliable metrics"
            else:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X), n_samples, replace=False)

                X_s = X[idx]
                cla_s = cla_flat[idx]
                qik_s = qik_flat[idx]

                # Compute scores
                sil_cla = silhouette_score(X_s, cla_s)
                sil_qik = silhouette_score(X_s, qik_s)
                db_cla  = davies_bouldin_score(X_s, cla_s)
                db_qik  = davies_bouldin_score(X_s, qik_s)

                # Format nice text
                metrics_text = (
                    f"Silhouette score (higher = better):\n"
                    f"  Classical K-Means → {sil_cla:>6.4f}\n"
                    f"  Quantum-Inspired  → {sil_qik:>6.4f}\n\n"
                    f"Davies-Bouldin index (lower = better):\n"
                    f"  Classical K-Means → {db_cla:>6.4f}\n"
                    f"  Quantum-Inspired  → {db_qik:>6.4f}\n\n"
                    f"Silhouette winner: {'QIKM' if sil_qik > sil_cla else 'Classical'}\n"
                    f"DB winner:         {'QIKM' if db_qik < db_cla else 'Classical'}"
                )

    except Exception as e:
        metrics_text = f"Metrics calculation failed:\n{str(e)}"

    # ─── Create the plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Segmentation Comparison", fontsize=16, y=0.98)

    axes[0].imshow(orig)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(classical)
    axes[1].set_title("Classical K-Means", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(qikm)
    axes[2].set_title("Quantum-Inspired K-Means", fontsize=13)
    axes[2].axis("off")

    # Add metrics text at the bottom
    fig.text(0.5, 0.02, metrics_text,
             ha='center', va='bottom',
             fontsize=11, family='monospace',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.10, 1, 0.95])  # leave space at bottom for text
    plt.show()

    # Optional: save the figure with metrics
    # save_path = os.path.join(os.path.dirname(original_path), "comparison_with_metrics.png")
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # print(f"Saved visualization with metrics: {save_path}")