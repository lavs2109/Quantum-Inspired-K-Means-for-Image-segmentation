# import os
# from src.classical_kmeans import classical_kmeans_segment
# from src.qikm import qikm_segment
# from src.visualization import show_segmentation_results
# from compare import compare_segmentation


# def main():

#     base = os.path.dirname(__file__)

#     input_dir = os.path.join(base, "data", "input")
#     results_dir = os.path.join(base, "results")

#     os.makedirs(results_dir, exist_ok=True)

#     # Supported image formats
#     valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#     images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

#     if not images:
#         print("No images found in data/input/")
#         return

#     for i, img_name in enumerate(images, 1):

#         print(f"\n[{i}/{len(images)}] Processing: {img_name}")

#         input_path = os.path.join(input_dir, img_name)

#         classical_output = os.path.join(results_dir, f"classical_{img_name}")
#         qikm_output = os.path.join(results_dir, f"qikm_{img_name}")

#         # Run Classical KMeans
#         classical_kmeans_segment(
       
#             input_path,
#            classical_output,
#             k=5,
#             beta=7.0,
#             sigma_factor=1.2
#         )

#         # Run Quantum Inspired KMeans
#         qikm_segment(
#             input_path,
#              qikm_output,
#             k=5

#         )

#         # Visualization
#         show_segmentation_results(
#             input_path,
#             classical_output,
#             qikm_output
#         )

#         # Label paths
#         classical_labels = os.path.splitext(classical_output)[0] + "_labels.png"
#         qikm_labels = os.path.splitext(qikm_output)[0] + "_labels.png"

#         # Compare results
#         compare_segmentation(
#             input_path,
#             classical_labels,
#             qikm_labels
#         )

#     print("\nAll images processed successfully!")


# if __name__ == "__main__":
#     main()

# main.py

import os
from src.classical_kmeans import classical_kmeans_segment
from src.qikm import qikm_segment
from src.visualization import show_segmentation_results
from compare import compare_segmentation


def main():
    base = os.path.dirname(__file__)

    input_dir = os.path.join(base, "data", "input")
    results_dir = os.path.join(base, "results")

    os.makedirs(results_dir, exist_ok=True)

    # Supported image formats
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    if not images:
        print("No images found in data/input/")
        return



    for i, img_name in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_name}")

        input_path = os.path.join(input_dir, img_name)

        classical_output = os.path.join(results_dir, f"classical_{img_name}")
        qikm_output      = os.path.join(results_dir, f"qikm_{img_name}")

        # Run Classical KMeans
        classical_kmeans_segment(
            input_path,
            classical_output,
            k=5,
            beta=7.0,
            sigma_factor=1.2
        )

        # Run Quantum Inspired KMeans
        qikm_segment(
            input_path,
            qikm_output,
            k=5
        )

        # Visualization
        show_segmentation_results(
            input_path,
            classical_output,
            qikm_output
        )

        # Label paths (these must match exactly what your segmentation functions save)
        classical_labels = os.path.splitext(classical_output)[0] + "_labels.png"
        qikm_labels      = os.path.splitext(qikm_output)[0] + "_labels.png"

        print("→ Comparing results for:", img_name)
        print("  Classical labels path:", classical_labels)
        print("  QIKM      labels path:", qikm_labels)

        # Compare results → this should print the metrics to console
        compare_segmentation(
            input_path,
            classical_labels,
            qikm_labels
        )

    print("\nAll images processed successfully!")


if __name__ == "__main__":
    main()