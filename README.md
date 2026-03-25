# Quantum-Inspired K-Means for Image Segmentation

This project implements a **Quantum-Inspired K-Means (QIKM)** algorithm for image segmentation and compares its performance with **Classical K-Means clustering**.

The algorithm incorporates ideas inspired by **quantum probability and wave interference** to improve clustering quality and produce more stable segmentation compared to traditional K-Means.

---

## Project Overview

Image segmentation is a fundamental task in computer vision where an image is divided into meaningful regions. Classical K-Means clustering is widely used for segmentation but often struggles with:

- Noise in images  
- Weak object boundaries  
- Overlapping color distributions  

The **Quantum-Inspired K-Means (QIKM)** algorithm improves segmentation by introducing:

- Probabilistic clustering
- Wave interference modeling
- Spatial smoothing for label consistency

---

## Algorithm Pipeline
Input Image
↓
Preprocessing / Normalization
↓
KMeans++ Initialization
↓
Quantum Probability Model
↓
Interference Weighting
↓
Centroid Update (Iterative)
↓
Spatial Smoothing
↓
Segmented Image
↓
Evaluation Metrics


---

## Features

- Modular project architecture
- Classical K-Means segmentation
- Quantum-Inspired clustering algorithm
- Robust image preprocessing
- Spatial smoothing using median filtering
- Segmentation visualization
- Quantitative evaluation metrics

---

## Project Structure


project/
│
├── main.py
├── compare.py
├── README.md
│
├── src/
│ ├── preprocessing.py
│ ├── classical_kmeans.py
│ ├── qikm.py
│ ├── smoothing.py
│ └── visualization.py
│
├── data/
│ └── input/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
└── results/


---

## Requirements

Install the required Python libraries:
pip install numpy opencv-python matplotlib scikit-learn


---

## How to Run

1. Place test images in:
data/input/

2. Run the main script:
python main.py


3. The program will:

- Perform segmentation using **Classical K-Means**
- Perform segmentation using **Quantum-Inspired K-Means**
- Display visual comparison of results
- Compute clustering evaluation metrics

---

## Output

Segmented images will be saved in:
results/


Example output files:
classical_image1.jpg
classical_image1_labels.png
qikm_image1.jpg
qikm_image1_labels.png


---

## Evaluation Metrics

The performance of the algorithms is evaluated using:

- **Silhouette Score** (higher values indicate better clustering)
- **Davies–Bouldin Index** (lower values indicate better clustering)

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Applications

The segmentation approach can be applied to:

- Medical image analysis
- Satellite and remote sensing imagery
- Object detection preprocessing
- Computer vision research
- Pattern recognition

---

## Author

N.TirypatiRao and team
Undergraduate Project – Quantum-Inspired Image Segmentation
