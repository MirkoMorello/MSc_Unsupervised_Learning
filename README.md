# Unsupervised Anomaly Detection on Mixed-Type Medical Data

> **Course:** Unsupervised Learning

## Overview

This project explores and compares a suite of unsupervised anomaly detection algorithms on a high-dimensional medical dataset. A key challenge of the dataset was its mixed-type nature, containing both numerical and categorical (boolean) features. This required developing custom solutions to handle the data appropriately, as standard library implementations often fall short.

## Core Contributions

### 1. Custom Gower's Distance Implementation
Standard distance metrics like Euclidean are not suitable for mixed-type data. To address this, we implemented a flexible and efficient version of **Gower's distance**. Our implementation combines Euclidean distance for numerical features and Hamming distance for categorical features, providing a unified and meaningful proximity measure for all data points.

### 2. Anomaly Detection Algorithms
We implemented and evaluated a wide range of anomaly detection techniques:
*   **Proximity-Based:**
    *   k-Nearest Neighbour (k-NN) Distance
    *   Local Outlier Factor (LOF)
    *   Connectivity Outlier Factor (COF)
*   **Clustering-Based:**
    *   DBSCAN (treating non-clustered points as anomalies)
    *   K-Means (treating points far from their cluster centroid as anomalies)
*   **Reconstruction-Based:**
    *   Principal Component Analysis (PCA)
    *   Autoencoders

### 3. Ensemble Methods
To build a more robust detection system, we combined the outputs of the most promising individual models (K-Means, Autoencoder, k-NN) using three different **ensemble strategies**:
*   **AND Ensemble (Consensus):** Flags an anomaly only if all models agree.
*   **OR Ensemble (Union):** Flags an anomaly if at least one model agrees.
*   **Weighted Sum Ensemble:** Combines the normalized anomaly scores from each model to produce a final, more robust score.

## Technologies Used

*   **Machine Learning:** Python, Scikit-learn, PyTorch
*   **Data Handling:** NumPy, Pandas
*   **Visualization:** Matplotlib, Seaborn
