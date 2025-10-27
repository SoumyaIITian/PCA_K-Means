# Unsupervised Learning Practice: Wine Dataset Clustering with PCA

This project is a hands-on notebook demonstrating a complete unsupervised learning workflow. It uses K-Means Clustering to find natural groupings in the Scikit-learn "Wine" dataset and Principal Component Analysis (PCA) to effectively visualize the results.

## Project Workflow

The notebook `unsupervised-practice-wine.ipynb` follows these key steps:

1. **Data Loading:** The built-in Wine dataset is loaded from `sklearn.datasets.load_wine`. This dataset contains 178 samples (wines) with 13 chemical features.

2. **Data Preprocessing:** The 13 features are scaled using `StandardScaler`. This is a crucial step because both K-Means and PCA are distance-based algorithms and are highly sensitive to features with different scales.

3. **Clustering Analysis (Elbow Method):**

   * To find the optimal number of clusters (K), the **Elbow Method** is implemented.

   * A loop runs `KMeans` for K=1 through K=10 on the scaled 13-feature data (`XScaler`).

   * The `inertia_` (sum of squared distances of samples to their closest cluster center) is plotted for each K.

   * The resulting graph clearly shows an "elbow" at **K=3**, indicating that 3 is the optimal number of clusters for this data.

4. **Dimensionality Reduction (PCA):**

   * To visualize the 13-dimensional data on a 2D plot, **Principal Component Analysis (PCA)** is used.

   * PCA is configured to reduce the 13 scaled features down to just **2 principal components** (`n_components=2`).

5. **Final Clustering & Visualization:**

   * The `KMeans` algorithm is run a final time with the optimal `n_clusters=3` on the 2-component PCA data (`X_pca_2`).

   * A scatter plot (`plt.scatter`) is generated, plotting Principal Component 1 vs. Principal Component 2.

   * The points are colored based on the 3 clusters discovered by K-Means.

## Result

The final visualization successfully separates the wines into three distinct clusters, demonstrating that the unsupervised K-Means algorithm was able to find the three underlying classes of wine present in the data without using any of the original labels.

## Key Techniques & Libraries

* **Algorithms:** K-Means Clustering, Principal Component Analysis (PCA)

* **Libraries:** Scikit-learn (`sklearn`), Matplotlib (`matplotlib`), NumPy
