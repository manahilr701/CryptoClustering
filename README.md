# Crypto Clustering Project

## 📖 Overview
This project uses machine learning techniques to cluster cryptocurrencies based on their market performance. By identifying distinct clusters, the analysis provides insights into market trends and patterns within the cryptocurrency ecosystem.

## 🎯 Objectives
- **Normalize the Data**: Scale the data using `StandardScaler` for uniformity.
- **Optimal Clusters**: Use the elbow method to determine the best number of clusters (`k`) for K-Means clustering.
- **Cluster Cryptocurrencies**:
  - Perform K-Means clustering on normalized data.
  - Visualize clusters using scatter plots.
- **Optimize Clusters**:
  - Reduce dimensionality with Principal Component Analysis (PCA).
  - Reapply clustering on PCA-transformed data.
- **Compare Results**: Analyze and contrast clustering with and without PCA.

## 🛠️ Technologies Used
- **Python**
- **Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `scikit-learn`: Scaling, PCA, and K-Means clustering.
  - `matplotlib`: Visualization.
  - `hvPlot`: Interactive plots (if supported).

## 📁 Files
- **`Crypto_Clustering.ipynb`**: Jupyter Notebook with the project code.
- **`crypto_market_data.csv`**: Dataset containing cryptocurrency market performance metrics.

## 🚀 Process

### 1️⃣ Normalize the Data
- Used `StandardScaler` to normalize numeric features, ensuring all columns have a mean of 0 and standard deviation of 1.

### 2️⃣ Determine Optimal Clusters
- Applied the **Elbow Method** to calculate inertia for `k` values ranging from 1 to 11.
- Plotted an elbow curve to determine the optimal `k` value visually.

### 3️⃣ K-Means Clustering
- Performed K-Means clustering using the optimal `k`.
- Created scatter plots to visualize clusters using the features:
  - `price_change_percentage_24h`
  - `price_change_percentage_7d`

### 4️⃣ Optimize Clustering with PCA
- Reduced the dataset's dimensionality to three principal components while retaining **88.86%** of the variance.
- Reapplied K-Means clustering to the PCA-transformed data.
- Visualized clusters using the first two principal components (PC1 and PC2).

### 5️⃣ Compare Results
- Compared the clustering results (with and without PCA) using composite plots.
- Observed the impact of dimensionality reduction on clustering quality.

## 📊 Key Findings
- **Optimal Clusters**: The elbow method identified the ideal number of clusters.
- **Impact of PCA**: PCA simplified the dataset while retaining most variance, resulting in clearer and more distinct clusters.
- **Cluster Comparison**: Clusters from PCA-reduced data were better defined and less scattered than clusters from the original dataset.

## 📈 Visualizations
- **Elbow Curves**: Plots showing the inertia values for different `k` values to determine the optimal cluster count.
- **Scatter Plots**:
  - Without PCA: Clusters plotted with `price_change_percentage_24h` and `price_change_percentage_7d`.
  - With PCA: Clusters plotted with principal components PC1 and PC2.

## 📝 Conclusion
Combining K-Means clustering with PCA for dimensionality reduction enhances the clustering process. PCA allows for better visualization and clearer distinction between clusters, making it a valuable tool for large datasets with multiple features.

## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/manahilr701/CryptoClustering.git
  
### Navigate to the project directory:
```bash
cd CryptoClustering
```
### Run all cells in the notebook to reproduce the analysis.

## 📂 Directory Structure
```plaintext
├── Crypto_Clustering.ipynb  # Jupyter Notebook with project code
├── crypto_market_data.csv    # Dataset used for clustering
└── README.md                 # Project documentation
```
##  🎓 Acknowledgments
- **Instructor**: For guidance on machine learning techniques.
- **Dataset Source**: Provided as part of the assignment.

