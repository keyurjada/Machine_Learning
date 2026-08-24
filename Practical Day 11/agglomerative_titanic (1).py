import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# 1. Load dataset
df = pd.read_csv("D:/Laptop backup/Summer 2025/ML - MCA/Programs/titanic.csv")
print("Dataset shape:", df.shape)


# 2. Select numeric columns only
num_df = df.select_dtypes(include=[np.number])


# Handle missing values (fill with column mean)
num_df = num_df.fillna(num_df.mean())


# 3. Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)


# 4. Apply Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clust.fit_predict(scaled_data)


# Add cluster labels back to dataframe
df['Cluster'] = labels


# 5. Visualize using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)


plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=labels, cmap='viridis', s=50)
plt.title('Agglomerative Clustering on Titanic Dataset (PCA projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()


# 6. Generate dendrogram
linked = linkage(scaled_data, method='ward')


plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title('Dendrogram (Hierarchical Clustering)')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()


# Save clustered dataset
df.to_csv("titanic_clusters.csv", index=False)
print("Clustered dataset saved as 'titanic_clusters.csv'")
