# Required Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data.data

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Medoids clustering
k = 2  # Number of clusters
kmedoids = KMedoids(n_clusters=k, random_state=0)
kmedoids.fit(X_scaled)

# Predicted cluster labels
labels = kmedoids.labels_

# Silhouette score for evaluation
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil_score}")

# Visualize the clustering results using the first two features
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title('K-Medoids Clustering on Breast Cancer Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()