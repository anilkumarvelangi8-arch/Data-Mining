import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

print("Statistical Description")
print(X.describe())

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

print("\nCluster Centers")
print(kmeans.cluster_centers_)

dist = euclidean_distances(X)

print("\nSimilarity / Dissimilarity Matrix (First 5 Rows)")
print(dist[:5, :5])