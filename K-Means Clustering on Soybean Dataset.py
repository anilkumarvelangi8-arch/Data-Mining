import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("soybean.csv")

print("Dataset Preview:")
print(data.head())

encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

kmeans = KMeans(n_clusters=4)

kmeans.fit(data)

labels = kmeans.labels_

data['Cluster'] = labels

print("\nClustered Data:")
print(data.head())

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

print("\nCluster Distribution:")
print(data['Cluster'].value_counts())