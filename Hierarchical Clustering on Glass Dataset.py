import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("glass.csv")

print("Dataset Preview:")
print(data.head())

encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

model = AgglomerativeClustering(n_clusters=3, linkage='average')

labels = model.fit_predict(data)

data['Cluster'] = labels

print("\nClustered Data:")
print(data.head())

print("\nCluster Distribution:")
print(data['Cluster'].value_counts())