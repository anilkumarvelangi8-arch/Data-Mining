import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Preview:")
print(X.head())

print("\nMissing Values:")
print(X.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

print("\nCluster Labels:")
print(kmeans.labels_[:10])

print("\nKDD Process Completed Successfully")