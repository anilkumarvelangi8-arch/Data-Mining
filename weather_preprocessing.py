import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("weather_preprocessed.csv")

print("Dataset Preview")
print(data.head())

data = data.drop_duplicates()

data = data.fillna(data.mean(numeric_only=True))

X = data.drop("normalized_label", axis=1)
y = data["normalized_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

processed = pd.DataFrame(X_scaled, columns=X.columns)
processed["normalized_label"] = y

print("\nPreprocessed Data")
print(processed.head())

processed.to_csv("weather_preprocessed_output.csv", index=False)

print("\nPreprocessing Completed")