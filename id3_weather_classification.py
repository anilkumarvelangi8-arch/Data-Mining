import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("weather_nominal.csv")

X = data.drop("play", axis=1)
y = data["play"]

le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col])

y = le.fit_transform(y)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

tree_rules = export_text(model, feature_names=list(X.columns))

print("Decision Tree Rules")
print(tree_rules)