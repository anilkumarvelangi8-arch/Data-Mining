import pandas as pd

data = pd.read_csv("data.csv")

print("Extracted Data:")
print(data.head())

data = data.drop_duplicates()

data = data.fillna(method='ffill')

data['Date'] = pd.to_datetime(data['Date'])

data = data[data['Status'] == 'Completed']

summary = data.groupby('Category')['Amount'].sum()

print("\nTransformed Data:")
print(data.head())

print("\nAggregated Result:")
print(summary)

data.to_csv("processed_data.csv", index=False)

print("\nData Loaded Successfully")