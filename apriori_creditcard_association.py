import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("supermarket.csv")

frequent_items = apriori(data, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)

print("Frequent Itemsets")
print(frequent_items)

print("\nAssociation Rules")
print(rules[['antecedents','consequents','support','confidence','lift']])