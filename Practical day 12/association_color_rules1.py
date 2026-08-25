import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the uploaded colors dataset
data = pd.read_csv("colors.csv", header=None, names=["TransactionID", "Colors"])

# Convert comma-separated colors into transactions
dataset = data["Colors"].dropna().apply(lambda x: [item.strip() for item in str(x).split(",") if item.strip()]).tolist()

# Step 1: Convert the dataset into a one-hot encoded DataFrame
te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

# Step 2: Apply the Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Step 3: Generate association rules
rules = association_rules(frequent_itemsets,metric="confidence",min_threshold=0.75)

# Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)