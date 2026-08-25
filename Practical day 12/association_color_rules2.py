import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the uploaded colors dataset
data = pd.read_csv("colors.csv", header=None, names=["TransactionID", "Colors"])

# Convert comma-separated colors into transactions
dataset = data["Colors"].dropna().apply(
    lambda x: [item.strip() for item in str(x).split(",") if item.strip()]
).tolist()

te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.75
)

# Print the rules
for i in range(len(rules)):
    LHS = list(rules["antecedents"].iloc[i])
    RHS = list(rules["consequents"].iloc[i])
    support = rules["support"].iloc[i]
    confidence = rules["confidence"].iloc[i]

    print(f"LHS: {{LHS}} -- RHS: {{RHS}}")
    print(f"Support: {{support}}")
    print(f"Confidence: {{confidence}}")
    print("---")
