# Importing necessary libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load breast cancer dataset
df = pd.read_csv("spambase.csv")
df.fillna(df.mean(), inplace=True)
# print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Splitting data into features and target
X = df.drop('spam', axis=1)
y = df['spam']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing Logistic Regression model
log_reg_model = LogisticRegression()

# Training the model
log_reg_model.fit(X_train, y_train)

# Making predictions
y_pred = log_reg_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
