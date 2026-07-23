import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("glass.csv")

# Features and Target
X = df.drop("Type", axis=1)
y = df["Type"]

# Display first five records
print(df.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample = pd.DataFrame([[1.52101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00]],columns=X.columns)

prediction = model.predict(sample)
sample = scaler.transform(sample)

# prediction = model.predict(sample)

# print("Predicted Glass Type:", prediction[0])

print("\nPredicted Glass Type:", prediction[0])