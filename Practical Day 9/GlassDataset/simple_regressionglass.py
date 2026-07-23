import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("glass.csv")

# Features and Target
x = data.iloc[:, 0:9]
y = data.iloc[:, 9]

print(x)
print(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating and fitting the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Print the coefficients and intercept
print("Coefficients:")
print(model.coef_)
print("Intercept:")
print(model.intercept_)

# Predict for a new glass sample
sample = [[1.52101, 13.64, 4.49, 1.10, 71.78, 0.06, 8.75, 0.00, 0.00]]
sample = scaler.transform(sample)

prediction = model.predict(sample)

print("Predicted Glass Type:", prediction[0])

# Predict on test data
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))