import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_excel("Multivariate Linear Regression Dataset.xlsx")

# Features and Target
X = df[['Square Feet', 'Number of Bed Rooms']]
y = df['Price of House']
# Display first five records
print(df.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("\nR square Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Model parameters
print("\nCoefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Example prediction
sample = pd.DataFrame([[2000, 3]],columns=['Square Feet', 'Number of Bed Rooms'])

prediction = model.predict(sample)

print("\nPredicted House Price:", prediction[0])