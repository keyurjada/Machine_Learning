import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
data = pd.read_excel("Multivariate Linear Regression Dataset.xlsx")

# Features (Square Feet, Number of Bed Rooms)
x = data.iloc[:, 0:2]

# Target (Price of House)
y = data.iloc[:, 2]

print(x)
print(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Creating and fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict price for a house with 2000 sq.ft and 3 bedrooms
sample = pd.DataFrame([[2000, 3]], columns=x.columns)

y_pred = model.predict(sample)

print("Predicted House Price:", y_pred)

# Predict on test data
y_pred = model.predict(X_test)

print("Predicted Prices:")
print(y_pred)

# Model evaluation
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("R2 Score:", metrics.r2_score(y_test, y_pred))