# ==========================================================
# Linear Regression on Multivariate House Price Dataset
# ==========================================================

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------
# Step 1: Load Dataset
# ----------------------------------------------------------
df = pd.read_excel("Multivariate Linear Regression Dataset.xlsx")

print("First Five Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names")
print(df.columns)

# ----------------------------------------------------------
# Step 2: Separate Features and Target
# ----------------------------------------------------------
X = df.iloc[:, 0:2]      # Square Feet, Number of Bed Rooms
y = df.iloc[:, 2]        # Price of House

# ----------------------------------------------------------
# Step 3: Split Dataset
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# ----------------------------------------------------------
# Step 4: Build Linear Regression Model
# ----------------------------------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# ----------------------------------------------------------
# Step 5: Prediction
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# Step 6: Model Evaluation
# ----------------------------------------------------------
print("\nR2 Score :", r2_score(y_test, y_pred))

print("\nMean Absolute Error")
print(mean_absolute_error(y_test, y_pred))

print("\nMean Squared Error")
print(mean_squared_error(y_test, y_pred))

print("\nCoefficient")
print(model.coef_)

print("\nIntercept")
print(model.intercept_)

# ----------------------------------------------------------
# Step 7: Predict New House Price
# ----------------------------------------------------------
sample = pd.DataFrame([[2000, 3]],columns=X.columns)
prediction = model.predict(sample)

print("\nPredicted House Price:", prediction[0])