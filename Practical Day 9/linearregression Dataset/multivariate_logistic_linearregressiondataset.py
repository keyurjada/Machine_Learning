# ==========================================================
# Linear Regression on Population-Profit Dataset
# ==========================================================

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------
# Step 1: Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("linearregressiondataset.csv")

print("First Five Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names")
print(df.columns)

# ----------------------------------------------------------
# Step 2: Separate Features and Target
# ----------------------------------------------------------
X = df[['Population']]
y = df['Profit']

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
print("\nR square Score :", r2_score(y_test, y_pred))

print("\nMean Absolute Error :", mean_absolute_error(y_test, y_pred))

print("\nMean Squared Error :", mean_squared_error(y_test, y_pred))

print("\nCoefficient :", model.coef_)

print("Intercept :", model.intercept_)

# ----------------------------------------------------------
# Step 7: Predict New Population
# ----------------------------------------------------------
sample = pd.DataFrame([[20.27]], columns=['Population'])

prediction = model.predict(sample)

print("\nPredicted Profit:", prediction[0])