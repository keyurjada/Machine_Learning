# ==========================================================
# Multinomial Logistic Regression on Glass Dataset
# ==========================================================

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------------------------
# Step 1: Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("glass.csv")

print("First Five Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names")
print(df.columns)

# ----------------------------------------------------------
# Step 2: Remove Id column (if present)
# ----------------------------------------------------------
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# ----------------------------------------------------------
# Step 3: Separate Features and Target
# ----------------------------------------------------------
X = df.drop('Type', axis=1)
y = df['Type']

# ----------------------------------------------------------
# Step 4: Split Dataset
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# Step 5: Standardize Features
# ----------------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# Step 6: Build Multinomial Logistic Regression Model
# ----------------------------------------------------------
#model = LogisticRegression(multi_class='auto',
#    solver='lbfgs',
#    max_iter=1000,
#    random_state=42
#)

model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------------------------------
# Step 7: Prediction
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# Step 8: Model Evaluation
# ----------------------------------------------------------
print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# Step 9: Predict New Glass Sample
# ----------------------------------------------------------
sample = [[1.518,13.6,3.5,1.3,72.7,0.5,8.7,0.0,0.1]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("\nPredicted Glass Type:", prediction[0])

# ----------------------------------------------------------
# Step 10: Probability of Each Class
# ----------------------------------------------------------
probability = model.predict_proba(sample)

print("\nPrediction Probabilities")
print(probability)