import pandas as pd
import sklearn.model_selection as ms
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("titanic.csv")

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin because it has many missing values
df.drop("Cabin", axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop unnecessary columns
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split data
x_train, x_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
dtc = tree.DecisionTreeClassifier(random_state=42)
dtc.fit(x_train, y_train)

# Prediction
y_predict = dtc.predict(x_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("\nClassification Report:\n")
print(classification_report(y_test, y_predict))