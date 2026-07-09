import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
df = pd.read_csv("spambase.csv")
df.fillna(df.mean(), inplace=True)
# print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
# Features and Target
X = df.drop("spam", axis=1)
y = df["spam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)