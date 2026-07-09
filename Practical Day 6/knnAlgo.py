import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split
df = pd.read_csv("spambase.csv")

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
max_k = len(X_train)
for k in range(1,max_k+1):
    classifier = KNeighborsClassifier(n_neighbors=k)

    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)*100
    print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')