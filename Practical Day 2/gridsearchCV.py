from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = load_iris()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    random_state=0
)

# Parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

print("Parameter grid:")
print(param_grid)

# Grid Search
grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Results
print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation score: {:.3f}".format(
    grid_search.best_score_
))

print("\nBest estimator:")
print(grid_search.best_estimator_)

# Test score
test_score = grid_search.best_estimator_.score(X_test, y_test)
print("\nTest set score: {:.3f}".format(test_score))

# Convert results to DataFrame
results = pd.DataFrame(grid_search.cv_results_)

print("\nFirst 5 rows of CV results:")
print(results.head())

# Create score matrix for heatmap
scores = results['mean_test_score'].values.reshape(
    len(param_grid['C']),
    len(param_grid['gamma'])
)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    scores,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=param_grid['gamma'],
    yticklabels=param_grid['C']
)

plt.title("Grid Search Cross-Validation Accuracy")
plt.xlabel("gamma")
plt.ylabel("C")
plt.tight_layout()
plt.show()