import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=pd.read_csv("linearregressiondataset.csv")

x=data.iloc[:,0:1]
y=data.iloc[:,1]

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Creating and fitting the model
model = LinearRegression()
model.fit(X_train,y_train)

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


y_pred=model.predict([[20.27]])
print('profit for the population of 20.27 lakh is:',y_pred)

y_pred = model.predict(X_test)

print("predicted salary is:",y_pred)

# Model evaluation 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))