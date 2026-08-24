import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 


home_data = pd.read_csv('D:/Winter 2023/MSc DS 2/Practicals/housing_dataset.csv', usecols = ['longitude', 'latitude', 'house_value'])
home_data.head()

sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'house_value')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['house_value']], test_size=0.33, random_state=0)

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
plt.show()

sns.boxplot(x = kmeans.labels_, y = y_train['house_value'])
plt.show()


print("silhouette score is",(silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')))
      
